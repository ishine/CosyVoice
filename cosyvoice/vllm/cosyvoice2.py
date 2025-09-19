# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import time
import logging
import torch
from typing import Dict, List, Optional
from vllm.model_executor.models.qwen2 import *
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.sequence import Logprob
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from flashinfer.sampling import top_k_top_p_sampling_from_probs

logger = logging.getLogger(__name__)


class CosyVoice2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              True,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.sampler = Sampler()   # 注释这行，则会使用vllm原始采样
        if hasattr(self, "sampler"):
            logger.info(f"{self} use user-define sampler.")
        else:
            logger.info(f"{self} use vllm sampler.")
        self.codebooknum = 1
        self.codec_id_max = 6561
        self.eosid = self.codec_id_max
        self.generated_tokens = {}   # cache of each decoded token sequence
        self.running_reqs = {}
        self.temperature = 1.0
        self.topp = 0.8
        self.topk = 5
        self.window = 10  # repeat aware sampling window
        self.early_stop_check = False

    def get_input_embeddings(self, input_ids: torch.Tensor, request_info=None) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    @torch.amp.autocast('cuda',torch.float32)
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata, self.lm_head.bias)
        if torch.isnan(logits).any():
            logger.error(f"logits has nan !!! which is abnormal."
                         f" \nhidden_states: {hidden_states}, \nlogits {logits}")
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    @torch.amp.autocast('cuda',torch.float32)
    def sample(self, logits, sampling_metadata, request_info):
        request_info = list(request_info.keys())
        # clean regularly
        curtime = time.time()
        for req_id in request_info:
            if req_id not in self.running_reqs:
                self.running_reqs[req_id] = curtime
            if req_id not in self.generated_tokens:
                self.generated_tokens[req_id] = torch.tensor([], dtype=torch.int32, device=logits.device)   # add null token sequence

        # vllm采样，得到输出，注意这里是全batch运算，不去掉nan
        output = self.sampler(logits, sampling_metadata)   # vllm sample
        sample_ids = torch.tensor([output.outputs[idx].samples[0].output_token for idx in range(len(request_info))],
                                  dtype=torch.int32, device=logits.device)  # B,

        # nan check in logits
        B = logits.size(0)
        device = logits.device
        is_nan_mask = torch.isnan(logits).any(dim=-1)   # B
        non_nan_logits = logits[~is_nan_mask]   # 非nan的logits可以进入后续采样，nan的logits直接置为eos
        non_nan_request = [request_info[ii] for ii in range(B) if not is_nan_mask[ii]]

        # sampling, for non nan request
        if non_nan_logits.size(0) > 0:
            sample_ids = self.ras_sample(non_nan_logits, non_nan_request)

            # Optional: check early stop
            if self.early_stop_check:
                for batch_idx in range(sample_ids.shape[0]):
                    for retry in range(20):
                        end = (sample_ids[batch_idx] == self.eosid).all()
                        if end and self.generated_tokens[
                            non_nan_request[batch_idx]].shape[0] < 15:
                            resample = self.ras_sample(
                                non_nan_logits[batch_idx:batch_idx + 1],
                                [non_nan_request[batch_idx]], retry)
                            sample_ids[batch_idx:batch_idx + 1] = resample
                        else:
                            break

        # 将logits中有logits的采样结果直接置为eosid，加入sample_ids
        new_sample_ids = torch.tensor([self.eosid,]*B, dtype=torch.int32, device=device)
        non_nan_idx = 0
        for idx in range(B):
            if not is_nan_mask[idx]:
                new_sample_ids[idx] = sample_ids[non_nan_idx]
                non_nan_idx += 1
        sample_ids = new_sample_ids

        # add new ids, for all request
        for idx, (sample_id, req_id) in enumerate(
                zip(sample_ids, request_info)):
            self.generated_tokens[req_id] = torch.cat(
                [self.generated_tokens[req_id],
                 torch.tensor([sample_id], dtype=torch.int32, device=sample_id.device)], dim=0)
            self.generated_tokens[req_id] = self.generated_tokens[req_id][-20:]
            output.outputs[idx].samples[0].output_token = sample_id
            output.outputs[idx].samples[0].logprobs = {sample_id: Logprob(float('inf'))}   # logprob貌似没有用，但是需要，设为inf

        output.sampled_token_ids = sample_ids.unsqueeze(1)  # B,1

        for idx, req_id in enumerate(request_info):
            if output.outputs[idx].samples[0].output_token == self.eosid or self.running_reqs[req_id] - curtime > 3600:
                self.generated_tokens.pop(req_id)  # finish, clear cache
                self.running_reqs.pop(req_id)
                logger.info(f'delete request {req_id}')

        return output

    def check_sample_id(self, sample_ids):
        vocab_size = self.config.vocab_size  # 或 self.codec_id_max
        if sample_ids.numel() > 0:
            smin = int(sample_ids.min().item())
            smax = int(sample_ids.max().item())
            if smin < 0 or smax >= vocab_size:
                logger.error("Invalid sample_ids range smin=%d smax=%d vocab=%d", smin, smax, vocab_size)

    def ras_sample(self, logits, request_info, retry_num=0):
        # first sample
        logits = logits.to(torch.float)
        logits_temp = (logits / (self.temperature + 0.05 * retry_num)).contiguous()
        probs_temp = torch.softmax(logits_temp, dim=-1)
        sample_ids = top_k_top_p_sampling_from_probs(probs_temp, self.topk, self.topp, check_nan=True)
        # self.check_sample_id(sample_ids)
        if retry_num > 0:
            return sample_ids

        # resample
        logits_temp1 = (logits / 1.1).contiguous()
        probs_temp1 = torch.softmax(logits_temp1, dim=-1)
        resample_ids = top_k_top_p_sampling_from_probs(probs_temp1, self.topk+20, self.topp+0.15, check_nan=True)
        # compute repeat
        for idx, req_id in enumerate(request_info):
            sp = sample_ids[idx]
            rep_nums = (self.generated_tokens[req_id][-self.window:] == sp).sum(0)
            if rep_nums >= 1:
                sample_ids[idx] = resample_ids[idx]
        # self.check_sample_id(sample_ids)
        return sample_ids

    def sample_vllm_ras(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
               request_info: Dict) -> SamplerOutput:
        """
        使用vLLM的Sampler实现ras_sample功能，包含重采样机制
        目前还容易出现连续采样出0的问题
        Args:
            logits: 模型输出的logits张量
            sampling_metadata: vLLM的采样元数据
            request_info: 请求信息字典

        Returns:
            SamplerOutput: 包含采样结果的vLLM输出对象
        """
        request_ids = list(request_info.keys())
        curtime = time.time()

        # 初始化请求跟踪数据结构
        for req_id in request_ids:
            if req_id not in self.running_reqs:
                self.running_reqs[req_id] = curtime
            if req_id not in self.generated_tokens:
                self.generated_tokens[req_id] = torch.tensor(
                    [], dtype=torch.int32, device=logits.device)

        # 第一次采样（使用原始参数）
        output = self.sampler(logits, sampling_metadata)
        sample_ids = torch.tensor([
            output.outputs[idx].samples[0].output_token
            for idx in range(len(request_ids))
        ], device=logits.device)

        # 修改拷贝的 seq_groups 中的采样参数
        copied_seq_groups = []
        for seq_group in sampling_metadata.seq_groups:
            copied_seq_group = SequenceGroupToSample(
                seq_ids=seq_group.seq_ids,
                sampling_params=seq_group.sampling_params.clone(),  # 此参数需要拷贝出来
                seq_data=seq_group.seq_data,
                seq_len=seq_group.seq_len,
                query_len=seq_group.query_len,
                generator=seq_group.generator,
                is_prompt=seq_group.is_prompt,
                prompt_logprob_indices=seq_group.prompt_logprob_indices,
                sample_indices=seq_group.sample_indices
            )

            copied_seq_group.sampling_params.temperature = 1.1
            if copied_seq_group.sampling_params.top_k is not None:
                copied_seq_group.sampling_params.top_k = seq_group.sampling_params.top_k + 15
            if copied_seq_group.sampling_params.top_p is not None:
                copied_seq_group.sampling_params.top_p = min(seq_group.sampling_params.top_p + 0.15, 1.0)

            copied_seq_groups.append(copied_seq_group)

        resampling_metadata = SamplingMetadata(
            seq_groups=copied_seq_groups,
            selected_token_indices=sampling_metadata.selected_token_indices,
            categorized_sample_indices=sampling_metadata.categorized_sample_indices,
            num_prompts=sampling_metadata.num_prompts,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output,
            reuse_sampling_tensors=sampling_metadata.reuse_sampling_tensors,
        )
        # 执行重采样
        resample_output = self.sampler(logits, resampling_metadata)
        resample_ids = torch.tensor([
            resample_output.outputs[idx].samples[0].output_token
            for idx in range(len(request_ids))
        ], device=logits.device)

        # 检查重复并决定是否使用重采样结果
        for idx, req_id in enumerate(request_ids):
            sp = sample_ids[idx]
            # 检查最近10个token是否有重复
            if len(self.generated_tokens[req_id]) > 0:
                rep_nums = (self.generated_tokens[req_id][-self.window:] == sp).sum()
                if rep_nums >= 1:
                    sample_ids[idx] = resample_ids[idx]

        # 更新生成的token历史
        for idx, (sample_id, req_id) in enumerate(zip(sample_ids, request_ids)):
            self.generated_tokens[req_id] = torch.cat([
                self.generated_tokens[req_id],
                torch.tensor([sample_id], dtype=torch.int32,
                             device=sample_id.device)
            ], dim=0)
            # 保持历史记录不超过10个token
            if len(self.generated_tokens[req_id]) > self.window:
                self.generated_tokens[req_id] = self.generated_tokens[req_id][-self.window:]

            # 更新输出对象
            output.outputs[idx].samples[0].output_token = sample_id.item()
            # 这里需要正确设置logprobs，对logits处理后，有时只剩1best, 这里可设置为inf
            output.outputs[idx].samples[0].logprobs = {sample_id.item(): Logprob(float('inf'))}

        output.sampled_token_ids = sample_ids.unsqueeze(1)  # B,1  这个是vllm计算embedding需要的id
        # 清理完成的请求
        for idx, req_id in enumerate(request_ids):
            if (output.outputs[idx].samples[0].output_token == self.eosid or
                    curtime - self.running_reqs[req_id] > 3600):
                if req_id in self.generated_tokens:
                    del self.generated_tokens[req_id]
                if req_id in self.running_reqs:
                    del self.running_reqs[req_id]
                logger.info(f'Finished and cleared request {req_id}')

        return output