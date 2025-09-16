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
from vllm.model_executor.models.qwen2 import *
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.sequence import Logprob
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

        # self.sampler = Sampler()   # 注释这行，则会使用vllm原始采样
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

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata, self.lm_head.bias)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def sample(self, logits, sampling_metadata, request_info):
        request_info = list(request_info.keys())
        # clean regularly
        curtime = time.time()
        for req_id in request_info:
            if req_id not in self.running_reqs:
                self.running_reqs[req_id] = curtime
            if req_id not in self.generated_tokens:
                self.generated_tokens[req_id] = torch.tensor([], dtype=torch.int32, device=logits.device)   # add null token sequence

        output = self.sampler(logits, sampling_metadata)   # vllm sample
        vllm_sample_tokens = [output.outputs[idx].samples[0].output_token for idx in range(len(request_info))]

        # sampling
        sample_ids = self.ras_sample(logits, request_info)
        # check early stop
        for batch_idx in range(sample_ids.shape[0]):
            for retry in range(20):
                end = (sample_ids[batch_idx] == self.eosid).all()
                if end and self.generated_tokens[
                    request_info[batch_idx]].shape[0] < 15:
                    resample = self.ras_sample(
                        logits[batch_idx:batch_idx + 1],
                        [request_info[batch_idx]], retry)
                    sample_ids[batch_idx:batch_idx + 1] = resample
                else:
                    break

        # add new ids
        for idx, (sample_id, req_id) in enumerate(
                zip(sample_ids, request_info)):
            self.generated_tokens[req_id] = torch.cat(
                [self.generated_tokens[req_id],
                 torch.tensor([sample_id], dtype=torch.int32, device=sample_id.device)], dim=0)
            self.generated_tokens[req_id] = self.generated_tokens[req_id][-20:]
            output.outputs[idx].samples[0].output_token = sample_id
            output.outputs[idx].samples[0].logprobs = {sample_id: Logprob(0.0)}   # logprob貌似没有用，但是需要，设为0

        output.sampled_token_ids = sample_ids.unsqueeze(1)  # B,1

        for idx, req_id in enumerate(request_info):
            if output.outputs[idx].samples[0].output_token == self.eosid or self.running_reqs[req_id] - curtime > 3600:
                self.generated_tokens.pop(req_id)  # finish, clear cache
                self.running_reqs.pop(req_id)
                logger.info(f'delete request {req_id}')

        return output

    def ras_sample(self, logits, request_info, retry_num=0):
        # first sample
        logits_temp = (logits / (self.temperature + 0.5 * retry_num)).contiguous()
        probs_temp = torch.softmax(logits_temp, dim=-1)
        # sample_ids = top_k_top_p_min_p_sampling_from_probs_torch(probs_temp,self.topk+50*retry_num, self.topp+0.5*retry_num) # [B]
        sample_ids = top_k_top_p_sampling_from_probs(probs_temp, self.topk, self.topp, filter_apply_order='joint')

        if retry_num > 0:
            return sample_ids

        # resample
        logits_temp = (logits / 1.1).contiguous()
        probs_temp = torch.softmax(logits_temp, dim=-1)
        # resample_ids = top_k_top_p_min_p_sampling_from_probs_torch(probs_temp,self.topk+40, self.topp+0.15) # [B]
        resample_ids = top_k_top_p_sampling_from_probs(probs_temp, self.topk+15, self.topp+0.15, filter_apply_order='joint')
        # compute repeat
        for idx, req_id in enumerate(request_info):
            sp = sample_ids[idx]
            rep_nums = (self.generated_tokens[req_id][-10:] == sp).sum(0)
            if rep_nums >= 1:
                sample_ids[idx] = resample_ids[idx]
        return sample_ids