import torch
import torch.nn as nn
from typing import Optional
from flashinfer.sampling import top_k_top_p_sampling_from_probs

def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor] = None,
    need_min_p_sampling: bool = False,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits, #[BS,NUM_CALSSES]
        temperature,
        top_k,
        top_p,
    ):
        # Post process logits
        logits = logits.div(torch.FloatTensor([[temperature]]).to(logits.device)) #[1,1]
        probs = torch.softmax(logits, dim=-1)

        max_top_k_round, batch_size = 32, probs.shape[0]
        uniform_samples = torch.rand(
            (max_top_k_round, batch_size), device=probs.device
        )
        batch_next_token_ids = top_k_top_p_sampling_from_probs(probs, top_k, top_p, filter_apply_order="joint")
        # batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
        #     probs,
        #     torch.LongTensor([top_k]).to(logits.device).repeat(batch_size), #[1]
        #     torch.FloatTensor([top_p]).to(logits.device).repeat(batch_size), #[1]
        #     need_min_p_sampling=False
        # )
        return batch_next_token_ids.to(torch.int32)

sampler = Sampler()

def ras_sampling2(weighted_scores_in, decoded_tokens_in, top_p=0.8, top_k=5,
                  temperature=1.0, win_size=10, tau_r=0.1,
                  resample={'topp': 0.9, 'topk': 10,'temperature':1.0}):
    # weigted_scores_in: [B, Num_classes]
    # decoded_tokens = decoded_tokens_in[0, ...]
    decoded_tokens = torch.tensor(decoded_tokens_in).to(weighted_scores_in.device)

    # st = time.perf_counter()
    top_ids = sampler(weighted_scores_in, temperature, top_p=top_p, top_k=top_k)
    # et = time.perf_counter()
    # print("sampling time:", et-st)
    rep_nums = (decoded_tokens[-win_size:] == top_ids).sum(0)

    # st = time.perf_counter()
    rep_top_ids = sampler(weighted_scores_in,
                          resample['temperature'],
                          top_p=resample['topp'],
                          top_k=resample['topk'])
    top_ids = torch.where(rep_nums >= win_size * tau_r, rep_top_ids, top_ids)
    # et = time.perf_counter()
    # print("resample time:", et-st)
    return top_ids

if __name__ == "__main__":
    logits = torch.randn(1, 1000).cuda()
    sampler = Sampler()
    result = sampler.forward(logits, 1.0, 0, 0.9)
    pass