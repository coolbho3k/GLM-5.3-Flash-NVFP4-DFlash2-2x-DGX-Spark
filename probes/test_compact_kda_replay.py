#!/usr/bin/env python3
"""GPU equivalence probe for compact speculative KDA replay."""

import torch

from vllm.model_executor.layers.mamba.ops.compact_kda_replay import (
    compact_kda_replay,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
    fused_recurrent_kda,
)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_sequences = 2
    tokens_per_sequence = 8
    num_tokens = num_sequences * tokens_per_sequence
    num_heads = 2
    head_dim = 16

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.2

    q = rand(1, num_tokens, num_heads, head_dim)
    k = rand(1, num_tokens, num_heads, head_dim)
    v = rand(1, num_tokens, num_heads, head_dim)
    raw_g = rand(1, num_tokens, num_heads, head_dim)
    raw_beta = rand(1, num_tokens, num_heads)
    a_log = torch.randn(num_heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(
        num_heads * head_dim, device=device, dtype=torch.float32
    ) * 0.1
    cu_seqlens = torch.arange(
        0,
        num_tokens + 1,
        tokens_per_sequence,
        device=device,
        dtype=torch.int32,
    )

    state_indices = torch.tensor(
        [list(range(1, 9)), list(range(9, 17))],
        device=device,
        dtype=torch.int32,
    )
    prior_accepted = torch.ones(num_sequences, device=device, dtype=torch.int32)
    initial = torch.randn(
        17,
        num_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float16,
    ) * 0.1
    # Each sequence's column-zero slot is its committed initial state.
    initial[9].copy_(initial[1])

    full_state = initial.clone()
    compact_state = initial.clone()
    compact_before = compact_state.clone()

    full_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=full_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=prior_accepted,
        compact_spec_replay=False,
    )
    compact_out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=compact_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=prior_accepted,
        compact_spec_replay=True,
    )
    torch.cuda.synchronize()

    output_max_abs = (full_out.float() - compact_out.float()).abs().max().item()
    verify_state_max_abs = (
        compact_state.float() - compact_before.float()
    ).abs().max().item()
    assert output_max_abs == 0.0, output_max_abs
    assert verify_state_max_abs == 0.0, verify_state_max_abs

    accepted = torch.tensor([1, 5], device=device, dtype=torch.int32)
    compact_kda_replay(
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        state=compact_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices[:, 0].contiguous(),
        num_accepted_tokens=accepted,
    )
    torch.cuda.synchronize()

    expected = torch.stack(
        [
            full_state[state_indices[0, accepted[0] - 1]],
            full_state[state_indices[1, accepted[1] - 1]],
        ]
    )
    actual = torch.stack([compact_state[1], compact_state[9]])
    replay_max_abs = (expected.float() - actual.float()).abs().max().item()
    assert replay_max_abs == 0.0, replay_max_abs

    print(
        {
            "output_max_abs": output_max_abs,
            "verification_state_max_abs": verify_state_max_abs,
            "replay_state_max_abs": replay_max_abs,
            "accepted": accepted.tolist(),
        }
    )


if __name__ == "__main__":
    main()
