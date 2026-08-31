#!/usr/bin/env python3
"""GPU equivalence probe for the GLM5Next compact KDA runtime path."""

import torch

from vllm.model_executor.layers.mamba.ops.compact_kda_replay import (
    compact_kda_replay,
)
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda


def main() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_sequences = 2
    tokens_per_sequence = 8
    num_tokens = num_sequences * tokens_per_sequence
    num_heads = 4
    head_dim = 128

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.2

    q = rand(1, num_tokens, num_heads, head_dim)
    k = rand(1, num_tokens, num_heads, head_dim)
    v = rand(1, num_tokens, num_heads, head_dim)
    raw_g = rand(1, num_tokens, num_heads, head_dim)
    raw_beta = rand(1, num_tokens, num_heads)
    a_log = (
        torch.randn(
            1, 1, num_heads, 1, device=device, dtype=torch.float32
        )
        * 0.1
    )
    dt_bias = (
        torch.randn(
            num_heads * head_dim, device=device, dtype=torch.float32
        )
        * 0.1
    )
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
    prior_accepted = torch.ones(
        num_sequences, device=device, dtype=torch.int32
    )
    initial = (
        torch.randn(
            17,
            num_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        * 0.1
    )
    # Each sequence's column-zero slot is its committed initial state.
    initial[9].copy_(initial[1])

    full_state = initial.clone()
    compact_state = initial.clone()
    compact_before = compact_state.clone()
    common = {
        "q": q,
        "k": k,
        "v": v,
        "g": raw_g,
        "beta": raw_beta,
        "a_log": a_log,
        "g_bias": dt_bias,
        "compute_gate": True,
        "sigmoid_beta": True,
        "lower_bound": -5.0,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": state_indices,
        "num_accepted_tokens": prior_accepted,
    }
    full_out, _ = fused_recurrent_kda(
        initial_state=full_state,
        compact_spec_replay=False,
        **common,
    )
    compact_out, _ = fused_recurrent_kda(
        initial_state=compact_state,
        compact_spec_replay=True,
        **{
            **common,
            "ssm_state_indices": state_indices[:, :1].contiguous(),
        },
    )
    torch.cuda.synchronize()

    output_max_abs = (
        full_out.float() - compact_out.float()
    ).abs().max().item()
    verification_state_max_abs = (
        compact_state.float() - compact_before.float()
    ).abs().max().item()
    assert output_max_abs == 0.0, output_max_abs
    assert verification_state_max_abs == 0.0, verification_state_max_abs

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
    replay_state_max_abs = (
        expected.float() - actual.float()
    ).abs().max().item()
    assert replay_state_max_abs == 0.0, replay_state_max_abs

    # A one-step comparison can hide a bad committed-state convention. Verify
    # that a second speculative pass, initialized by a nontrivial prior
    # acceptance length, remains exactly equivalent too.
    q2 = rand(1, num_tokens, num_heads, head_dim)
    k2 = rand(1, num_tokens, num_heads, head_dim)
    v2 = rand(1, num_tokens, num_heads, head_dim)
    raw_g2 = rand(1, num_tokens, num_heads, head_dim)
    raw_beta2 = rand(1, num_tokens, num_heads)
    common2 = {
        **common,
        "q": q2,
        "k": k2,
        "v": v2,
        "g": raw_g2,
        "beta": raw_beta2,
        "num_accepted_tokens": accepted,
    }
    full_out2, _ = fused_recurrent_kda(
        initial_state=full_state,
        compact_spec_replay=False,
        **common2,
    )
    compact_before2 = compact_state.clone()
    compact_out2, _ = fused_recurrent_kda(
        initial_state=compact_state,
        compact_spec_replay=True,
        **{
            **common2,
            "ssm_state_indices": state_indices[:, :1].contiguous(),
        },
    )
    torch.cuda.synchronize()
    second_output_max_abs = (
        full_out2.float() - compact_out2.float()
    ).abs().max().item()
    second_verification_state_max_abs = (
        compact_state.float() - compact_before2.float()
    ).abs().max().item()
    assert second_output_max_abs == 0.0, second_output_max_abs
    assert second_verification_state_max_abs == 0.0, (
        second_verification_state_max_abs
    )

    accepted2 = torch.tensor([8, 3], device=device, dtype=torch.int32)
    compact_kda_replay(
        k=k2,
        v=v2,
        raw_g=raw_g2,
        raw_beta=raw_beta2,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        state=compact_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices[:, 0].contiguous(),
        num_accepted_tokens=accepted2,
    )
    torch.cuda.synchronize()
    expected2 = torch.stack(
        [
            full_state[state_indices[0, accepted2[0] - 1]],
            full_state[state_indices[1, accepted2[1] - 1]],
        ]
    )
    actual2 = torch.stack([compact_state[1], compact_state[9]])
    second_replay_state_max_abs = (
        expected2.float() - actual2.float()
    ).abs().max().item()
    assert second_replay_state_max_abs == 0.0, second_replay_state_max_abs

    print(
        {
            "output_max_abs": output_max_abs,
            "verification_state_max_abs": verification_state_max_abs,
            "replay_state_max_abs": replay_state_max_abs,
            "accepted": accepted.tolist(),
            "second_output_max_abs": second_output_max_abs,
            "second_verification_state_max_abs": (
                second_verification_state_max_abs
            ),
            "second_replay_state_max_abs": second_replay_state_max_abs,
            "accepted2": accepted2.tolist(),
        }
    )


if __name__ == "__main__":
    main()
