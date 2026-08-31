# SPDX-License-Identifier: Apache-2.0
"""Accepted-prefix replay for compact speculative KDA state.

The target verification kernel can keep its evolving recurrent state in
registers while producing all candidate outputs.  After rejection sampling,
this kernel reapplies only the accepted K/V/gate/beta transitions to the one
committed recurrent-state slot.  It uses the same FP32 accumulation and gate
math as the NVIDIA fused recurrent KDA kernel.
"""

import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.op import exp, log
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2


# The multimodal GLM wrapper constructs its text model through vLLM's
# registered-model indirection. Those inner modules are not visible from
# ``GPUModelRunner.model.modules()`` on this build, so the live KDA layers
# register themselves here for the post-sampling commit hook. This registry is
# process-local and resets naturally whenever the worker process restarts.
_REGISTERED_COMPACT_KDA_LAYERS: list[object] = []
_REGISTERED_COMPACT_KDA_LAYER_IDS: set[int] = set()


def register_compact_kda_layer(layer: object) -> None:
    layer_id = id(layer)
    if layer_id not in _REGISTERED_COMPACT_KDA_LAYER_IDS:
        _REGISTERED_COMPACT_KDA_LAYER_IDS.add(layer_id)
        _REGISTERED_COMPACT_KDA_LAYERS.append(layer)


def get_registered_compact_kda_layers() -> tuple[object, ...]:
    return tuple(_REGISTERED_COMPACT_KDA_LAYERS)


@triton.heuristics(
    {
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "stride_beta_token"])
def _compact_kda_replay_kernel(
    k,
    v,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    state,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    lower_bound,
    N: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_kv_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token,
    stride_state_token: tl.constexpr,
    num_stages: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    i_v = pid % tl.cdiv(V, BV)
    i_nh = pid // tl.cdiv(V, BV)
    i_n, i_h = i_nh // H, i_nh % H

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    accepted = tl.load(num_accepted_tokens + i_n).to(tl.int64)
    accepted = tl.minimum(accepted, eos - bos)
    if accepted <= 0:
        return

    state_index = tl.load(state_indices + i_n).to(tl.int64)
    if state_index <= 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_state = m_v[:, None] & m_k[None, :]

    p_state = (
        state
        + state_index * stride_state_token
        + i_h * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_state = tl.load(p_state, mask=m_state, other=0.0).to(tl.float32)

    p_k = k + bos * stride_kv_token + i_h * K + o_k
    p_v = v + bos * stride_kv_token + i_h * V + o_v
    p_g = raw_g + bos * stride_g_token + i_h * K + o_k
    p_beta = raw_beta + bos * stride_beta_token + i_h

    for _ in tl.range(0, accepted, num_stages=num_stages):
        b_k = tl.load(p_k, mask=m_k, other=0.0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=m_v, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        b_gate = tl.load(
            p_g, mask=m_k, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        if HAS_DT_BIAS:
            b_gate += tl.load(
                dt_bias + i_h * K + o_k, mask=m_k, other=0.0
            ).to(tl.float32)
        b_a = exp(tl.load(A_log + i_h).to(tl.float32))
        if USE_LOWER_BOUND:
            # Match the GLM verification kernel's bounded safe-gate expression
            # exactly, including operation ordering.
            b_gate = lower_bound / (1.0 + tl.exp(-(b_a * b_gate)))
        else:
            b_softplus = tl.where(
                b_gate > 20.0,
                b_gate,
                log(1.0 + tl.exp(b_gate)),
            )
            b_gate = -b_a * b_softplus

        b_state *= exp(b_gate[None, :])
        b_v -= tl.sum(b_state * b_k[None, :], axis=1)
        b_beta = tl.sigmoid(
            tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        )
        b_v *= b_beta
        b_state += b_v[:, None] * b_k[None, :]

        p_k += stride_kv_token
        p_v += stride_kv_token
        p_g += stride_g_token
        p_beta += stride_beta_token

    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=m_state)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


def compact_kda_replay(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Replay each sequence's accepted KDA transitions into ``state``."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    if B != 1 or v.shape != (B, T, H, V):
        raise ValueError("compact KDA replay expects packed [1,T,H,D] K/V")
    if raw_g.shape != k.shape or raw_beta.shape != (B, T, H):
        raise ValueError("compact KDA replay gate/beta shapes do not match K/V")
    if state.shape[1:] != (H, V, K):
        raise ValueError("compact KDA replay state shape does not match K/V")
    N = cu_seqlens.numel() - 1
    if state_indices.shape != (N,) or num_accepted_tokens.shape != (N,):
        raise ValueError("compact KDA replay sequence metadata shape mismatch")
    if not (
        cu_seqlens.is_contiguous()
        and state_indices.is_contiguous()
        and num_accepted_tokens.is_contiguous()
    ):
        raise ValueError("compact KDA replay metadata must be contiguous")

    head_sequences = H * N
    if head_sequences <= 48:
        BV, num_stages = 4, 4
    elif head_sequences <= 96:
        BV, num_stages = 8, 3
    elif head_sequences <= 192:
        BV, num_stages = 16, 3
    else:
        BV, num_stages = 8, 3

    grid = (cdiv(V, BV) * N * H,)
    _compact_kda_replay_kernel[grid](
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        N=N,
        H=H,
        K=K,
        V=V,
        BK=next_power_of_2(K),
        BV=BV,
        stride_kv_token=k.stride(1),
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        stride_state_token=state.stride(0),
        num_stages=num_stages,
        num_warps=1,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
