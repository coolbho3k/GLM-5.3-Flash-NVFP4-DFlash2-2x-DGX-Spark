#!/usr/bin/env python3
"""Numerical and safety gates for the SM121 W4A16 expert-map backport."""

import torch

from flashinfer import B12xMoEWrapper
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe.cute_dsl.b12x_moe import _prepare_ep_expert_map


DEVICE = torch.device("cuda")
E, TOPK, H, N, TOKENS = 6, 2, 256, 512, 24


def quantize(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, cols = weights.shape
    global_scale = torch.tensor([1.0], device=DEVICE, dtype=torch.float32)
    packed, scales = fp4_quantize(
        weights.reshape(experts * rows, cols),
        global_scale=global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    packed = packed.view(experts, rows, cols // 2)
    scales = convert_sf_to_mma_layout(
        scales, m=rows, k=cols, num_groups=experts, sf_vec_size=16
    )
    return packed, scales


def make_weights(experts: int) -> dict[str, torch.Tensor]:
    w1_bf16 = torch.randn(experts, 2 * N, H, device=DEVICE, dtype=torch.bfloat16)
    w2_bf16 = torch.randn(experts, H, N, device=DEVICE, dtype=torch.bfloat16)
    w1, s1 = quantize(w1_bf16)
    w2, s2 = quantize(w2_bf16)
    return {
        "w1_bf16": w1_bf16,
        "w2_bf16": w2_bf16,
        "w1": w1,
        "s1": s1,
        "w2": w2,
        "s2": s2,
        "a1": torch.ones(experts, device=DEVICE, dtype=torch.float32),
        "a2": torch.ones(experts, device=DEVICE, dtype=torch.float32),
    }


def run(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    ids: torch.Tensor,
    routing: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    local_e = weights["w1"].shape[0]
    wrapper = B12xMoEWrapper(
        num_experts=E,
        num_local_experts=local_e,
        expert_map=expert_map,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=N,
        quant_mode="w4a16",
        use_cuda_graph=False,
    )
    return wrapper.run(
        x=x,
        w1_weight=weights["w1"],
        w1_weight_sf=weights["s1"],
        w1_alpha=weights["a1"],
        w2_weight=weights["w2"],
        w2_weight_sf=weights["s2"],
        w2_alpha=weights["a2"],
        token_selected_experts=ids,
        token_final_scales=routing,
    ).clone()


def shard(full: dict[str, torch.Tensor], global_ids: torch.Tensor):
    global_ids = global_ids.to(device=DEVICE, dtype=torch.long)
    local = {
        "w1_bf16": full["w1_bf16"].index_select(0, global_ids).contiguous(),
        "w2_bf16": full["w2_bf16"].index_select(0, global_ids).contiguous(),
    }
    local["w1"], local["s1"] = quantize(local["w1_bf16"])
    local["w2"], local["s2"] = quantize(local["w2_bf16"])
    local_e = global_ids.numel()
    local["a1"] = torch.ones(local_e, device=DEVICE, dtype=torch.float32)
    local["a2"] = torch.ones(local_e, device=DEVICE, dtype=torch.float32)
    expert_map = torch.full((E,), -1, device=DEVICE, dtype=torch.int32)
    expert_map[global_ids] = torch.arange(local_e, device=DEVICE, dtype=torch.int32)
    return local, expert_map


def main() -> None:
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(4302)

    valid = torch.tensor([0, -1, 1], dtype=torch.int32)
    assert _prepare_ep_expert_map(
        valid, num_local_experts=2, num_experts=3
    ) is valid
    try:
        _prepare_ep_expert_map(
            torch.tensor([0, 0], dtype=torch.int32),
            num_local_experts=2,
            num_experts=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe duplicate expert map was accepted")

    full = make_weights(E)
    x = (torch.randn(TOKENS, H, device=DEVICE) * 0.1).to(torch.bfloat16)
    ids = torch.randint(E, (TOKENS, TOPK), device=DEVICE, dtype=torch.int32)
    routing = torch.softmax(
        torch.randn(TOKENS, TOPK, device=DEVICE, dtype=torch.float32), dim=-1
    )

    expected = run(full, x, ids, routing)
    rank0, map0 = shard(full, torch.tensor([0, 1, 2]))
    rank1, map1 = shard(full, torch.tensor([3, 4, 5]))
    actual = run(rank0, x, ids, routing, map0) + run(rank1, x, ids, routing, map1)
    torch.cuda.synchronize()

    assert torch.count_nonzero(expected).item() > 0
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    assert cosine > 0.999
    print(f"B12X W4A16 EP partial-sum test passed (cosine={cosine:.8f})")


if __name__ == "__main__":
    main()
