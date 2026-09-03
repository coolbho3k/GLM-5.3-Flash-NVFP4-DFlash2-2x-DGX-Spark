from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match in {path}, found {count}")
    path.write_text(source.replace(old, new))


# 1. Let the verification recurrence keep candidate states in registers rather
# than writing seven full rollback snapshots.
path = ROOT / "models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py"
replace_once(
    path,
    '''    IS_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
''',
    '''    IS_SPEC_DECODING: tl.constexpr,
    COMPACT_SPEC_REPLAY: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
''',
    "KDA kernel compact flag",
)
replace_once(
    path,
    '''    if IS_SPEC_DECODING:
        initial_token = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    else:
        initial_token = 0
''',
    '''    if IS_SPEC_DECODING and not COMPACT_SPEC_REPLAY:
        initial_token = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    else:
        # Compact replay commits the previously accepted state in-place, so
        # verification always starts from column zero's running state.
        initial_token = 0
''',
    "KDA compact initial state",
)
replace_once(
    path,
    '''        final_state_index = tl.load(state_indices + i_n * stride_indices_seq + i_t).to(
            tl.int64
        )
        if final_state_index > 0:
            p_final_state = (
                state
                + final_state_index * stride_state_token
                + i_h * V * K
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                p_final_state,
                b_state.to(p_final_state.dtype.element_ty),
                mask=m_state,
            )
''',
    '''        if not COMPACT_SPEC_REPLAY:
            final_state_index = tl.load(
                state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_index > 0:
                p_final_state = (
                    state
                    + final_state_index * stride_state_token
                    + i_h * V * K
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(
                    p_final_state,
                    b_state.to(p_final_state.dtype.element_ty),
                    mask=m_state,
                )
''',
    "KDA compact snapshot suppression",
)
replace_once(
    path,
    '''    use_beta_sigmoid_in_kernel: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    '''    use_beta_sigmoid_in_kernel: bool = False,
    out: torch.Tensor | None = None,
    compact_spec_replay: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    "KDA forward wrapper compact arg",
)
replace_once(
    path,
    '''        IS_SPEC_DECODING=num_accepted_tokens is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
''',
    '''        IS_SPEC_DECODING=num_accepted_tokens is not None,
        COMPACT_SPEC_REPLAY=compact_spec_replay,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
''',
    "KDA kernel compact launch",
)
replace_once(
    path,
    '''    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    '''    out: torch.Tensor | None = None,
    fuse_gate: bool | None = None,
    compact_spec_replay: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    "KDA public wrapper compact arg",
)
replace_once(
    path,
    '''        use_beta_sigmoid_in_kernel=fuse_gate,
        out=out,
    )
''',
    '''        use_beta_sigmoid_in_kernel=fuse_gate,
        out=out,
        compact_spec_replay=compact_spec_replay,
    )
''',
    "KDA public wrapper compact pass-through",
)


# 2. The GLM5Next layer calls the shared flash-linear-attention recurrence,
# not Kimi K3's vendored recurrence above.  Add the same verification contract
# to that actual runtime kernel and thread it through the public KDA wrapper.
path = ROOT / "third_party/flash_linear_attention/ops/fused_recurrent.py"
replace_once(
    path,
    '''    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
''',
    '''    IS_SPEC_DECODING: tl.constexpr,
    COMPACT_SPEC_REPLAY: tl.constexpr,
    IS_KDA: tl.constexpr,
''',
    "FLA recurrence compact flag",
)
replace_once(
    path,
    '''            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
''',
    '''            if IS_SPEC_DECODING and not COMPACT_SPEC_REPLAY:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                # Compact replay always keeps the committed state in column
                # zero; accepted candidate transitions are applied later.
                i_t = 0
''',
    "FLA recurrence compact initial state",
)
replace_once(
    path,
    '''        if INPLACE_FINAL_STATE:
            # Load state index and check for invalid entries
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            # Only store if state index is valid (not NULL_BLOCK_ID=0)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
''',
    '''        if INPLACE_FINAL_STATE:
            if not COMPACT_SPEC_REPLAY:
                # Load state index and check for invalid entries
                final_state_idx = tl.load(
                    ssm_state_indices + i_n * stride_indices_seq + i_t
                ).to(tl.int64)
                # Only store if state index is valid (not NULL_BLOCK_ID=0)
                if final_state_idx > 0:
                    p_ht = ht + final_state_idx * stride_final_state_token
                    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
''',
    "FLA recurrence compact snapshot suppression",
)
replace_once(
    path,
    '''        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
''',
    '''        INPLACE_FINAL_STATE=inplace_final_state,
        COMPACT_SPEC_REPLAY=False,
        IS_KDA=False,
''',
    "non-KDA recurrence compact default",
)

path = ROOT / "third_party/flash_linear_attention/ops/kda.py"
replace_once(
    path,
    '''    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    '''    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
    compact_spec_replay: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    "runtime KDA forward compact arg",
)
replace_once(
    path,
    '''        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=True,
''',
    '''        INPLACE_FINAL_STATE=inplace_final_state,
        COMPACT_SPEC_REPLAY=compact_spec_replay,
        IS_KDA=True,
''',
    "runtime KDA compact launch",
)
replace_once(
    path,
    '''    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    '''    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
    compact_spec_replay: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
    "runtime KDA public compact arg",
)
replace_once(
    path,
    '''        compute_gate=compute_gate,
        lower_bound=lower_bound,
    )
''',
    '''        compute_gate=compute_gate,
        lower_bound=lower_bound,
        compact_spec_replay=compact_spec_replay,
    )
''',
    "runtime KDA public compact pass-through",
)


# 3. Keep the generic Kimi path compatible with the compact replay hook.  This
# checkpoint uses the model-specific GLM layer patched below, but retaining the
# generic integration makes the experimental image internally consistent.
# Stage compact transition inputs, then replay the accepted prefix after
# sampling.
# Advertise zero speculative *state blocks* while the convolution cache
# retains its existing widened speculative window.
path = ROOT / "model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py"
replace_once(
    path,
    '''from collections.abc import Callable

import torch
''',
    '''from collections.abc import Callable
from dataclasses import replace
import os

import torch
''',
    "KDA dataclass replace import",
)
replace_once(
    path,
    '''from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..ops.gather_initial_states import gather_initial_states
''',
    '''from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..ops.compact_kda_replay import compact_kda_replay
from ..ops.gather_initial_states import gather_initial_states
''',
    "KDA replay import",
)
replace_once(
    path,
    '''        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
''',
    '''        super().__init__(config, vllm_config, prefix)
        self.compact_spec_replay = (
            self.num_spec > 0
            and os.environ.get("VLLM_COMPACT_SPEC_REPLAY", "1") == "1"
        )

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
''',
    "KDA compact replay gate",
)
replace_once(
    path,
    '''        self.local_num_heads = divide(self.num_heads, self.tp_size)

        self.projection_size = self.head_dim * self.num_heads
''',
    '''        self.local_num_heads = divide(self.num_heads, self.tp_size)

        if self.compact_spec_replay:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            max_replay_tokens = max_num_reqs * (self.num_spec + 1)
            replay_dtype = self.model_config.dtype
            token_shape = (
                1,
                max_replay_tokens,
                self.local_num_heads,
                self.head_dim,
            )
            self.register_buffer(
                "_compact_replay_k",
                torch.empty(token_shape, dtype=replay_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_v",
                torch.empty(token_shape, dtype=replay_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_g",
                torch.empty(token_shape, dtype=replay_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_beta",
                torch.empty(
                    (1, max_replay_tokens, self.local_num_heads),
                    dtype=replay_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_cu",
                torch.empty(max_num_reqs + 1, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_state_indices",
                torch.empty(max_num_reqs, dtype=torch.int32),
                persistent=False,
            )
            self._compact_replay_pending = False
            self._compact_replay_num_tokens = 0
            self._compact_replay_num_sequences = 0
            self._compact_replay_sequence_mask: torch.Tensor | None = None

        self.projection_size = self.head_dim * self.num_heads
''',
    "KDA compact replay buffers",
)
replace_once(
    path,
    '''    def rearrange_mixed_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
''',
    '''    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        spec = super().get_kv_cache_spec(vllm_config)
        if self.compact_spec_replay:
            # Candidate states are reconstructed from compact transition inputs;
            # only the running/committed align-mode blocks remain in the pool.
            spec = replace(spec, num_speculative_blocks=0)
        return spec

    def _stage_compact_spec_replay(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> None:
        num_tokens = k.shape[1]
        num_sequences = state_indices.shape[0]
        self._compact_replay_k[:, :num_tokens].copy_(k)
        self._compact_replay_v[:, :num_tokens].copy_(v)
        self._compact_replay_g[:, :num_tokens].copy_(raw_g)
        self._compact_replay_beta[:, :num_tokens].copy_(raw_beta)
        self._compact_replay_cu[: num_sequences + 1].copy_(
            cu_seqlens[: num_sequences + 1]
        )
        self._compact_replay_state_indices[:num_sequences].copy_(
            state_indices[:, 0]
        )
        self._compact_replay_num_tokens = num_tokens
        self._compact_replay_num_sequences = num_sequences
        self._compact_replay_sequence_mask = sequence_mask
        self._compact_replay_pending = True

    def compact_replay_sequence_mask(self) -> torch.Tensor:
        assert self._compact_replay_pending
        sequence_mask = self._compact_replay_sequence_mask
        assert sequence_mask is not None
        return sequence_mask

    def replay_compact_spec_state(
        self,
        accepted: torch.Tensor,
    ) -> None:
        if not self.compact_spec_replay or not self._compact_replay_pending:
            return
        num_sequences = self._compact_replay_num_sequences
        num_tokens = self._compact_replay_num_tokens
        assert accepted.numel() == num_sequences
        compact_kda_replay(
            k=self._compact_replay_k[:, :num_tokens],
            v=self._compact_replay_v[:, :num_tokens],
            raw_g=self._compact_replay_g[:, :num_tokens],
            raw_beta=self._compact_replay_beta[:, :num_tokens],
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            state=self.kv_cache[1],
            cu_seqlens=self._compact_replay_cu[: num_sequences + 1],
            state_indices=self._compact_replay_state_indices[:num_sequences],
            num_accepted_tokens=accepted,
        )
        self._compact_replay_pending = False

    def rearrange_mixed_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
''',
    "KDA compact replay methods",
)
replace_once(
    path,
    '''            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
''',
    '''            if self.compact_spec_replay:
                self._stage_compact_spec_replay(
                    k=k_spec,
                    v=v_spec,
                    raw_g=g1_spec,
                    raw_beta=beta_spec,
                    cu_seqlens=spec_cu_seqlens,
                    state_indices=spec_state_indices_tensor,
                    sequence_mask=spec_sequence_masks,
                )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
''',
    "KDA compact replay staging",
)
replace_once(
    path,
    '''                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
            )
''',
    '''                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
                compact_spec_replay=self.compact_spec_replay,
            )
''',
    "KDA compact verification",
)


# 4. GLM5Next owns a model-specific KDA layer and does not instantiate the
# generic Kimi class above.  Stage its post-convolution transition inputs,
# suppress speculative snapshots in the shared recurrence, and replay only the
# accepted prefix after sampling.
path = ROOT / "models/glm5next/nvidia/kda.py"
replace_once(
    path,
    '''import torch
from torch import nn
''',
    '''import os

import torch
from torch import nn
''',
    "GLM KDA compact debug import",
)
replace_once(
    path,
    '''from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
''',
    '''from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.compact_kda_replay import (
    compact_kda_replay,
    register_compact_kda_layer,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
''',
    "GLM KDA compact replay import",
)
replace_once(
    path,
    '''        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config

        # Linear-attention head config: read the flattened top-level fields when
''',
    '''        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
        self.compact_spec_replay = (
            self.num_spec > 0
            and os.environ.get("VLLM_COMPACT_SPEC_REPLAY", "1") == "1"
        )

        # Linear-attention head config: read the flattened top-level fields when
''',
    "GLM KDA compact replay gate",
)
replace_once(
    path,
    '''        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
''',
    '''        self.local_num_heads = divide(self.num_heads, self.tp_size)

        if self.compact_spec_replay:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            max_replay_tokens = max_num_reqs * (self.num_spec + 1)
            replay_dtype = vllm_config.model_config.dtype
            token_shape = (
                1,
                max_replay_tokens,
                self.local_num_heads,
                self.head_dim,
            )
            self.register_buffer(
                "_compact_replay_k",
                torch.empty(token_shape, dtype=replay_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_v",
                torch.empty(token_shape, dtype=replay_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_g",
                torch.empty(token_shape, dtype=replay_dtype),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_beta",
                torch.empty(
                    (1, max_replay_tokens, self.local_num_heads),
                    dtype=replay_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_cu",
                torch.empty(max_num_reqs + 1, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_state_indices",
                torch.empty(max_num_reqs, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_sequence_mask",
                torch.empty(max_num_reqs, dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "_compact_replay_accepted",
                torch.empty(max_num_reqs, dtype=torch.int32),
                persistent=False,
            )
            self._compact_replay_pending = False
            # CUDA graphs replay captured tensor operations, but not this
            # Python method or its scalar assignments. Keep replay metadata at
            # fixed maximum shapes so every graph updates the same buffers.
            self._compact_replay_num_tokens = max_replay_tokens
            self._compact_replay_num_sequences = max_num_reqs
            self._compact_replay_debug = (
                os.environ.get("VLLM_COMPACT_REPLAY_DEBUG", "0") == "1"
            )
            self._compact_replay_debug_steps = 0
            register_compact_kda_layer(self)

        projection_size = self.head_dim * self.num_heads
''',
    "GLM KDA compact replay buffers",
)
replace_once(
    path,
    '''    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
''',
    '''    def _stage_compact_spec_replay(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        sequence_mask: torch.Tensor,
        previous_accepted: torch.Tensor,
        num_sequences: int,
    ) -> None:
        # vLLM can rebound model modules after warmup; register the exact layer
        # instance that produced these transition tensors.
        register_compact_kda_layer(self)
        num_tokens = k.shape[1]
        self._compact_replay_k[:, :num_tokens].copy_(k)
        self._compact_replay_v[:, :num_tokens].copy_(v)
        self._compact_replay_g[:, :num_tokens].copy_(raw_g)
        self._compact_replay_beta[:, :num_tokens].copy_(raw_beta)
        self._compact_replay_cu[: num_sequences + 1].copy_(
            cu_seqlens[: num_sequences + 1]
        )
        self._compact_replay_cu[num_sequences + 1 :].fill_(num_tokens)
        self._compact_replay_state_indices.fill_(0)
        self._compact_replay_state_indices[:num_sequences].copy_(
            state_indices[:num_sequences, 0]
        )
        self._compact_replay_sequence_mask.fill_(False)
        self._compact_replay_sequence_mask[: sequence_mask.numel()].copy_(
            sequence_mask
        )
        self._compact_replay_pending = True
        if (
            self._compact_replay_debug
            and self.prefix.endswith("layers.0.self_attn")
            and self._compact_replay_debug_steps < 8
        ):
            torch.cuda.synchronize()
            print(
                "COMPACT_STAGE"
                f" rank={torch.distributed.get_rank()}"
                f" step={self._compact_replay_debug_steps}"
                f" previous_accepted={previous_accepted[:num_sequences].tolist()}"
                f" state_indices={state_indices[:num_sequences].tolist()}"
                f" cu={cu_seqlens[: num_sequences + 1].tolist()}"
                , flush=True
            )

    def compact_replay_sequence_mask(self) -> torch.Tensor:
        return self._compact_replay_sequence_mask

    def replay_compact_spec_state(
        self,
        accepted: torch.Tensor,
    ) -> None:
        if not self.compact_spec_replay:
            return
        num_sequences = self._compact_replay_num_sequences
        num_tokens = self._compact_replay_num_tokens
        if accepted.numel() > num_sequences:
            raise ValueError("too many speculative sequences for compact replay")
        self._compact_replay_accepted.zero_()
        self._compact_replay_accepted[: accepted.numel()].copy_(accepted)
        debug_this_step = (
            self._compact_replay_debug
            and self.prefix.endswith("layers.0.self_attn")
            and self._compact_replay_debug_steps < 8
        )
        if debug_this_step:
            torch.cuda.synchronize()
            state_index = int(self._compact_replay_state_indices[0].item())
            before_norm = float(self.kv_cache[1][state_index].float().norm().item())
        compact_kda_replay(
            k=self._compact_replay_k[:, :num_tokens],
            v=self._compact_replay_v[:, :num_tokens],
            raw_g=self._compact_replay_g[:, :num_tokens],
            raw_beta=self._compact_replay_beta[:, :num_tokens],
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.kda_lower_bound,
            state=self.kv_cache[1],
            cu_seqlens=self._compact_replay_cu[: num_sequences + 1],
            state_indices=self._compact_replay_state_indices[:num_sequences],
            num_accepted_tokens=self._compact_replay_accepted,
        )
        if debug_this_step:
            torch.cuda.synchronize()
            after_norm = float(self.kv_cache[1][state_index].float().norm().item())
            print(
                "COMPACT_COMMIT"
                f" rank={torch.distributed.get_rank()}"
                f" step={self._compact_replay_debug_steps}"
                f" accepted={accepted.tolist()}"
                f" state_index={state_index}"
                f" before_norm={before_norm:.8f}"
                f" after_norm={after_norm:.8f}"
                , flush=True
            )
            self._compact_replay_debug_steps += 1
        self._compact_replay_pending = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
''',
    "GLM KDA compact replay methods",
)
replace_once(
    path,
    '''            conv_idx = spec_state_indices_tensor[:, 0][:num_spec_decodes]
            conv_mql = spec_state_indices_tensor.size(-1)
            qkv_spec = causal_conv1d_update(
''',
    '''            conv_idx = spec_state_indices_tensor[:, 0][:num_spec_decodes]
            # Compact recurrent state has one table column, but convolution
            # rollback still spans the full draft-verify window.
            conv_mql = self.num_spec + 1
            qkv_spec = causal_conv1d_update(
''',
    "GLM compact convolution window",
)
replace_once(
    path,
    '''            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
''',
    '''            if self.compact_spec_replay:
                self._stage_compact_spec_replay(
                    k=_rearr(k_spec),
                    v=_rearr(v_spec),
                    raw_g=g1_spec,
                    raw_beta=beta_spec,
                    cu_seqlens=spec_query_start_loc,
                    state_indices=spec_state_indices_tensor,
                    sequence_mask=spec_sequence_masks,
                    previous_accepted=num_accepted_tokens,
                    num_sequences=num_spec_decodes,
                )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
''',
    "GLM compact replay staging",
)
replace_once(
    path,
    '''                compute_gate=True,
                lower_bound=lower_bound,
            )

        # --- core attention: non-spec path (prefill or plain decode) ---
''',
    '''                compute_gate=True,
                lower_bound=lower_bound,
                compact_spec_replay=self.compact_spec_replay,
            )

        # --- core attention: non-spec path (prefill or plain decode) ---
''',
    "GLM compact verification",
)


# 5. Cache discovery may operate through the shared Mamba interface rather
# than the concrete pluggable layer.  Make the authoritative MambaSpec carry
# the same compact-state contract for this exact GLM+DFlash configuration.
path = ROOT / "model_executor/layers/mamba/abstract.py"
replace_once(
    path,
    '''from math import prod

import torch
''',
    '''from math import prod
import os

import torch
''',
    "authoritative compact Mamba environment import",
)
replace_once(
    path,
    '''            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
''',
    '''            num_speculative_blocks=(
                0
                if (
                    vllm_config.speculative_config is not None
                    and vllm_config.model_config.architecture
                    == "Glm5NextForConditionalGeneration"
                    and os.environ.get("VLLM_COMPACT_SPEC_REPLAY", "1") == "1"
                )
                else (
                    vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config
                    else 0
                )
            ),
''',
    "authoritative compact Mamba cache spec",
)


# 6. Commit the accepted KDA prefix before the existing align-mode state-copy
# postprocess migrates a running block across a page boundary.
path = ROOT / "v1/worker/gpu_model_runner.py"
replace_once(
    path,
    '''        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        if self.cache_config.mamba_cache_mode == "align":
''',
    '''        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        if not hasattr(self, "_compact_replay_layers"):
            from vllm.model_executor.layers.mamba.ops.compact_kda_replay import (
                get_registered_compact_kda_layers,
            )

            self._compact_replay_layers = get_registered_compact_kda_layers()
        pending_replays = [
            (layer, layer.replay_compact_spec_state)
            for layer in self._compact_replay_layers
            if getattr(layer, "_compact_replay_pending", False)
        ]
        if pending_replays:
            sequence_mask = pending_replays[0][0].compact_replay_sequence_mask()
            accepted = self.num_accepted_tokens.gpu[:num_reqs][
                sequence_mask[:num_reqs]
            ].contiguous()
            for _, replay in pending_replays:
                replay(accepted)

        if self.cache_config.mamba_cache_mode == "align":
''',
    "runner compact replay post-sampling hook",
)


# 7. In compact mode the temporal state at src_col has already been replayed
# to the accepted boundary.  Conv state still uses token_bias exactly as before.
path = ROOT / "v1/worker/mamba_utils.py"
replace_once(
    path,
    '''    TEMPORAL_TILES: tl.constexpr,
):
''',
    '''    TEMPORAL_TILES: tl.constexpr,
    COMPACT_KDA_REPLAY: tl.constexpr,
):
''',
    "mamba copy compact flag",
)
replace_once(
    path,
    '''    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
''',
    '''    if COMPACT_KDA_REPLAY:
        temporal_src_col = src_col
    else:
        temporal_src_col = src_col + token_bias
    actual_src_block_id = tl.load(block_table_base + temporal_src_col).to(tl.int64)
''',
    "mamba compact temporal source",
)
source = path.read_text()
old = '''    TEMPORAL_TILES: tl.constexpr = 1,
):
'''
new = '''    TEMPORAL_TILES: tl.constexpr = 1,
    COMPACT_KDA_REPLAY: tl.constexpr = False,
):
'''
if source.count(old) != 2:
    raise RuntimeError(
        f"mamba fused kernel compact flags: expected two matches, found {source.count(old)}"
    )
path.write_text(source.replace(old, new))


source = path.read_text()
old = '''        TEMPORAL_TILES,
    )
'''
new = '''        TEMPORAL_TILES,
        COMPACT_KDA_REPLAY,
    )
'''
if source.count(old) != 2:
    raise RuntimeError(
        f"mamba copy compact pass-through: expected two matches, found {source.count(old)}"
    )
path.write_text(source.replace(old, new))


replace_once(
    path,
    '''    num_groups: int

    # Output buffer for num_accepted_tokens updates
''',
    '''    num_groups: int
    compact_kda_replay: bool

    # Output buffer for num_accepted_tokens updates
''',
    "mamba context compact field",
)
replace_once(
    path,
    '''            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
''',
    '''            num_groups=len(mamba_group_ids),
            compact_kda_replay=mamba_spec.num_speculative_blocks == 0,
            num_accepted_tokens_out=torch.zeros(
''',
    "mamba context compact initialization",
)
source = path.read_text()
old = '''            TEMPORAL_TILES=_TEMPORAL_TILES,
        )
'''
new = '''            TEMPORAL_TILES=_TEMPORAL_TILES,
            COMPACT_KDA_REPLAY=self.compact_kda_replay,
        )
'''
if source.count(old) != 3:
    raise RuntimeError(
        f"mamba compact launches: expected three matches, found {source.count(old)}"
    )
path.write_text(source.replace(old, new))


# 8. This deployment uses the V2 GPU model runner. Commit the accepted compact
# KDA transition prefix immediately after rejection sampling, before
# postprocess_sampled migrates align-mode Mamba state across block boundaries.
path = ROOT / "v1/worker/gpu/model_runner.py"
replace_once(
    path,
    '''        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
''',
    '''        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        from vllm.model_executor.layers.mamba.ops.compact_kda_replay import (
            get_registered_compact_kda_layers,
        )

        compact_replay_layers = get_registered_compact_kda_layers()
        if len(compact_replay_layers) != getattr(
            self, "_compact_replay_layer_count", -1
        ):
            self._compact_replay_layer_count = len(compact_replay_layers)
        active_replays = [
            (layer, layer.replay_compact_spec_state)
            for layer in compact_replay_layers
            if getattr(layer, "compact_spec_replay", False)
        ]
        # Tensor staging is part of the captured graph, whereas Python flags
        # set by the model forward are not replayed.  Use the current input's
        # draft count to re-arm this commit after every graph execution.
        if active_replays and input_batch.num_draft_tokens > 0:
            sequence_mask = active_replays[0][0].compact_replay_sequence_mask()
            accepted = num_sampled[: input_batch.num_reqs][
                sequence_mask[: input_batch.num_reqs]
            ].contiguous()
            for _, replay in active_replays:
                replay(accepted)

        if self.pp_handler is not None:
''',
    "V2 runner compact replay post-sampling hook",
)
