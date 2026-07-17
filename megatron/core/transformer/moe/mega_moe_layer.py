# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mega-EP (MegaMoE) MoE layer.

This is a standalone, functional-verification MoE layer that replaces Megatron's
token-dispatcher + grouped-expert pipeline with the fused expert-parallel kernel
("Mega-EP", a.k.a. MegaMoE) from the Triton-distributed project
(https://github.com/ByteDance-Seed/Triton-distributed).

The fused kernel (``TritonDistFusedEpMoeFunction``) performs the whole routed-MoE
path in a single autograd Function: intra-node all-to-all dispatch over the
expert-parallel (EP) group, grouped GEMM for ``fc1`` (gate/up), SwiGLU activation,
grouped GEMM for ``fc2`` (down), and the combine/scatter back to the original
token order -- all in BF16.

Design constraints (from the Triton-distributed kernel):
  * BF16 activations and weights only.
  * Gated SwiGLU experts only (gate + up + down).
  * Intra-node EP group with ``ep_size <= 8``; the kernel's backward asserts
    ``ep_size == 8``.
  * NVSHMEM must be initialized over the EP process group before first use.

This layer intentionally does NOT modify the existing ``MoELayer``. It is selected
via the ``moe_use_mega_ep`` config flag (see ``moe_module_specs.py``). The standard
``TopKRouter`` is reused unchanged; shared experts (if configured) run through the
standard ``SharedExpertMLP`` path and are summed into the output.
"""

import os
from typing import Optional

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size

# Module-level guard so NVSHMEM is initialized exactly once per process.
_GLOBAL_SETUP_DONE = False


def _ensure_global_setup(ep_group, max_tokens_per_rank):
    """One-time process-global Mega-EP setup: NVSHMEM init + shared stream +
    MAX_TOKENS_PER_RANK + the triton scratch allocator.

    Mirrors the global bookkeeping of ``init_triton_dist_ep_op`` but does NOT
    create the shared singleton operator — each MoE layer owns its own operator
    (see ``_create_mega_op``). This is required for multi-layer TRAINING: the
    operator's NVSHMEM comm buffers are reused in-place, so a single shared
    operator gets its layer-i forward state overwritten by later layers' forwards
    before layer-i's backward runs, corrupting gradients. Per-layer operators keep
    each layer's buffers alive until its own backward.
    """
    global _GLOBAL_SETUP_DONE
    if _GLOBAL_SETUP_DONE:
        return

    import triton
    from triton_dist.function.nvidia import common
    from triton_dist.utils import init_nvshmem_by_torch_process_group

    init_nvshmem_by_torch_process_group(ep_group)
    common.DITRON_EP_STREAM = torch.cuda.Stream()
    common.MAX_TOKENS_PER_RANK = max_tokens_per_rank

    triton.set_allocator(_zeroing_triton_alloc_fn)
    torch.distributed.barrier(ep_group)
    _GLOBAL_SETUP_DONE = True


def _zeroing_triton_alloc_fn(size, alignment, stream):
    """Triton scratch allocator that ZEROS its memory.

    The default allocator (torch.empty, used by init_triton_dist_ep_op) returns
    uninitialized memory. Once torch's caching allocator starts recycling freed
    blocks (after a few iterations), that scratch contains stale garbage; any
    kernel that reads/accumulates scratch before fully overwriting it then
    produces corrupted gradients. Zeroing makes every invocation deterministic.
    """
    return torch.zeros(size, device="cuda", dtype=torch.int8)


# Scratch buffers handed to triton.set_allocator are returned WITHOUT the runtime
# retaining a Python reference; their refcount drops to 0 right after the kernel
# launch captures the data_ptr, so torch's caching allocator may recycle that memory
# while the (async) kernel is still using it. In the full Megatron run many triton
# kernels share this one global allocator -> heavy recycling pressure -> the Mega-EP
# kernel's scratch gets clobbered mid-kernel. Retaining each scratch tensor (so it is
# never freed/recycled mid-run) removes that race. Bounded so memory does not grow
# unboundedly: a small per-size pool is reused.
_SCRATCH_KEEPALIVE = []


def _retaining_triton_alloc_fn(size, alignment, stream):
    """Triton scratch allocator that RETAINS every buffer (never freed/recycled) so
    torch cannot reuse in-use scratch mid-kernel. See _SCRATCH_KEEPALIVE note above.
    Retain-all (no reuse) to cleanly test the recycling hypothesis."""
    buf = torch.zeros(size, device="cuda", dtype=torch.int8)
    _SCRATCH_KEEPALIVE.append(buf)
    return buf


_GLOBAL_OP_INITED = False


def _ensure_global_op(ep_group, max_tokens_per_rank, hidden_size, topk, num_experts, dtype):
    """Official single global-singleton operator via init_triton_dist_ep_op (NVSHMEM
    init + heap sizing + op + allocator), exactly as the Triton-distributed unit test
    sets it up. Reused across iterations. Correct only for a single MoE layer."""
    global _GLOBAL_OP_INITED
    if _GLOBAL_OP_INITED:
        return
    import triton
    from triton_dist.function.nvidia.common import init_triton_dist_ep_op
    from triton_dist.utils import init_nvshmem_by_torch_process_group

    init_nvshmem_by_torch_process_group(ep_group)
    init_triton_dist_ep_op(
        ep_group, max_tokens_per_rank, hidden_size, topk, get_pg_rank(ep_group),
        num_experts, get_pg_size(ep_group), dtype=dtype, weight_dtype=torch.float32,
        num_sm=64, num_buffers=1, capacity=float(os.environ.get('MEGAMOE_CAPACITY', '4.0')),
    )
    # Scratch allocator choice (see notes on the alloc fns above):
    #   default                  -> retaining allocator (never recycles in-use scratch)
    #   MEGAMOE_SCRATCH_ALLOC=zero  -> torch.zeros, non-retained (legacy)
    #   MEGAMOE_SCRATCH_ALLOC=empty -> upstream torch.empty, non-retained (unit-test path)
    _alloc_mode = os.environ.get("MEGAMOE_SCRATCH_ALLOC", "retain")
    if _alloc_mode == "zero":
        triton.set_allocator(_zeroing_triton_alloc_fn)
    elif _alloc_mode == "empty":
        pass  # keep init_triton_dist_ep_op's own torch.empty allocator
    else:
        triton.set_allocator(_retaining_triton_alloc_fn)
    _GLOBAL_OP_INITED = True


def _full_reset_op(op):
    """Fully zero the operator's NVSHMEM counter/barrier buffers before each forward.

    The kernel's combine_preprocess only resets the [0:M_recv] slice of the combine
    counter/barrier buffers, leaving the tail stale; across training iterations the
    backward combine can read/accumulate that stale state, making gradients diverge.
    A full reset each step keeps every invocation starting from a clean buffer.
    Enabled by env MEGAMOE_FULL_RESET=1.
    """
    for name, fill in (
        ("mega_combine_counter_buf", 0),
        ("mega_combine_barrier_buf", 0),
        ("mega_combine_scatter_output_barrier_buf", -1),
        ("mega_dispatch_counter_buf", 0),
        ("mega_dispatch_barrier_buf", 0),
    ):
        buf = getattr(op, name, None)
        if buf is not None:
            buf.fill_(fill)


def _create_mega_op(ep_group, max_tokens_per_rank, hidden_size, topk, num_experts, dtype):
    """Create a per-layer Mega-EP operator with its own NVSHMEM comm buffers."""
    from triton_dist.layers.nvidia.ep_a2a_fused_layer import EpAll2AllFusedOp

    ep_size = get_pg_size(ep_group)
    op = EpAll2AllFusedOp(
        ep_group,
        max_tokens_per_rank,
        hidden_size,
        topk,
        get_pg_rank(ep_group),
        num_experts,
        min(8, ep_size),
        ep_size,
        dtype=dtype,
        weight_dtype=torch.float32,
        num_sm=64,
        sm_margin=0,
        duplicate_comm_buffer=1,
        capacity=float(os.environ.get('MEGAMOE_CAPACITY', '4.0')),
        FWD_GEMM_BLOCK_SIZE_N=256,
        need_reversed_token_scatter_idx=True,
        lazy=True,
    )
    op.sync()  # collective: allocate the symmetric NVSHMEM buffers
    torch.distributed.barrier(ep_group)
    return op


class MegaMoELayer(BaseMoELayer):
    """MoE layer backed by the Triton-distributed Mega-EP fused kernel.

    Drop-in alternative to :class:`MoELayer`; same constructor signature so it can
    be substituted in the MoE module spec. Reuses the standard router and replaces
    dispatch + experts + combine with the fused kernel.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.submodules = submodules

        # --- Validate kernel constraints ---------------------------------------
        ep_size = get_pg_size(self.ep_group)
        assert ep_size <= 8, (
            f"Mega-EP is intra-node only (ep_size <= 8); got ep_size={ep_size}."
        )
        assert config.gated_linear_unit, (
            "Mega-EP kernel implements gated SwiGLU experts; set gated_linear_unit=True."
        )
        assert get_pg_size(self.attn_tp_group) == 1, (
            "Mega-EP functional integration assumes tensor-parallel size 1 "
            f"(got tp={get_pg_size(self.attn_tp_group)})."
        )

        self.hidden_size = config.hidden_size
        self.moe_ffn_hidden_size = config.moe_ffn_hidden_size or config.ffn_hidden_size
        self.topk = config.moe_router_topk
        self.num_experts = config.num_moe_experts

        # Buffer sizing for the fused kernel's symmetric NVSHMEM allocation. May be
        # overridden via env for large sequence lengths.
        self.max_tokens_per_rank = int(
            os.environ.get("MEGAMOE_MAX_TOKENS_PER_RANK", str(8192 * 4))
        )

        # Per-layer Mega-EP operator (own NVSHMEM comm buffers) + own side-stream,
        # created lazily on first forward. Distinct buffers AND stream per layer are
        # required for correct multi-layer training backward (the kernel captures
        # both into the autograd ctx; sharing them lets later layers' fwd/bwd
        # overwrite an earlier layer's saved state before its backward runs).
        self._mega_op = None
        self._mega_stream = None

        # --- Router (reuse standard TopKRouter) --------------------------------
        router_cls = submodules.router if submodules is not None else TopKRouter
        self.router = router_cls(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
        )

        # --- Routed-expert weights (BF16, sharded across EP) -------------------
        # fc1 is gate/up SwiGLU: gate -> fc1_gate, up -> fc1_up.
        # Shapes match the Triton-distributed kernel:
        #   fc1_gate/fc1_up: [num_local_experts, moe_ffn_hidden_size, hidden_size]
        #   fc2:             [num_local_experts, hidden_size, moe_ffn_hidden_size]
        e = self.num_local_experts
        f = self.moe_ffn_hidden_size
        h = self.hidden_size
        device = torch.cuda.current_device()

        self.fc1_gate = torch.nn.Parameter(
            torch.empty(e, f, h, dtype=torch.bfloat16, device=device)
        )
        self.fc1_up = torch.nn.Parameter(
            torch.empty(e, f, h, dtype=torch.bfloat16, device=device)
        )
        self.fc2 = torch.nn.Parameter(
            torch.empty(e, h, f, dtype=torch.bfloat16, device=device)
        )
        self._init_expert_weights()

        # --- Shared experts (optional, standard path) --------------------------
        if self.use_shared_expert:
            assert (
                submodules is not None and submodules.shared_experts is not None
            ), "Shared experts requested but no shared_experts builder in submodules."
            assert not self.shared_expert_overlap, (
                "Mega-EP does not support shared-expert overlap; "
                "set moe_shared_expert_overlap=False."
            )
            self.shared_experts = submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )

    def _init_expert_weights(self):
        """Initialize expert weights and tag them as expert-parallel parameters.

        ``param.allreduce = False`` tells Megatron's DDP/grad-buffer machinery to
        treat these as expert-parallel params (reduced over the expert-data-parallel
        group instead of the global data-parallel group); see
        ``param_and_grad_buffer.py`` (``is_expert_parallel = not param.allreduce``).
        """
        init_method = self.config.init_method
        output_init_method = self.config.output_layer_init_method or init_method
        with torch.no_grad():
            # init_method expects fp32-ish tensors; init on a float copy then cast.
            for p, im in (
                (self.fc1_gate, init_method),
                (self.fc1_up, init_method),
                (self.fc2, output_init_method),
            ):
                tmp = torch.empty_like(p, dtype=torch.float32)
                im(tmp)
                p.copy_(tmp.to(torch.bfloat16))

        for p in (self.fc1_gate, self.fc1_up, self.fc2):
            # Expert-parallel: skip the global DP all-reduce of grads.
            setattr(p, 'allreduce', False)

    def _build_topk_inputs(self, probs: torch.Tensor, routing_map: torch.Tensor):
        """Convert dense router outputs to the kernel's per-token top-k format.

        Args:
            probs: [num_tokens, num_experts] routing weights (0 for unselected).
            routing_map: [num_tokens, num_experts] bool mask, exactly ``topk`` True
                per row.

        Returns:
            expert_index: [num_tokens, topk] int32 expert ids.
            gate_weights: [num_tokens, topk] float32 routing weights.
        """
        num_tokens = routing_map.shape[0]
        per_token = routing_map.sum(dim=1)
        assert torch.all(per_token == self.topk), (
            "Mega-EP requires exactly moe_router_topk experts per token "
            "(token dropping / capacity not supported)."
        )
        # nonzero() returns row-major (token, expert) pairs sorted by token then
        # expert; the boolean-mask gather of probs follows the same order, so the
        # two outputs stay aligned per (token, slot).
        expert_index = (
            torch.nonzero(routing_map, as_tuple=False)[:, 1]
            .view(num_tokens, self.topk)
            .to(torch.int32)
        )
        gate_weights = probs[routing_map].view(num_tokens, self.topk).to(torch.float32)
        return expert_index.contiguous(), gate_weights.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass: route -> fused Mega-EP MoE -> (+ shared experts).

        Args:
            hidden_states: [seq_length, batch, hidden_size].

        Returns:
            (output [seq_length, batch, hidden_size], None).
        """
        from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction

        in_shape = hidden_states.shape
        in_dtype = hidden_states.dtype
        h = self.hidden_size

        # Route on the original [S, B, H] tensor (router flattens internally).
        probs, routing_map = self.router(hidden_states, padding_mask, input_ids)
        expert_index, gate_weights = self._build_topk_inputs(probs, routing_map)

        tokens = hidden_states.reshape(-1, h).to(torch.bfloat16).contiguous()

        if not os.environ.get("MEGAMOE_PERLAYER_OP"):
            # DEFAULT: official global-singleton path (exactly like the Triton-distributed
            # unit test's init_triton_dist_ep_op). Most stable single-layer behaviour
            # (no NaN). Note: the Mega-EP kernel itself diverges after ~3 training steps
            # (an upstream numerical bug, not this integration) and multiple MoE layers
            # share one operator. Set MEGAMOE_PERLAYER_OP=1 for per-layer operators
            # (distinct buffers per layer; helps multi-layer but does not fully fix it).
            _ensure_global_op(
                self.ep_group, self.max_tokens_per_rank, h, self.topk, self.num_experts,
                torch.bfloat16,
            )
        else:
            # One-time global setup + this layer's own operator and side-stream.
            _ensure_global_setup(self.ep_group, self.max_tokens_per_rank)
            if self._mega_op is None:
                self._mega_op = _create_mega_op(
                    self.ep_group, self.max_tokens_per_rank, h, self.topk, self.num_experts,
                    torch.bfloat16,
                )
                self._mega_stream = torch.cuda.Stream()
            # Point the kernel's globals at THIS layer's operator and stream so the
            # autograd ctx (built inside the fused Function) captures this layer's
            # buffers + stream; backward then reads them from the ctx, immune to other
            # layers overwriting the globals.
            from triton_dist.function.nvidia import common as _td_common
            _td_common.triton_dist_ep_op = self._mega_op
            _td_common.DITRON_EP_STREAM = self._mega_stream

        if os.environ.get("MEGAMOE_FULL_RESET"):
            from triton_dist.function.nvidia import common as _td_common
            _full_reset_op(_td_common.triton_dist_ep_op)

        if os.environ.get("MEGAMOE_SYNC_STREAM"):
            # Run the kernel's overlapped comm/compute on the MAIN stream instead of
            # its private side-stream. The fused kernel's cross-stream synchronization
            # has a gap that corrupts gradients after a few steps (the failure is
            # timing-dependent — a race). Folding the side-stream onto the current
            # stream serializes everything and removes the race (loses async overlap).
            from triton_dist.function.nvidia import common as _td_common
            _td_common.DITRON_EP_STREAM = torch.cuda.current_stream()

        output = TritonDistFusedEpMoeFunction.apply(
            self.num_experts,
            gate_weights,
            expert_index,
            tokens,
            self.fc1_gate,
            self.fc1_up,
            self.fc2,
            self.ep_group,
        )

        if os.environ.get("MEGAMOE_DEBUG") and get_pg_rank(self.ep_group) == 0:
            # Diagnostic: per-tensor fwd/bwd grad norms. Used to root-cause the
            # Mega-EP kernel's multi-layer / repeated-use backward-buffer issue.
            ln = self.layer_number
            for name, t in (
                ("out", output), ("tokens", tokens), ("gate", gate_weights),
                ("fc1g", self.fc1_gate), ("fc2", self.fc2),
            ):
                if t.requires_grad:
                    t.register_hook(
                        lambda g, n=name, ln=ln: print(
                            f"[MEGAMOE_DBG L{ln}] d_{n}|{g.float().norm():.3e}", flush=True
                        )
                    )

        output = output.view(in_shape).to(in_dtype)

        if self.use_shared_expert:
            output = output + self.shared_experts(hidden_states)

        return output, None
