# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size

from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer, dist_all_gather_func
from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class LayerWiseParamOwnership:
    """Whole-parameter ownership for LayerWise optimizer state."""

    param_to_owner_rank: Dict[torch.nn.Parameter, int]
    owner_rank_to_params: List[List[torch.nn.Parameter]]
    owner_rank_loads: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class _CompactParamBufferInfo:
    buffer: _ParamAndGradBuffer
    param_start: int
    param_end: int
    bucket_id: int


def compute_layer_wise_param_ownership(
    params: Iterable[torch.nn.Parameter], world_size: int
) -> LayerWiseParamOwnership:
    """Assign whole params to logical LayerWise owner ranks using LPT bin packing."""
    assert world_size >= 1
    owner_rank_to_params: List[List[torch.nn.Parameter]] = [[] for _ in range(world_size)]
    owner_rank_loads = [0 for _ in range(world_size)]
    param_to_owner_rank: Dict[torch.nn.Parameter, int] = {}

    indexed_params = list(enumerate(params))
    indexed_params.sort(key=lambda item: (-item[1].numel(), item[0]))
    for _, param in indexed_params:
        owner_rank = min(range(world_size), key=lambda rank: (owner_rank_loads[rank], rank))
        owner_rank_to_params[owner_rank].append(param)
        owner_rank_loads[owner_rank] += param.numel()
        param_to_owner_rank[param] = owner_rank

    return LayerWiseParamOwnership(
        param_to_owner_rank=param_to_owner_rank,
        owner_rank_to_params=owner_rank_to_params,
        owner_rank_loads=owner_rank_loads,
    )


class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Experimental distributed optimizer wrapper that distributes weight to DP ranks by layer.
    Implemented as ChainedOptimizer to support multiple optimizers (e.g. muon + adamW)
    Legacy mode uses full DDP gradients and LayerWise param all-gather. Compact-DDP mode
    keeps DDP's DistributedOptimizer-style compact param/grad buffers while retaining
    whole-parameter LayerWise optimizer ownership.

    How LayerWiseDistributedOptimizer work:
    1. weights are split into lists and each rank only keeps its shard in its optimizer
    2. DDP reduces gradients; compact mode reconstructs logical-owner full grads after RS
    3. optimizer is modified so only params that belong to this DP rank are updated
    4. grad_norm and zero counting reduce metrics globally in step
    5. regular chained optimizers update the local logical-owner params
    6. updated params are gathered or staged back into compact DDP shards, then all-gathered
    """

    def __init__(
        self,
        optimizers: List[MegatronOptimizer],
        config: OptimizerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        init_state_fn_list: Optional[List[Callable]] = None,
        model_chunks: Optional[List] = None,
    ) -> None:
        """
        Initialize LayerWiseDistributedOptimizer.

        Args:
            optimizers: List of MegatronOptimizers.
            config: OptimizerConfig.
            pg_collection: ProcessGroupCollection.
            init_state_fn_list: List of init state functions.
            model_chunks: DDP-wrapped model chunks (needed for overlap_param_gather).
        """

        self.pg_collection = pg_collection
        model_chunks = model_chunks or []
        self.model_chunks = model_chunks
        self.use_compact_ddp_layout = config.use_layer_wise_compact_ddp_layout
        if self.use_compact_ddp_layout:
            assert self.pg_collection is not None, "pg_collection is required for compact LayerWise"
            assert model_chunks, "model_chunks are required for compact LayerWise"
            if config.reuse_grad_buf_for_mxfp8_param_ag:
                raise NotImplementedError(
                    "LayerWise compact DDP layout does not yet support "
                    "reuse_grad_buf_for_mxfp8_param_ag / FP8 parameter gather."
                )
            for model_chunk in model_chunks:
                assert model_chunk.ddp_config.use_distributed_optimizer, (
                    "LayerWise compact DDP layout requires DDP use_distributed_optimizer=True"
                )
                if model_chunk.ddp_config.fp8_param_gather:
                    raise NotImplementedError(
                        "LayerWise compact DDP layout does not implement FP8 parameter gather."
                    )
                if model_chunk.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                    raise NotImplementedError(
                        "LayerWise compact DDP layout does not support "
                        "reuse_grad_buf_for_mxfp8_param_ag / FP8 parameter gather."
                    )
                assert model_chunk.ddp_config.num_distributed_optimizer_instances == 1, (
                    "LayerWise compact DDP layout currently supports one distributed optimizer "
                    "instance per data-parallel group"
                )
        self.shard_params(optimizers)
        self._compact_param_buffer_infos: Dict[torch.nn.Parameter, _CompactParamBufferInfo] = {}
        if self.use_compact_ddp_layout:
            self._build_compact_param_buffer_infos(model_chunks)

        # Set up overlap param gather using DDP bucket infrastructure.
        self.overlap_param_gather = config.overlap_param_gather
        if self.overlap_param_gather and not self.use_compact_ddp_layout:
            assert (
                model_chunks is not None
            ), "model_chunks must be provided if overlap_param_gather is True"
            self.set_bucket_layerwise_params_list(model_chunks)

        if init_state_fn_list:
            assert len(init_state_fn_list) == len(
                optimizers
            ), "init_state_fn_list must be the same length as optimizers if provided"

        # Wrap base torch optimizers with Float16 for bf16 training.
        # Callers pass base optimizers; wrapping happens here *after*
        # shard_params so master weights are only created for the local shard.
        if config.bf16:
            for i in range(len(optimizers)):
                opt = optimizers[i]
                if isinstance(opt, (Float16OptimizerWithFloat16Params, FP32Optimizer)):
                    raise TypeError(
                        'LayerWiseDistributedOptimizer expects base torch optimizers, '
                        f'got {type(opt).__name__}. Do not pre-wrap with Megatron optimizers.'
                    )
                optimizers[i] = Float16OptimizerWithFloat16Params(
                    opt, config, None, init_state_fn_list[i] if init_state_fn_list else None
                )

        super().__init__(optimizers)
        self.model_chunks = model_chunks

        # TODO(kunlun, deyuf): potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in
        # contiguous gbuf. So instead of shard param by layer randomly, we can shard by
        # buf range but keep some "extras" to keep boundary weight not sharded.
        # This way each rank do some duplicated work but allgather_v is no longer needed
        # All current distopt optimization can also be potentially applied

    def shard_params(self, optimizers):
        """Shard all params into lists by rank."""
        if self.use_compact_ddp_layout:
            self._shard_params_by_logical_ownership(optimizers)
            return

        # list of parameter are sorted by numel and assigned to ranks in ping-pong style
        # example of 4 ranks and 10 parameters p0-p9 after sorting, then dp_cp_params_list will be
        # [[p0, p7, p8], [p1, p6, p9], [p2, p5], [p3, p4]]

        # simplify when dp_cp group size is 1
        if get_pg_size(self.pg_collection.dp_cp) == 1:
            self.dp_cp_params_list = None
            self.expt_dp_params_list = None
            return

        dp_cp_idx, expt_dp_idx = 0, 0
        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)
        # create ping-pong style loop so memory is more balanced
        dp_cp_loop = list(range(dp_cp_size)) + list(range(dp_cp_size))[::-1]
        expt_dp_loop = list(range(expt_dp_size)) + list(range(expt_dp_size))[::-1]
        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]
        # get all param groups
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups

        # sort param in all groups by param numel and assign to each rank evenly
        param_list = []
        for group_index, group in enumerate(param_groups):
            for p in group["params"]:
                param_list.append((p, group_index))
        param_list.sort(key=lambda x: x[0].numel())
        param_groups_this_rank = [[] for g in param_groups]

        # assign params to rank in ping-pong style loop
        for p, group_index in param_list:
            if param_groups[group_index].get("is_expert_parallel", False):
                if expt_dp_loop[expt_dp_idx] == get_pg_rank(self.pg_collection.expt_dp):
                    param_groups_this_rank[group_index].append(p)
                self.expt_dp_params_list[expt_dp_loop[expt_dp_idx]].append(p)
                expt_dp_idx = (expt_dp_idx + 1) % len(expt_dp_loop)
            else:
                if dp_cp_loop[dp_cp_idx] == get_pg_rank(self.pg_collection.dp_cp):
                    param_groups_this_rank[group_index].append(p)
                self.dp_cp_params_list[dp_cp_loop[dp_cp_idx]].append(p)
                dp_cp_idx = (dp_cp_idx + 1) % len(dp_cp_loop)

        # now we modify the group to only handle local params
        for groups, params in zip(param_groups, param_groups_this_rank):
            groups["params"] = params

        # simplify when expt_dp group size is 1 or expert parallel is off
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    def _shard_params_by_logical_ownership(self, optimizers):
        """Shard optimizer param groups by whole-param logical ownership."""
        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)
        dp_cp_rank = get_pg_rank(self.pg_collection.dp_cp)
        expt_dp_rank = get_pg_rank(self.pg_collection.expt_dp)

        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups

        dense_entries = []
        expert_entries = []
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                entry = (param, group_index)
                if group.get("is_expert_parallel", False):
                    expert_entries.append(entry)
                else:
                    dense_entries.append(entry)

        dense_ownership = compute_layer_wise_param_ownership(
            [param for param, _ in dense_entries], dp_cp_size
        )
        expert_ownership = compute_layer_wise_param_ownership(
            [param for param, _ in expert_entries], expt_dp_size
        )
        self.dp_cp_ownership = dense_ownership
        self.expt_dp_ownership = expert_ownership
        self.dp_cp_params_list = dense_ownership.owner_rank_to_params
        self.expt_dp_params_list = expert_ownership.owner_rank_to_params

        param_groups_this_rank = [[] for _ in param_groups]
        for param, group_index in dense_entries:
            if dense_ownership.param_to_owner_rank[param] == dp_cp_rank:
                param_groups_this_rank[group_index].append(param)
        for param, group_index in expert_entries:
            if expert_ownership.param_to_owner_rank[param] == expt_dp_rank:
                param_groups_this_rank[group_index].append(param)

        for group, params in zip(param_groups, param_groups_this_rank):
            group["params"] = params

        if dp_cp_size == 1:
            self.dp_cp_params_list = None
        if expt_dp_size == 1 or len(expert_entries) == 0:
            self.expt_dp_params_list = None

    def _build_compact_param_buffer_infos(self, model_chunks):
        """Index DDP compact param-buffer locations by model parameter."""
        for model_chunk in model_chunks:
            for buffer in model_chunk.buffers + model_chunk.expert_parallel_buffers:
                for param, (param_start, param_end, bucket_id) in buffer.param_index_map.items():
                    self._compact_param_buffer_infos[param] = _CompactParamBufferInfo(
                        buffer=buffer,
                        param_start=param_start,
                        param_end=param_end,
                        bucket_id=bucket_id,
                    )

    @staticmethod
    def _local_shard_intersection(
        param_start: int,
        param_end: int,
        bucket_start: int,
        bucket_end: int,
        rank: int,
        world_size: int,
    ) -> Optional[Tuple[int, int]]:
        """Return the param/bucket intersection owned by ``rank`` in compact layout."""
        assert (bucket_end - bucket_start) % world_size == 0
        shard_numel = (bucket_end - bucket_start) // world_size
        shard_start = bucket_start + rank * shard_numel
        shard_end = shard_start + shard_numel
        start = max(param_start, shard_start)
        end = min(param_end, shard_end)
        if start >= end:
            return None
        return start, end

    def _all_gather_compact_grad_bucket_groups(self, bucket_groups):
        """Reconstruct full reduced grad buffers from compact reduce-scatter shards."""
        for bucket_group in bucket_groups:
            group = bucket_group.intra_distributed_optimizer_instance_group
            rank = bucket_group.intra_distributed_optimizer_instance_rank
            world_size = bucket_group.intra_distributed_optimizer_instance_size
            for bucket in bucket_group.buckets:
                local_shard = bucket.grad_data.chunk(world_size)[rank]
                dist_all_gather_func(bucket.grad_data, local_shard, group=group)
                for param in bucket.params_with_extra_main_grads:
                    if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                        param.main_grad.copy_(param.main_grad_copy_in_grad_buffer)

    @torch.no_grad()
    def redispatch_grads_from_compact_buffers(self) -> None:
        """Make compact DDP reduce-scattered grads visible to logical LayerWise owners."""
        if not self.use_compact_ddp_layout:
            return
        # Phase 1 correctness path: reconstruct compact reduced grad buffers on all
        # ranks using the existing full compact buffer storage. This oversends versus
        # owner-only routing but avoids LayerWise padding and leaves optimizer state
        # sharded by logical whole-param ownership.
        for model_chunk in self.model_chunks:
            self._all_gather_compact_grad_bucket_groups(model_chunk.bucket_groups)
            self._all_gather_compact_grad_bucket_groups(model_chunk.expert_parallel_bucket_groups)

    def _redispatch_updated_params_for_ownership(self, params_list, group):
        """Stage updated whole params into this rank's compact DDP param-buffer shard."""
        if not params_list:
            return
        rank = get_pg_rank(group)
        world_size = get_pg_size(group)
        for owner_rank, params in enumerate(params_list):
            if not params:
                continue
            src_global_rank = torch.distributed.get_global_rank(group, owner_rank)
            for param in params:
                info = self._compact_param_buffer_infos[param]
                buffer = info.buffer
                bucket_start, bucket_end = buffer.bucket_indices[info.bucket_id]
                intersection = self._local_shard_intersection(
                    info.param_start,
                    info.param_end,
                    bucket_start,
                    bucket_end,
                    rank,
                    world_size,
                )

                if rank == owner_rank:
                    full_param = param.data.detach()
                else:
                    full_param = torch.empty_like(param.data)
                # Phase 1 correctness path: broadcast each logical-owner full param,
                # then copy only the slice owned by this rank's compact DDP shard.
                torch.distributed.broadcast(full_param, src_global_rank, group=group)

                if intersection is None:
                    continue
                start, end = intersection
                src_start = start - info.param_start
                src_end = end - info.param_start
                buffer.param_data.view(-1)[start:end].copy_(
                    full_param.view(-1)[src_start:src_end]
                )

    @torch.no_grad()
    def redispatch_updated_params_to_compact_buffers(self) -> None:
        """Stage logical-owner updates into compact DDP shards before param all-gather."""
        if not self.use_compact_ddp_layout:
            return
        self._redispatch_updated_params_for_ownership(
            self.dp_cp_params_list, self.pg_collection.dp_cp
        )
        self._redispatch_updated_params_for_ownership(
            self.expt_dp_params_list, self.pg_collection.expt_dp
        )

    def set_bucket_layerwise_params_list(self, model_chunks):
        """Map sharded params to DDP buckets for async all-gather.

        For each bucket in each model chunk's bucket groups, build per-rank param lists
        by cross-referencing the layer-wise sharded param lists with the bucket's params.

        Args:
            model_chunks: DDP-wrapped model chunks with bucket_groups.
        """
        for model_chunk in model_chunks:
            for group in model_chunk.bucket_groups:
                for bucket in group.buckets:
                    bucket_params_list = [[] for _ in range(get_pg_size(self.pg_collection.dp_cp))]
                    for bucket_list, full_params_list in zip(
                        bucket_params_list, self.dp_cp_params_list
                    ):
                        for param in full_params_list:
                            if param in bucket.params:
                                bucket_list.append(param)
                    bucket.set_layerwise_params_list(bucket_params_list)
            # Do the same for expert parallel bucket groups.
            for group in model_chunk.expert_parallel_bucket_groups:
                for bucket in group.buckets:
                    if self.expt_dp_params_list is not None:
                        bucket_params_list = [
                            [] for _ in range(get_pg_size(self.pg_collection.expt_dp))
                        ]
                        for bucket_list, full_params_list in zip(
                            bucket_params_list, self.expt_dp_params_list
                        ):
                            for param in full_params_list:
                                if param in bucket.params:
                                    bucket_list.append(param)
                    else:
                        # expt_dp_size == 1: single rank owns all params, no
                        # all-gather needed but data structures must be initialized.
                        bucket_params_list = [list(bucket.params_list)]
                    bucket.set_layerwise_params_list(bucket_params_list)

    @torch.no_grad()
    def allgather_params(self) -> None:
        """All-gather updated params from all ranks."""
        if self.use_compact_ddp_layout:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync(force_sync=True)
            return

        # helper function to flatten local params, all-gather,
        # unflatten and copy to model params
        def _allgather_helper(params_list, group):
            device = params_list[0][0].device
            dtype = params_list[0][0].dtype
            rank = get_pg_rank(group)
            dp_size = get_pg_size(group)
            # Flatten this rank's params.
            src = (
                _flatten_dense_tensors(params_list[rank])
                if len(params_list[rank]) > 0
                else torch.empty(0, device=device, dtype=dtype)
            )
            flat_sizes = [sum(p.numel() for p in params) for params in params_list]
            if max(flat_sizes) == 0:
                return

            # Allocate per-rank receive buffers with actual sizes (no padding).
            # PyTorch's NCCL backend handles uneven sizes in all_gather via
            # grouped send/recv internally. Reuse src for local rank's slot.
            gather_list = []
            for i in range(dp_size):
                if i == rank:
                    gather_list.append(src)
                else:
                    gather_list.append(torch.empty(flat_sizes[i], device=device, dtype=dtype))

            torch.distributed.all_gather(gather_list, src, group=group)

            # Unflatten and copy gathered params for each rank.
            for idx, params in enumerate(params_list):
                if len(params) == 0 or idx == rank:
                    continue
                updated_params = _unflatten_dense_tensors(gather_list[idx], params)
                for updated_p, model_p in zip(updated_params, params):
                    model_p.data.copy_(updated_p)

        if self.pg_collection is None:
            return
        if self.dp_cp_params_list:
            _allgather_helper(self.dp_cp_params_list, self.pg_collection.dp_cp)
        if self.expt_dp_params_list:
            _allgather_helper(self.expt_dp_params_list, self.pg_collection.expt_dp)

    @torch.no_grad()
    def broadcast_params(self):
        """All rank broadcast updated local params."""
        # Broadcast linear layer weights to all other ranks. Kept as reference test.
        if self.dp_cp_params_list is None:
            return
        for i, params in enumerate(self.dp_cp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.dp_cp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.dp_cp)
        if self.expt_dp_params_list is None:
            return
        for i, params in enumerate(self.expt_dp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.expt_dp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.expt_dp)

    @torch.no_grad()
    def get_grad_norm(self):
        # similar to dist opt, always aggregate globally
        grads_for_norm = []
        for optimizer in self.chained_optimizers:
            grads_for_norm += optimizer.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        return grad_norm

    @torch.no_grad()
    def count_zeros(self):
        params = []
        for optimizer in self.chained_optimizers:
            params += optimizer.get_parameters()
        return count_zeros_fp32(
            params,
            grad_stats_parallel_group=None,
            use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
        )

    @torch.no_grad()
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer."""
        update_successful, grad_norm, num_zeros_in_grad = super().step()

        # All gather updated params. If overlap_param_gather is True, the allgather
        # is deferred to the forward pre-hooks via DDP bucket infrastructure.
        if update_successful and not self.overlap_param_gather:
            self.allgather_params()

        return update_successful, grad_norm, num_zeros_in_grad

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Prepare logical-owner gradients before the chained optimizers read them."""
        self.redispatch_grads_from_compact_buffers()
        return super().prepare_grads()

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step local logical-owner params and stage compact DDP param shards."""
        success = super().step_with_ready_grads()
        if success:
            self.redispatch_updated_params_to_compact_buffers()
        return success

    # TODO(deyuf): need to improve dist checkpointing design to properly handle this
    # fp32_from_fp16_params is list, each sub list could be empty if group is empty
    # this breaks dist checkpointing assumption since extract_sharded_base drop list structure
    # for now, we convert it to dict with index as key and convert back in load_state_dict
    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) == 1:
            wrapped_state_dict = {1: state_dict}
        else:
            wrapped_state_dict = state_dict
        for sd in wrapped_state_dict.values():
            if 'fp32_from_fp16_params' in sd and isinstance(sd['fp32_from_fp16_params'], dict):
                logger.info('[layerwise] converting fp32_from_fp16_params from dict to list')
                sd['fp32_from_fp16_params'] = [
                    v for k, v in sorted(sd['fp32_from_fp16_params'].items())
                ]
        super().load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        """
        Sharded state dict for torch_dist format checkpointing.
        For fixed DP usage only, set replica_id to 0 for all ShardedTensor.
        """
        sharded_state_dict = super().sharded_state_dict(
            model_sharded_state_dict, is_loading, **kwargs
        )

        # for fixed DP usage only
        for sh_base in nested_values(sharded_state_dict):
            if hasattr(sh_base, 'replica_id'):
                assert (
                    isinstance(sh_base.replica_id, int) or len(sh_base.replica_id) == 3
                ), f'Expected replica_id as int or (PP, TP, DP), got: {sh_base}'
                sh_base.replica_id = (
                    0 if isinstance(sh_base.replica_id, int) else (*sh_base.replica_id[:2], 0)
                )

        # later code assume list but chained optimizer fallback to non-list if there's only one
        if len(self.chained_optimizers) == 1:
            wrapped_sharded_state_dict = {1: sharded_state_dict}
        else:
            wrapped_sharded_state_dict = sharded_state_dict

        # Adjust dict rank 0 output correct global metadata into common_dict
        for sd in wrapped_sharded_state_dict.values():
            # wrap empty containers into LocalNonpersistentObject so it won't be saved/loaded
            # params is already wrapped, we only need to handle fp32_from_fp16_params and state
            # more details in load_state_dict comment
            if 'fp32_from_fp16_params' in sd:
                sd['fp32_from_fp16_params'][:] = [
                    group if group else LocalNonpersistentObject(group)
                    for group in sd['fp32_from_fp16_params']
                ]
                sd['fp32_from_fp16_params'] = {
                    i: v for i, v in enumerate(sd['fp32_from_fp16_params'])
                }
            # state is a single dict and will be empty if optimizer is fully empty
            if not sd['optimizer']['state']:
                sd['optimizer']['state'] = LocalNonpersistentObject(sd['optimizer']['state'])
            # group keys(e.g. 'step') might be missing or not updated
            for i, group in enumerate(sd['optimizer']['param_groups']):
                # keep local param tensor so we only gather metadata
                local_params = group.pop('params')
                # save whether this group is empty, so we can use non-empty rank for metadata
                group['params'] = bool(local_params.unwrap())
                all_rank_groups = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(all_rank_groups, group)
                # find first non-empty group if it exists
                nonempty_rank_group = next((g for g in all_rank_groups if g['params']), group)
                nonempty_rank_group['params'] = local_params
                sd['optimizer']['param_groups'][i] = nonempty_rank_group
        return sharded_state_dict

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer. For torch format only.
        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer. For torch format only."""
        super().load_state_dict(torch.load(filename))
