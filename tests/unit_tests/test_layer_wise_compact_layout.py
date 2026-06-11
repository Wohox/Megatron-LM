# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math

import torch

from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    compute_layer_wise_param_ownership,
)


def _make_param(numel: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.empty(numel, dtype=torch.bfloat16))


def test_compact_ddp_layout_is_not_layerwise_padded():
    """Compact DDP layout should stay near raw param size, not DP * max param."""
    dp_size = 64
    params = [_make_param(numel) for numel in (4096, 2048, 1024, 512)]
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)

    layout = DistributedOptimizer._compute_per_buffer_param_layout(
        params=params,
        bucket_size=None,
        data_parallel_world_size=dp_size,
        ddp_config=ddp_config,
        param_indices=list(range(len(params))),
    )

    compact_numel = layout.bucket_indices[-1][1]
    raw_numel = sum(param.numel() for param in params)
    alignment_slack_bound = 64 * len(params) + math.lcm(dp_size, 128)

    assert compact_numel <= raw_numel + alignment_slack_bound
    assert compact_numel < dp_size * max(param.numel() for param in params)


def test_layerwise_logical_ownership_assigns_whole_params_once():
    params = [_make_param(numel) for numel in (100, 90, 40, 30)]

    ownership = compute_layer_wise_param_ownership(params, world_size=2)

    assert {id(param) for param in ownership.param_to_owner_rank.keys()} == {
        id(param) for param in params
    }
    assert sorted(ownership.owner_rank_loads) == [130, 130]
    owner_param_ids = [
        id(param) for owner_params in ownership.owner_rank_to_params for param in owner_params
    ]
    assert sorted(owner_param_ids) == sorted(id(param) for param in params)


def test_compact_shard_intersection_splits_param_fragments():
    param_start, param_end = 50, 250
    bucket_start, bucket_end = 0, 320
    world_size = 4

    intersections = [
        LayerWiseDistributedOptimizer._local_shard_intersection(
            param_start, param_end, bucket_start, bucket_end, rank, world_size
        )
        for rank in range(world_size)
    ]

    assert intersections == [(50, 80), (80, 160), (160, 240), (240, 250)]
