# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.backends import (
    BackendSpecProvider,
    InferenceSpecProvider,
    LocalSpecProvider,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_layer import MlpBuilder


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> MlpBuilder:
    """Helper function to get module spec for MoE.

    Called by hybrid_layer_specs.py for standard (non-inference) MoE specs.
    The GPT layer specs call get_moe_module_spec_for_backend directly.

    Args:
        use_te: Whether to use Transformer Engine.
        num_experts: Number of experts.
        moe_grouped_gemm: Whether to use grouped GEMM.
    """
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> MlpBuilder:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    experts = backend.grouped_mlp_modules(moe_grouped_gemm is not None and moe_grouped_gemm)
    # shared experts spec
    shared_experts = partial(SharedExpertMLP, submodules=mlp)

    submodules = MoESubmodules(experts=experts, shared_experts=shared_experts)

    # MoE module spec. The builder selects MegaMoELayer at layer-construction time
    # when config.moe_use_mega_ep is set, otherwise the standard MoELayer. The class
    # choice is deferred to call time because the spec is built before the config is
    # available in some flows.
    return partial(_build_moe_layer, submodules=submodules)


def _build_moe_layer(*args, submodules: MoESubmodules, **kwargs):
    """MlpBuilder that dispatches to MegaMoELayer or MoELayer based on config."""
    config = kwargs.get("config")
    if config is None and args:
        config = args[0]
    if config is not None and getattr(config, "moe_use_mega_ep", False):
        # Imported lazily so Megatron stays importable without Triton-distributed.
        from megatron.core.transformer.moe.mega_moe_layer import MegaMoELayer

        return MegaMoELayer(*args, submodules=submodules, **kwargs)
    return MoELayer(*args, submodules=submodules, **kwargs)


def get_inference_optimized_moe_spec() -> MlpBuilder:
    """MoE module spec for inference-optimized transformer impl.

    Uses InferenceSpecProvider to select inference-optimized modules:
    InferenceTopKRouter, InferenceGroupedMLP. MoELayer detects inference mode
    via config.transformer_impl and sets up the inference dispatcher internally.

    Called by hybrid_layer_specs.py and gpt_layer_specs.py.
    """
    backend = InferenceSpecProvider()
    activation_func = backend.activation_func()

    experts = backend.grouped_mlp_modules(True)
    shared_experts = partial(
        SharedExpertMLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
            activation_func=activation_func,
        ),
    )

    return partial(
        MoELayer,
        submodules=MoESubmodules(
            router=InferenceTopKRouter, experts=experts, shared_experts=shared_experts
        ),
    )
