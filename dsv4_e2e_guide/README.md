# DeepSeek-V4 Flash End-to-End Guide

## Contents

| File | Description |
|------|-------------|
| `Dockerfile` | Training container with all dependencies |
| `dsv4_flash_arg_example.sh` | Example launch script for DSv4 Flash pre-training |

Note: The example script does not exactly match DeepSeek's published settings (model architecture arguments are accurate, but training/infrastructure arguments may differ).

DeepSeek-v4 training has three stages:

| Stage | Additional args | Notes |
|-------|----------------|-------|
| Dense | `--csa-dense-mode` | All attention is dense (no sparse indexer) |
| Sparse warmup | _(neither of the two)_ | Only the DSA indexer is trained; rest of the model is frozen. This frozen pattern is not fully supported in Megatron-Core yet. |
| Sparse | `--dsa-indexer-use-sparse-loss` | Full model training with sparse attention loss |

## Links

- **Hybrid Attention kernels**: https://gitlab-master.nvidia.com/cudnn/cudnn_frontend/-/merge_requests/2037
- **DeepSeek-v4 dev branch PRs**:
  - https://github.com/NVIDIA/Megatron-LM/pull/4458
  - https://github.com/NVIDIA/Megatron-LM/pull/4481
  - https://github.com/NVIDIA/Megatron-LM/pull/4518
- **Prebuilt container**: `gitlab-master.nvidia.com/hongxiaob/containers/megatron_pytorch:mcore-moe-pytorch26.04-20260430-te01aef4f-hybridep1b8f467-dsa-arm`

## Known Issues

1. Muon optimizer is untested and may not work correctly with all argument combinations.
