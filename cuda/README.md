# CUDA GPU MCTS / Transformer Suite

This directory holds the **GPU MCTS** and GPU transformer-inference code (kernels,
self-play driver, and their tests). It is distinct from the **Rust MCTS** in
`../src`.

## Why CUDA is not gated in CI

The CUDA tests **launch real kernels** and therefore require an NVIDIA GPU.
GitHub-hosted Actions runners (`ubuntu-latest`) have **no GPU**, so the suite
*cannot run* there. We do not fabricate a green CI job that never executes the
tests it claims to.

What CI *does* do: `.github/workflows/cuda-build.yml` installs `nvcc` and
**compiles** the CUDA sources (no kernel launch). That catches compile/link
regressions on every push touching `cuda/**`. Correctness must be verified
locally on a GPU, as below.

## Prerequisites

- NVIDIA GPU + driver
- CUDA toolkit (`nvcc`), tested with 12.0+
- CMake 3.18+
- Movegen attack tables in `cuda/tables/*.bin`. If missing, regenerate from the
  repo root with:
  ```bash
  cargo run --bin export_tables
  ```

## Build

From this `cuda/` directory (replace `89` with your GPU's compute capability —
89 = Ada / RTX 40xx / RTX 5060 Ti):

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
make -j
```

## Running the tests

Tests load `cuda/tables/*.bin` via a path relative to the current directory, so
run them **from the repository root**, not from `build/`:

```bash
# from the repo root
./cuda/build/test_selfplay
./cuda/build/test_transformer
./cuda/build/test_transformer_vs_pytorch
```

There is a **single GPU**: run GPU tests and throughput benchmarks
**one at a time**, never concurrently.

### Test-result semantics

`RUN_TEST` prints one of three outcomes per test:

- `OK`     — assertions ran and passed.
- `FAILED` — an assertion failed (non-zero process exit).
- `SKIP (reason)` — a prerequisite was missing (e.g. trained transformer
  weights at `weights/transformer4/candidate_1.bin`, or PyTorch reference
  dumps). **A skip is reported separately and does NOT count as a pass.** A
  green `OK` always means an assertion actually executed.

Skips are expected when running without trained weights / reference data; the
summary line reports them, e.g. `1/8 tests passed, 7 skipped`.

## Available targets

Tests: `test_tree_store`, `test_movegen`, `test_quiescence`, `test_quick_checks`,
`test_mcts_kernel`, `test_nn_ops`, `test_block_ops`, `test_multi_tree_eval`,
`test_transformer`, `test_selfplay`, `test_transformer_vs_pytorch`,
`test_selfplay_and_ops`, `test_pool_assertions`, `test_budget_reuse`.

Self-play driver: `selfplay`. Benchmarks / profiling: `bench_throughput`,
`bench_forward_b2`, `test_profile_latency`, `test_layer_timing`,
`test_batched_conv_scaling`.

## Self-play pool-size requirement

The self-play driver validates the node pool per tree (`selfplay.cu`,
`validate_pool_size`): `max_nodes_per_tree` must be at least
`max(MIN_POOL_PER_TREE=8192, sims_per_move * POOL_FACTOR_PER_SIM=35)`. Tests and
callers must size the pool accordingly or self-play aborts.
