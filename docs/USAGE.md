# Building, running and testing

## Building and running

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install torch numpy python-chess

cargo build --release                      # search without the network
cargo build --release --features neural    # with the network, on CPU (candle)
cargo build --release --features cuda      # with the network on the GPU (needs nvcc >= 12.8 for RTX 50-series)
cargo build --release --features cudnn     # as cuda, with cuDNN convolutions (fastest; libcudnn.so on LIBRARY_PATH)
./target/release/caissawary                # the engine, speaking the Universal Chess Interface (UCI)

# Inference settings, read from the environment by self_play and evaluate_models
KF_PRECISION=f32        # f16 (default on CUDA with cudnn), f32 (default otherwise) or bf16
KF_CUDA_GRAPHS=0        # turn off replaying each half-precision batch as a recorded CUDA graph
KF_PIPELINE=0           # launch a batch only after the previous one returns
KF_FULL_POLICY=1        # read back the full policy and compute legal-move priors on the CPU
KF_INFER_CPUS=0,16      # pin the batching thread to these CPUs, and search threads to the others
KF_RESERVE_MOVES=1      # start each move list with room for 256 moves
KF_LOG_MATE_LEAVES=f:N  # self_play: write every Nth position searched for mates to file f
# self_play flags: --batch-size N (cap, default 128), --inference-workers N (batching threads,
# default 4), --threads N (default: one per game), --syzygy DIR, --mate-checks N (default 5),
# --mate-in N, --mate-quiet M, --no-mate-search (the last five also in evaluate_models),
# --watch FILE (start each game with the newest weights named in FILE, for continuous training)
./target/release/mate_bench positions.fen 5 5,0,0 2,2,0   # time the mate search: depth,every-move depth,quiet moves
./target/release/mate_bench positions.fen 5 --orders all --half tune   # compare move orders on half the positions

cd cuda && mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 && make
```

## Testing

About 1,400 tests: about 900 in Rust, 314 in CUDA and 237 in Python. The fast `cargo test` suite takes
about 50 s. See [TESTING.md](../TESTING.md).

```bash
cargo test                                        # fast Rust tests (~50 s)
cargo test --features neural                      # plus the network, compared with PyTorch reference outputs
cargo test --features cudnn --test unit_tests     # plus cuDNN and channels-last tests
cargo test --features cudnn --test cuda_graph_tests  # CUDA graph replay, in a process of its own
cargo test --features slow-tests                  # full suite, including move-generation counts (~200 s)
cd python && python -m pytest test_*.py -v        # training pipeline
cuda/build/test_mcts_kernel                       # GPU search
cuda/build/test_transformer
cuda/build/test_block_ops
cuda/build/test_budget_reuse
cuda/build/test_nn_ops
cuda/build/bench_throughput 100 200 36 [p2]       # end-to-end self-play throughput
cuda/build/bench_forward_b2 1000                  # batched convolution microbenchmark
cargo test --release --features cudnn --test nn_bench -- --ignored --nocapture   # network latency by batch size
```

## Binaries

| Binary | Description |
|--------|-------------|
| `caissawary` | The engine (UCI) |
| `self_play` | Self-play data generation with game logs |
| `evaluate_models` | Head-to-head model evaluation with SPRT |
| `round_robin` | Tournament with Elo estimation |
| `mcts_inspector` | Search tree visualization (Graphviz DOT) |
| `verbose_search` | Real-time search narration |
| `verbose_game` | Full game between two classical engines |
| `benchmark` | Performance testing |
| `strength_test` | Engine strength assessment |
| `run_experiments` | Ablation studies |
| `elo_tournament` | Elo estimation for classical engines |
| `texel_tune` | Evaluation weight tuning |
| `generate_training_data` | Standalone training data generation |
| `play_engine` | Play the engine yourself |
| `profile_engine` | Per-operation timing profiler |
| `qsearch_gui` | Interactive quiescence-search tree viewer (web UI at localhost:8088) |

