# Testing Guide for Caissawary

~1,350 tests (829 Rust + 314 CUDA + 203 Python) covering the engine, search, GPU kernels, and training pipeline. The fast `cargo test` suite runs ~750 of the Rust tests in ~50s.

## Quick Start

```bash
cargo test                                        # Fast Rust tests (~50s, ~750 tests)
cargo test --features slow-tests                  # Full suite including perft (~200s)
cd python && python -m pytest test_*.py -v        # Python pipeline tests (~175 tests)
./scripts/test.sh                                 # Scripted run (unit → integration → property)
```

## Rust Test Categories

### Unit Tests (`tests/unit/`)

Focus on individual components in isolation:
- **Board:** FEN parsing, state representation, castling rights
- **Move Generation:** Validity of moves, pseudo-legal vs legal generation
- **Node:** MCTS node value logic, terminal state handling
- **Selection:** UCB/PUCT calculations, tactical priority ordering
- **Make Move:** Incremental Zobrist hashing, castling updates
- **SEE:** Static exchange evaluation
- **Quiescence:** `forced_material_balance()`, material-only Q-search
- **Training Data:** Binary serialization, sample extraction

### Integration Tests (`tests/integration/`)

Test the interaction between subsystems:
- **Mate Search:** Verifies the engine finds mates in complex positions
- **Tactical Priority:** Ensures tactical moves (captures, checks) are prioritized
- **Neural Integration:** Tests the flow between the search tree and (mocked) inference server
- **KOTH:** King of the Hill win detection and geometric pruning

### Property Tests (`tests/property/`)

Uses `proptest` to generate random inputs and verify invariants:
- **Legal Moves:** Random positions ensure `generate_legal_moves` never produces illegal states
- **Value Domains:** Verifies evaluations stay within valid bounds

### Perft Tests (`tests/perft_tests.rs`)

Performance and correctness tests for move generation (gated behind `--features slow-tests`). Walk the game tree to a fixed depth and compare leaf counts against known correct values.

## Python Tests

Located in `python/`:

| File | Coverage |
|------|----------|
| `test_orchestrate.py` | Training loop, SPRT gating, state management, eval data reuse |
| `test_train.py` | Model training, loss computation, data loading, chunk refreshing |
| `test_augmentation.py` | D4/horizontal flip transforms, policy vector permutation |
| `test_replay_buffer.py` | Elo-based weighting, FIFO eviction, manifest handling |
| `test_architectures.py` | OracleNet variants, k-head, SE-ResNet blocks |

```bash
cd python && python -m pytest test_*.py -v
```

## Running Specific Tests

```bash
# Run only unit tests
cargo test --test unit_tests

# Run a specific test case
cargo test test_castling_blocked_by_check

# Run a specific Python test file
cd python && python -m pytest test_augmentation.py -v
```
