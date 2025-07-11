# Caissawary Chess Engine (formerly Kingfisher)
## A Tactics-Enhanced Hybrid MCTS Engine with State-Dependent Search Logic

Caissawary is a chess engine that combines the strategic guidance of a modern Monte Carlo Tree Search (MCTS) with the ruthless tactical precision of classical search. Its unique, state-dependent search algorithm prioritizes forcing moves and minimizes expensive neural network computations to create a brutally efficient and tactically sharp engine.

![Caissawary Logo](Caissawary.png)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange)](https://rustup.rs/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## The Name: Caissawary
Like the engine itself, the name Caissawary is also a hybrid:

- **Caïssa**: The mythical goddess of chess, representing the engine's strategic intelligence and artistry.
- **Cassowary**: A large, formidable, and famously aggressive bird, representing the engine's raw tactical power and speed.

## Core Architecture
Caissawary's intelligence stems from how it handles each node during an MCTS traversal. Instead of a single, uniform approach, its behavior adapts based on the node's state, ensuring that cheap, powerful analysis is always performed before expensive strategic evaluation.

### The MCTS Node Handling Flow
When the MCTS search selects a node, its state determines the next action:

#### 1. If the node is a new LEAF (never visited):
It is evaluated immediately to determine its value.

- **Tier 1 (Mate Search)**: A fast, parallel mate search is run. If a mate is found, this becomes the node's value.
- **Tier 2 (Quiescence Eval)**: If no mate is found, a tactical quiescence search is run to get a stable, accurate evaluation score. This score becomes the leaf's value, which is then backpropagated.

#### 2. If the node is INTERNAL with unexplored TACTICAL moves:
The engine is forced to explore a tactical move first.

- A simple heuristic (e.g., MVV-LVA) selects the next capture or promotion to analyze. 
- Quiet moves are ignored until all tactical options at this node have been tried.

#### 3. If the node is INTERNAL with only QUIET moves left:
The engine engages the powerful ResNet policy network with a "lazy evaluation" strategy.

- **First Visit**: The policy network is called exactly once to compute and store the policy priors for all available quiet moves.
- **Subsequent Visits**: The standard UCB1 formula is used to select a move, using the already-stored policy priors without needing to call the network again.

## Tier 1: Detailed Parallel Mate Search
To find checkmates with maximum speed, the Tier 1 search is not a single algorithm but a portfolio of three specialized searches that run in parallel against a shared node budget. Each search has a different trade-off between speed and completeness:

### Search A: The "Spearhead" (Checks-Only)
- **Constraint**: The side to move can only play checking moves.
- **Behavior**: This search has a tiny branching factor, allowing it to reach immense depths within the node budget. It is designed to quickly find long, spectacular "check-check-check-mate" sequences.

### Search B: The "Flanker" (One Quiet Move)
- **Constraint**: The side to move can play at most one non-checking ("quiet") move in its entire sequence.
- **Behavior**: This search is slightly less deep than the Spearhead but can uncover mates that require a critical setup move (e.g., blocking an escape square).

### Search C: The "Guardsman" (Exhaustive)
- **Constraint**: No constraints on move types.
- **Behavior**: This is a standard, exhaustive alpha-beta search. It is the shallowest of the three but guarantees finding any mate that exists within its search depth, covering complex situations the other two might miss.

**Balancing**: The first search to find a mate terminates the others immediately. This portfolio approach ensures that the most efficient algorithm for the specific type of mate has the best chance of finding it within the allocated computational budget.

## Tier 2: Quiescence Search for Leaf Evaluation
When the MCTS traversal reaches a new leaf node and the Tier 1 search does not find a mate, the engine must still produce a robust evaluation for that position. This is the role of the Tier 2 Quiescence Search.

Instead of relying on a potentially noisy value from a neural network in a sharp position, this search resolves all immediate tactical possibilities to arrive at a stable, "quiet" position to evaluate.

- **Process**: The search expands tactical moves—primarily captures, promotions, and pressing checks—and ignores quiet moves. It continues until no more tactical moves are available.
- **Evaluation**: The final, quiet position is then scored by a very fast evaluation function (e.g., an NNUE network or a handcrafted Piece-Square Table).
- **Purpose**: This process avoids the classic problem of a fixed-depth search mis-evaluating a position in the middle of a capture sequence. The resulting score is a much more reliable measure of the leaf node's true value, which is then backpropagated up the MCTS tree.

## Training Philosophy
Caissawary is designed for high learning efficiency, making it feasible to train without nation-state-level resources.

- **Supervised Pre-training**: The recommended approach is to begin with supervised learning. The ResNet policy and the NNUE evaluation function should be pre-trained on a large corpus of high-quality human games. This bootstraps the engine with a strong foundation of strategic and positional knowledge.

- **Efficient Reinforcement Learning**: During subsequent self-play (RL), the engine's learning is accelerated. The built-in tactical search (Tiers 1 and 2) acts as a powerful "inductive bias," preventing the engine from making simple tactical blunders. This provides a cleaner, more focused training signal to the neural networks, allowing them to learn high-level strategy far more effectively than a "blank slate" MCTS architecture.

## Configuration
The node budgets for the tactical searches and other key parameters are designed to be configurable.

```rust
pub struct CaissawaryConfig {
    pub max_iterations: u32,
    pub time_limit: Duration,
    pub exploration_constant: f64,
    
    // Node budget for the parallel mate search at each node
    pub mate_search_nodes: u32,
    
    // Node budget for the quiescence search at each leaf
    pub quiescence_nodes: u32,
}
```

## Technical Stack
- **Core Logic**: Rust, for its performance, memory safety, and fearless concurrency.
- **Neural Networks**: PyTorch (in Python) for training the ResNet policy and NNUE evaluation networks.
- **Board Representation**: Bitboards, for highly efficient move generation and position manipulation.
- **Inference**: A Rust-based inference engine (like tch-rs or a custom ONNX runtime) to run the neural networks during search.
- **Parallelism**: Rust's rayon library for managing the parallel mate searches.

## Building and Running

### Prerequisites
First, ensure you have the Rust toolchain installed.

```bash
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

For the neural network components, you will also need Python and PyTorch.

```bash
# Install Python dependencies
pip install torch numpy python-chess
```

### Build
Clone the repository and build the optimized release binary:

```bash
git clone https://github.com/aaholmes/caissawary.git
cd caissawary
cargo build --release
```

### Usage
The primary binary is a UCI-compliant engine, suitable for use in any standard chess GUI like Arena, Cute Chess, or BanksiaGUI.

```bash
# Run the engine in UCI mode
./target/release/caissawary uci
```
## Testing and Benchmarking
The project includes a comprehensive suite of tests and benchmarks to validate functionality and performance.

```bash
# Run all unit and integration tests
cargo test

# Run a specific benchmark for mate-finding performance
cargo run --release --bin mate_benchmark
```

## Binary Targets
The crate is organized to produce several distinct binaries for different tasks:

- **caissawary**: The main UCI chess engine.
- **benchmark**: A suite for performance testing, measuring nodes-per-second and puzzle-solving speed.
- **train**: The binary for running the training pipeline, handling both supervised pre-training and reinforcement learning loops.

## References
The architecture of Caissawary is inspired by decades of research in computer chess and artificial intelligence. Key influences include:

- Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Campbell, M. et al. (2002). "Deep Blue"
- The Stockfish Engine and the NNUE architecture.

## License
This project is licensed under the terms of the MIT License. Please see the LICENSE file for details.
