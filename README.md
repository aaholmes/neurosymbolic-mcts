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

#### 1. Safety Gates (Tier 1):
Before any expansion, the engine runs ultra-fast "Safety Gates" to detect immediate win/loss conditions:
- **Checks-Only Mate Search:** A depth-limited DFS that only considers checking moves. It instantly spots forced mate sequences (like Mate-in-2) that standard MCTS might miss due to low visit counts.
- **KOTH Geometric Gate:** A geometric pruning algorithm that detects if a King can reach the center (King of the Hill win) within 3 moves faster than the opponent.

#### 2. Tactical Integration (Tier 2):
If the node is not a terminal state, the engine performs a "Tactical Graft":
- **Quiescence Search (QS):** An integer-based tactical search runs on the CPU to resolve captures and checks using hard-coded piece values (1, 3, 3, 5, 9).
- **Grafting:** The best tactical move found by QS is "grafted" into the MCTS tree immediately.
- **Dynamic Value Extrapolation:** The engine uses the **Symbolic Residual Formula** to price the material won by the CPU:
  $$V_{final} = \tanh\left(\text{arctanh}(V_{parent}) + k \cdot \Delta M\right)$$
  Where $k$ is a **position-specific confidence scalar** predicted by the neural network. This allows the engine to determine if a material advantage is decisive or irrelevant in the current strategic context.

#### 3. Strategic Evaluation (Tier 3):
If no tactical resolution is sufficient, the engine engages the **LogosNet** (if enabled):
- **Dual Value Heads:** The network predicts both a deep strategic logit ($V_{net}$) and a material confidence logit ($K_{net}$).
- **Lazy Evaluation:** The network is queried only when necessary, and its predictions guide the selection of "Quiet" moves via PUCT.

## Tier 2: Tactical Grafting
Instead of treating all new nodes as equal, Caissawary injects tactical knowledge directly into the tree structure. This "Neurosymbolic" approach separates **Logical Truth** from **Contextual Interpretation**.

- **The CPU (Logical Truth):** Runs minimax on captures using raw integers. It is blazing fast and ignores strategic "noise."
- **The Neural Net (Contextual Interpretation):** Predicts $k$, the "price" of material.
- **Symbolic Recombination:** By combining these, the engine "grafts" tactical sequences into the MCTS tree with highly accurate initial values, solving the "cold start" problem for sharp positions.

## Tier 3: LogosNet Architecture (Optional)
The engine supports a **Neurosymbolic** mode using the LogosNet architecture.

- **Architecture:** A 10-block ResNet backbone with a standard Policy head and a **Symbolic Residual Value Head**.
- **Dynamic K:** The network learns how much to trust material imbalance. At initialization ($K_{net} = 0$), $k$ is exactly $0.5$. During training, the network adjusts $k$ to prioritize material or strategic compensation.
- **Inference:** Uses **tch-rs** (LibTorch) for high-performance inference. The forward pass accepts both the board features and the raw material scalar.

### Final Layer Details
The value head splits into two paths to predict the final evaluation:

1.  **Deep Value Logit ($V_{net}$):** Represents the network's intuition about the position's value in logit space.
2.  **Confidence Logit ($K_{net}$):** Represents the network's confidence in the material imbalance.

These are combined using the **Dynamic Symbolic Residual Formula**:

$$k = \frac{\text{Softplus}(K_{net})}{2 \ln 2}$$

$$V_{final} = \tanh(V_{net} + k \cdot \Delta M)$$

Where $\Delta M$ is the material imbalance. This architecture allows the network to learn a residual correction to the material advantage, effectively "pricing" the material in the current strategic context.

> **Note:** Neural network support is optional. Compile with `cargo build --features neural` to enable it. You must have a compatible LibTorch installed or let `tch-rs` download one.

## Training Philosophy
Caissawary is designed for high learning efficiency, making it feasible to train without nation-state-level resources.

- **Supervised Pre-training**: The recommended approach is to begin with supervised learning. The ResNet policy and the fast evaluation function should be pre-trained on a large corpus of high-quality human games. This bootstraps the engine with a strong foundation of strategic and positional knowledge.

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
- **Core Logic**: Rust, for its performance, memory safety, and concurrency.
- **Parallelism**: **Rayon** for data parallelism in the mate search portfolio.
- **Neural Networks**: **PyTorch** (in Python) for training; **tch-rs** (LibTorch) for Rust inference.
- **Board Representation**: Bitboards, for highly efficient move generation and position manipulation.

## Building and Running

### Prerequisites
First, ensure you have the Rust toolchain installed.

```bash
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

For the neural network components (optional), you will also need Python and PyTorch.

```bash
# Install Python dependencies
pip install torch numpy python-chess
```

### Build
Clone the repository and build the optimized release binary:

```bash
git clone https://github.com/aaholmes/caissawary.git
cd caissawary

# Standard Build (Tactical MCTS only)
cargo build --release

# Hybrid Build (With Neural Network support)
# Requires LibTorch. Automatic download may happen.
cargo build --release --features neural
```

### Usage
The primary binary is a UCI-compliant engine, suitable for use in any standard chess GUI like Arena, Cute Chess, or BanksiaGUI.

```bash
# Run the engine in UCI mode
./target/release/kingfisher
```
(Type `uci` to verify connection)

### Self-Play Data Generation
To generate training data for the neural network, use the `self_play` binary. This runs parallel games where the engine plays against itself.

```bash
# Generate 100 games with 800 simulations per move, saving to 'data/'
cargo run --release --bin self_play -- 100 800 data
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
- **self_play**: A high-throughput data generation tool that plays games against itself to create training datasets for the neural network.

## References
The architecture of Caissawary is inspired by decades of research in computer chess and artificial intelligence. Key influences include:

- Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Campbell, M. et al. (2002). "Deep Blue"
- The Stockfish Engine and the NNUE architecture.

## License
This project is licensed under the terms of the MIT License. Please see the LICENSE file for details.
