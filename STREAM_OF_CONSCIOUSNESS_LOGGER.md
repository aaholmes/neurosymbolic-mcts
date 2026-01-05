# Caissawary Stream of Consciousness Logger

The **Stream of Consciousness Logger** is a real-time narration system for Caissawary's search process. It provides a human-readable "thought process" of the engine as it navigates the search tree, explaining why specific moves are prioritized and how different tiers of the hybrid search architecture interact.

## 🎯 Purpose

- **Explainable Search:** Understand the "why" behind the engine's move choices.
- **Debugging Tier Transitions:** Verify that Tier 1 gates and Tier 2 grafts are firing correctly in sharp positions.
- **Educational Tool:** Learn how MCTS and PUCT selection work through real-time examples.
- **Performance Analysis:** Identify where the engine is spending time (e.g., mate search vs. neural inference).

## 📊 Verbosity Levels

The logger supports five hierarchical verbosity levels:

| Level | Description | Key Events Logged |
|-------|-------------|-------------------|
| **Silent (0)** | No output | None |
| **Minimal (1)** | Crucial events only | Tier 1 Gate triggers, Search completion, 100-iteration summaries |
| **Normal (2)** | Standard narration | Tier 2 Grafts, Tier 3 NN values, Tactical move counts, 50-iteration summaries |
| **Verbose (3)** | Detailed trace | Mate search starts, QS PVs, Selection decisions (UCB components), Policy priors |
| **Debug (4)** | Full internal state | Iteration starts, Node entry/exit, Backpropagation, Raw state dumps |

## 🎙️ The `verbose_search` Tool

Caissawary includes a dedicated CLI tool to run searches with the logger enabled.

### Basic Usage
```bash
cargo run --release --bin verbose_search -- "FEN_STRING" [options]
```

### Options
- `--verbosity <level>`: Set output level (`silent`, `minimal`, `normal`, `verbose`, `debug`). Defaults to `normal`.
- `--iterations <n>`: Number of MCTS iterations to run. Defaults to 200.
- `--no-emoji`: Disable emoji icons in the output for better compatibility with some terminals.

### Example: Analyzing a Mate-in-1
```bash
cargo run --release --bin verbose_search -- "4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1" --verbosity verbose
```

**Example Output:**
```
🎙️ Verbose MCTS Search
======================
FEN: 4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1
Verbosity: Verbose
Iterations: 200

--- Search begins ---

   0.0ms 🔍 Starting mate search at depth 5
   0.3ms 🔍 Mate search: MATE FOUND: e8e1 (score: 1000001)
   0.3ms 🚨 TIER 1 GATE: Mate in 5 found (winning) → e8e1

   0.3ms ✅ Search complete!
   Best move: e8e1
   Iterations: 0
   Nodes expanded: 0
   Mates found: 1

--- Search complete ---
🏆 Best Move: e8e1
```

## 🏗 Technical Implementation

### Thread-Safe Design
The logger is designed to be shared across threads using `Arc<SearchLogger>`. It uses internal mutability with `Mutex` and `Atomic` types to ensure that logging from different search threads (if enabled) or asynchronous inference callbacks is safe and consistent.

### Output Sinks
The system supports multiple output destinations:
- **Console:** Direct output to stdout.
- **File:** Log search traces to a specific file for later analysis.
- **Buffer:** Accumulate logs in memory (primarily used for unit testing).
- **Multi:** Broadcast logs to multiple sinks simultaneously.

### Performance Conscious
When logging is disabled or set to a lower verbosity than a specific message, the logger uses short-circuit checks to prevent expensive string formatting or I/O operations, ensuring minimal impact on search speed during standard play.

## 🧪 Testing the Logger
You can run the dedicated test suite to verify logger behavior and verbosity filtering:
```bash
cargo test --test unit_tests search_logger
```
