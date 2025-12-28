# Kingfisher Chess Engine - Technical Architecture

This document provides a detailed technical overview of the Kingfisher Chess Engine architecture, focusing on our novel mate-search-first MCTS approach and neural network integration.

## 🏗 Core Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Kingfisher Chess Engine                 │
├─────────────────────────────────────────────────────────────┤
│  UCI Interface │ Time Management │ Position Analysis        │
├─────────────────────────────────────────────────────────────┤
│           Search Algorithms & Decision Making               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Alpha-Beta     │  │ MCTS w/ Mate    │  │ Neural MCTS  │ │
│  │  + Enhancements │  │ Search First    │  │ Integration  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Evaluation Systems                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Pesto Classic  │  │ Texel Tuning    │  │ Neural Net   │ │
│  │  Evaluation     │  │ Optimization    │  │ Policy/Value │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                 Core Chess Infrastructure                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Bitboard       │  │ Magic Bitboard  │  │ Move Types   │ │
│  │  Representation │  │ Move Generation │  │ & Validation │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Mate-Search-First MCTS Innovation

### Problem Statement
Traditional neural network chess engines face a fundamental challenge: expensive neural network evaluations on positions where exact analysis is possible. This is particularly problematic during training when the neural network is randomly initialized and provides poor guidance.

### Solution: Hybrid Classical-Modern Approach

Our mate-search-first MCTS performs the following at each leaf node:

```rust
fn mcts_leaf_evaluation(board: &Board) -> (f64, Vec<f32>) {
    // 1. First attempt: Classical mate search
    if let Some(mate_result) = mate_search(board, depth=3) {
        return (exact_value(mate_result), exact_policy(mate_result));
    }
    
    // 2. Fallback: Neural network evaluation
    neural_network.evaluate(board)
}
```

### Key Benefits

1. **Exact Tactical Analysis**: Forced wins/losses get perfect evaluation
2. **Reduced NN Dependency**: Fewer expensive neural network calls
3. **Training Acceleration**: Correct play in tactical positions from day 1
4. **Interpretable Results**: Classical analysis provides explainable decisions

### Implementation Details

**Mate Search Integration** (`src/mcts/mod.rs`):
- Depth-limited classical search (typically 3-5 plies)
- Alpha-beta with aggressive pruning for mate detection
- Return values: `(score, best_move, depth_to_mate)`

**MCTS Tree Integration** (`src/mcts/node.rs`):
- Exact results bypass neural network evaluation
- Immediate backpropagation of forced results
- Enhanced move ordering based on tactical threats

## 🤖 Neural Network Architecture

### Network Design

**ResNet-Style Architecture**:
```python
Input: 17×8×8 tensor (piece_types × colors × extra_info × squares)
  ↓
Initial Convolution: 17 → 256 channels
  ↓
8× Residual Blocks (256 channels each)
  ↓
┌─────────────────┐    ┌─────────────────┐
│   Policy Head   │    │   Value Head    │
│   256 → 4672    │    │   256 → 1       │
│ (move encoding) │    │ (position eval) │
└─────────────────┘    └─────────────────┘
```

**Board Representation**:
- 17 channels total:
    - 12 channels: 6 piece types × 2 colors
    - 1 channel: En Passant target square
    - 4 channels: Castling rights (White K/Q, Black k/q) - full planes of 1s
- 8×8 spatial dimensions preserve chess board structure
- Binary encoding: 1.0 if piece/feature present, 0.0 otherwise

**Move Encoding**:
- 4096 possible moves: 64×64 from-to square encoding
- Handles all legal chess moves including promotions
- Probability distribution over legal moves

### Training Pipeline

**Data Collection** (`python/data_collection.py`):
```python
# Quality filtering criteria
min_player_rating = 1800
min_game_length = 20
max_game_length = 100
min_time_control = 180  # 3 minutes
```

**Position Extraction** (`python/training_pipeline.py`):
- Sample every 3rd move from middle game (moves 10-90)
- Filter out check positions (too tactical)
- Require minimum piece count (avoid trivial endgames)
- Engine analysis for training targets

**Training Loop**:
```python
loss = policy_loss(predicted_policy, target_policy) + \
       value_loss(predicted_value, target_value)

# Policy loss: Cross-entropy with actual moves
# Value loss: Mean squared error with game outcomes
```

## 🎯 Search Algorithm Enhancements

### Alpha-Beta Framework (`src/search/alpha_beta.rs`)

**Core Enhancements**:
- Iterative Deepening with aspiration windows
- Transposition table with Zobrist hashing
- Null move pruning with zugzwang detection
- Late move reductions (LMR) for non-critical moves
- Quiescence search with SEE (Static Exchange Evaluation)

**Move Ordering**:
1. Transposition table move (if available)
2. Killer moves (non-capture moves that caused beta cutoffs)
3. History heuristic (moves that historically performed well)
4. MVV-LVA (Most Valuable Victim - Least Valuable Attacker)

### MCTS Implementation (`src/mcts/`)

**Selection Phase**:
```rust
fn uct_select(node: &MctsNode) -> usize {
    let exploration = EXPLORATION_CONSTANT * 
        (parent.visits.ln() / child.visits).sqrt();
    
    child_index = argmax(child.q_value + exploration)
}
```

**Expansion Strategy**:
- Prioritize captures and checks first
- Use neural network policy for move priors
- Expand most promising unexplored moves

**Backpropagation**:
- Update visit counts and value accumulation
- Track variance for confidence estimation
- Support both exact (mate) and estimated values

## ⚖️ Evaluation Systems

### Pesto Evaluation (`src/eval.rs`)

**Tapered Evaluation**:
```rust
fn tapered_eval(mg_score: i32, eg_score: i32, game_phase: i32) -> i32 {
    (mg_score * game_phase + eg_score * (256 - game_phase)) / 256
}
```

**Components**:
- **Material**: Piece values with game phase adjustments
- **Piece-Square Tables**: Position-dependent piece values
- **Pawn Structure**: Passed, isolated, doubled, backward pawns
- **King Safety**: Pawn shield, castling rights, attack evaluation
- **Piece Coordination**: Rook placement, bishop pairs
- **Mobility**: Legal move count weighted by piece type

### Texel Tuning (`src/tuning/`)

**Optimization Process**:
1. Collect positions with known outcomes (win/loss/draw)
2. Minimize error between evaluation and actual results
3. Use gradient descent on evaluation parameters
4. Validate improvements on held-out test set

**Parameter Groups**:
- Piece values (separate for middlegame/endgame)
- Piece-square table adjustments
- Pawn structure bonuses/penalties
- King safety factors
- Mobility weightings

## 📊 Benchmarking & Analysis

### Strength Testing Framework (`src/benchmarks/strength_testing.rs`)

**Engine Variants**:
1. **Alpha-Beta**: Pure classical search baseline
2. **MCTS-Classical**: Standard MCTS with classical evaluation
3. **MCTS-Mate-Priority**: Our mate-search-first innovation
4. **MCTS-Neural**: MCTS with neural network guidance
5. **MCTS-Complete**: Combined mate-search + neural network

**Evaluation Metrics**:
- **Accuracy**: Percentage of positions with "correct" moves
- **Move Quality**: Evaluation improvement after move
- **Speed**: Time to decision per position
- **Elo Estimation**: Performance-based rating calculation

### Elo Calculation (`src/benchmarks/elo_estimation.rs`)

**Rating Formula**:
```rust
estimated_elo = baseline_elo + 
    accuracy_component + 
    quality_component + 
    speed_component

// Where components are:
accuracy_component = (accuracy - 0.5) * 200.0
quality_component = (move_quality - 0.6) * 300.0
speed_component = speed_bonus_or_penalty()
```

**Confidence Estimation**:
- Based on sample size and result consistency
- Higher confidence for more positions tested
- Statistical significance testing for comparisons

## 🔧 Performance Optimizations

### Memory Management
- **Bitboards**: 64-bit integers for efficient position representation
- **Move Generation**: Precomputed magic bitboard tables
- **Transposition Tables**: Hash-based caching of search results
- **Neural Network**: Efficient tensor operations with PyTorch

### Search Optimizations
- **Fail-Soft Alpha-Beta**: Extended alpha-beta bounds
- **Aspiration Windows**: Narrow search windows for faster iteration
- **Adaptive Depth**: Dynamic depth adjustment based on position complexity
- **Parallel Evaluation**: Concurrent neural network batching (future)

### Training Optimizations
- **Data Filtering**: Quality-based position selection
- **Batch Processing**: Efficient GPU utilization
- **Checkpointing**: Resume training from interruptions
- **Validation Monitoring**: Early stopping to prevent overfitting

## 🚀 Future Enhancements

### Immediate Improvements
- **Opening Books**: Integrate standard opening repertoires
- **Endgame Tablebases**: Perfect play in simplified positions
- **Time Management**: Sophisticated time allocation
- **Parallel Search**: Multi-threaded MCTS exploration

### Research Directions
- **Self-Play Training**: Generate training data through engine games
- **Architecture Search**: Automated neural network design optimization
- **Multi-Objective Training**: Balance tactical and positional play
- **Transfer Learning**: Adapt to chess variants and different playing styles

### Advanced Features
- **Explanation System**: Natural language move justification
- **Style Adaptation**: Mimic human player characteristics
- **Progressive Training**: Curriculum learning from simple to complex
- **Adversarial Robustness**: Defend against exploitative play

---

This architecture represents a novel approach to chess engine design, successfully bridging classical chess programming with modern AI techniques while maintaining interpretability and training efficiency.