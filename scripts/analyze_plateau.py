#!/usr/bin/env python3
"""Analyze the training plateau: does the NN learn beyond classical fallback?

Compares gen 0 (zero-initialized) vs gen 19 (best tiered) on diverse positions.
Measures: v_logit distributions, policy entropy, k-head adaptation.

Usage:
    python scripts/analyze_plateau.py [--tiered-dir DIR] [--device DEVICE]
"""

import argparse
import json
import math
import sys
import os

import chess
import numpy as np
import torch
import torch.nn.functional as F

# Add python/ to path for model import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from model import SeresNet


# --- Diverse test positions ---
# Mix of openings, middlegames, endgames, tactical positions
TEST_POSITIONS = [
    # Openings
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "opening", "1.e4"),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "opening", "1.d4"),
    ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "opening", "French"),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "opening", "Sicilian"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "opening", "Open game"),
    ("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "opening", "Alekhine"),
    ("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "opening", "Caro-Kann"),
    ("rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", "opening", "Modern"),

    # Middlegame — balanced
    ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 7", "middlegame", "Italian balanced"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "middlegame", "Italian early"),
    ("r1b1kb1r/ppq2ppp/2n1pn2/3p4/3P4/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 0 7", "middlegame", "QGD middlegame"),
    ("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQ - 0 6", "middlegame", "d4 middlegame"),
    ("r1bq1rk1/pppn1ppp/4pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7", "middlegame", "Nimzo-like"),
    ("r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 10", "middlegame", "Dragon-like"),
    ("r1bqr1k1/pp3pbp/2np1np1/2p1p3/4P3/2NP1NP1/PPP1QPBP/R1B2RK1 w - - 0 10", "middlegame", "KID-like"),
    ("r2qr1k1/1b1nbppp/pp1ppn2/8/2PNP3/1PN1B3/P4PPP/R2QKB1R w KQ - 0 12", "middlegame", "Hedgehog"),

    # Middlegame — tactical
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "tactical", "Italian with tension"),
    ("rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 4", "tactical", "Slav with d5 tension"),
    ("r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6", "tactical", "Giuoco Piano"),
    ("r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R b KQkq - 0 5", "tactical", "Ruy with Bg5"),
    ("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w kq - 0 7", "tactical", "Ruy closed"),
    ("r1b1kb1r/1pqn1ppp/p2ppn2/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 0 8", "tactical", "Najdorf-like"),
    ("r2q1rk1/ppp1bppp/2n2n2/3pp1B1/2PP4/2N2N2/PP2PPPP/R2QKB1R w KQ - 0 7", "tactical", "Central tension"),
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/4P3/2N2NP1/PPPP1PBP/R1BQK2R w KQ - 0 6", "tactical", "English+KID"),

    # Endgame — rook endgames
    ("8/5pk1/6p1/8/8/6P1/5PK1/4R3 w - - 0 1", "endgame", "R+P vs P"),
    ("8/8/4kpp1/8/8/4KPP1/4R3/4r3 w - - 0 1", "endgame", "R+2P vs R+2P"),
    ("5k2/5p2/4p3/8/3R4/8/5PPP/6K1 w - - 0 1", "endgame", "R vs pawns"),
    ("4r1k1/5pp1/8/8/8/8/5PPP/4R1K1 w - - 0 1", "endgame", "R vs R equal"),

    # Endgame — minor piece
    ("8/5pk1/6p1/8/2B5/6P1/5PK1/8 w - - 0 1", "endgame", "B+P vs P"),
    ("8/5pk1/4b1p1/8/8/4B1P1/5PK1/8 w - - 0 1", "endgame", "B vs B same color"),
    ("8/5pk1/4b1p1/8/8/4N1P1/5PK1/8 w - - 0 1", "endgame", "N vs B"),
    ("8/2k5/8/8/3NK3/8/8/8 w - - 0 1", "endgame", "K+N vs K"),

    # Endgame — pawn endgames
    ("8/5pk1/6p1/8/8/6PP/5PK1/8 w - - 0 1", "endgame", "K+3P vs K+2P"),
    ("8/8/4k3/8/4PK2/8/8/8 w - - 0 1", "endgame", "K+P vs K opposition"),
    ("8/p7/1p6/8/1P6/P7/8/4K2k w - - 0 1", "endgame", "Outside passer"),

    # KOTH-relevant (king near center)
    ("rnbq1bnr/pppp1ppp/8/4k3/4P3/8/PPPP1PPP/RNBQKBNR w KQ - 0 3", "koth", "King on e5"),
    ("rnbq1bnr/ppppkppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQ - 0 3", "koth", "King on e7 approach"),
    ("r1bq1bnr/pppp1ppp/2n5/4k3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQ - 0 4", "koth", "King exposed center"),
    ("rnbq1bnr/pppp1ppp/8/8/3kP3/8/PPPP1PPP/RNBQKBNR w KQ - 0 3", "koth", "King on d4"),

    # Imbalanced material
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "imbalanced", "Equal material"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 4", "imbalanced", "Near equal after O-O"),
    ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2", "imbalanced", "Weak pawn structure"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "opening", "Starting position"),

    # More diverse middlegame positions
    ("r2q1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 10", "middlegame", "Yugoslav attack"),
    ("r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9", "middlegame", "Closed Ruy"),
    ("rn1qk2r/pb1pbppp/1p2pn2/2p5/2PP4/5NP1/PP2PPBP/RNBQK2R w KQkq - 0 6", "middlegame", "Catalan-like"),
]


def fen_to_tensor(fen: str) -> np.ndarray:
    """Convert FEN to 17x8x8 tensor in STM perspective."""
    board = chess.Board(fen)
    tensor = np.zeros((17, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    is_white = board.turn == chess.WHITE

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            tensor_rank = 7 - rank if is_white else rank
            is_us = piece.color == board.turn
            color_offset = 0 if is_us else 6
            channel = color_offset + piece_map[piece.piece_type]
            tensor[channel, tensor_rank, file] = 1.0

    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor_rank = 7 - rank if is_white else rank
        tensor[12, tensor_rank, file] = 1.0

    if is_white:
        rights = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]
    else:
        rights = [
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
        ]

    for i, allowed in enumerate(rights):
        if allowed:
            tensor[13 + i, :, :] = 1.0

    return tensor


def policy_entropy(log_probs: np.ndarray) -> float:
    """Compute entropy of a log-probability distribution."""
    probs = np.exp(log_probs)
    # Filter out zero probabilities
    mask = probs > 1e-10
    return -np.sum(probs[mask] * log_probs[mask])


def load_model(checkpoint_path, num_blocks=2, hidden_dim=64, device="cpu"):
    """Load an SeresNet checkpoint."""
    model = SeresNet(num_blocks=num_blocks, hidden_dim=hidden_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, positions, device="cpu"):
    """Run inference on a list of (fen, category, label) positions.

    Returns dict with per-position results.
    """
    results = []
    tensors = []
    for fen, category, label in positions:
        tensors.append(fen_to_tensor(fen))

    board_batch = torch.from_numpy(np.stack(tensors)).float().to(device)
    material_batch = torch.zeros(len(tensors), 1, device=device)

    with torch.no_grad():
        policies, v_logits, ks = model(board_batch, material_batch)

    for i, (fen, category, label) in enumerate(positions):
        log_probs = policies[i].cpu().numpy()
        v = v_logits[i, 0].item()
        k = ks[i, 0].item()
        ent = policy_entropy(log_probs)

        # Top-5 moves by probability
        probs = np.exp(log_probs)
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_probs = probs[top5_idx]

        results.append({
            "fen": fen,
            "category": category,
            "label": label,
            "v_logit": v,
            "k": k,
            "entropy": ent,
            "top5_probs": top5_probs.tolist(),
            "top1_prob": float(top5_probs[0]),
        })

    return results


def print_comparison(gen0_results, gen19_results):
    """Print detailed comparison between gen0 and gen19."""

    print("=" * 80)
    print("TRAINING PLATEAU ANALYSIS: Gen 0 vs Gen 19")
    print("=" * 80)

    # --- 1. Overall v_logit statistics ---
    gen0_vlogits = [r["v_logit"] for r in gen0_results]
    gen19_vlogits = [r["v_logit"] for r in gen19_results]

    print("\n--- V_logit Distribution ---")
    print(f"{'':>20} {'Gen 0':>12} {'Gen 19':>12} {'Delta':>12}")
    print(f"{'Mean':>20} {np.mean(gen0_vlogits):>+12.4f} {np.mean(gen19_vlogits):>+12.4f} {np.mean(gen19_vlogits)-np.mean(gen0_vlogits):>+12.4f}")
    print(f"{'Std':>20} {np.std(gen0_vlogits):>12.4f} {np.std(gen19_vlogits):>12.4f} {np.std(gen19_vlogits)-np.std(gen0_vlogits):>+12.4f}")
    print(f"{'Min':>20} {np.min(gen0_vlogits):>+12.4f} {np.min(gen19_vlogits):>+12.4f}")
    print(f"{'Max':>20} {np.max(gen0_vlogits):>+12.4f} {np.max(gen19_vlogits):>+12.4f}")
    print(f"{'|Mean|':>20} {np.mean(np.abs(gen0_vlogits)):>12.4f} {np.mean(np.abs(gen19_vlogits)):>12.4f}")

    # --- 2. v_logit by category ---
    categories = sorted(set(r["category"] for r in gen0_results))
    print("\n--- V_logit by Position Type ---")
    print(f"{'Category':>15} {'Gen0 mean':>12} {'Gen19 mean':>12} {'Gen0 std':>10} {'Gen19 std':>10}")
    for cat in categories:
        g0 = [r["v_logit"] for r in gen0_results if r["category"] == cat]
        g19 = [r["v_logit"] for r in gen19_results if r["category"] == cat]
        print(f"{cat:>15} {np.mean(g0):>+12.4f} {np.mean(g19):>+12.4f} {np.std(g0):>10.4f} {np.std(g19):>10.4f}")

    # --- 3. Policy entropy ---
    gen0_ent = [r["entropy"] for r in gen0_results]
    gen19_ent = [r["entropy"] for r in gen19_results]

    print("\n--- Policy Entropy (lower = more focused) ---")
    print(f"{'':>20} {'Gen 0':>12} {'Gen 19':>12} {'Delta':>12}")
    print(f"{'Mean':>20} {np.mean(gen0_ent):>12.2f} {np.mean(gen19_ent):>12.2f} {np.mean(gen19_ent)-np.mean(gen0_ent):>+12.2f}")
    print(f"{'Std':>20} {np.std(gen0_ent):>12.2f} {np.std(gen19_ent):>12.2f}")
    # Uniform over ~30 legal moves: entropy ≈ ln(30) ≈ 3.4
    print(f"{'Uniform(30 moves)':>20} {'~3.40':>12}")

    print("\n--- Policy Entropy by Position Type ---")
    print(f"{'Category':>15} {'Gen0 ent':>12} {'Gen19 ent':>12} {'Delta':>12} {'%change':>10}")
    for cat in categories:
        g0 = [r["entropy"] for r in gen0_results if r["category"] == cat]
        g19 = [r["entropy"] for r in gen19_results if r["category"] == cat]
        delta = np.mean(g19) - np.mean(g0)
        pct = 100 * delta / np.mean(g0) if np.mean(g0) > 0 else 0
        print(f"{cat:>15} {np.mean(g0):>12.2f} {np.mean(g19):>12.2f} {delta:>+12.2f} {pct:>+9.1f}%")

    # --- 4. k-head adaptation ---
    gen0_k = [r["k"] for r in gen0_results]
    gen19_k = [r["k"] for r in gen19_results]

    print("\n--- K-head (material confidence) ---")
    print(f"{'':>20} {'Gen 0':>12} {'Gen 19':>12} {'Delta':>12}")
    print(f"{'Mean':>20} {np.mean(gen0_k):>12.4f} {np.mean(gen19_k):>12.4f} {np.mean(gen19_k)-np.mean(gen0_k):>+12.4f}")
    print(f"{'Std':>20} {np.std(gen0_k):>12.4f} {np.std(gen19_k):>12.4f}")
    print(f"{'Init value':>20} {'0.5000':>12}")

    print("\n--- K by Position Type ---")
    print(f"{'Category':>15} {'Gen0 k':>12} {'Gen19 k':>12} {'Gen0 std':>10} {'Gen19 std':>10}")
    for cat in categories:
        g0 = [r["k"] for r in gen0_results if r["category"] == cat]
        g19 = [r["k"] for r in gen19_results if r["category"] == cat]
        print(f"{cat:>15} {np.mean(g0):>12.4f} {np.mean(g19):>12.4f} {np.std(g0):>10.4f} {np.std(g19):>10.4f}")

    # --- 5. Top-1 probability (move confidence) ---
    gen0_top1 = [r["top1_prob"] for r in gen0_results]
    gen19_top1 = [r["top1_prob"] for r in gen19_results]

    print("\n--- Top-1 Move Probability (higher = more confident) ---")
    print(f"{'':>20} {'Gen 0':>12} {'Gen 19':>12} {'Delta':>12}")
    print(f"{'Mean':>20} {np.mean(gen0_top1):>12.4f} {np.mean(gen19_top1):>12.4f} {np.mean(gen19_top1)-np.mean(gen0_top1):>+12.4f}")
    print(f"{'Uniform(30 moves)':>20} {'~0.0333':>12}")

    # --- 6. Per-position comparison ---
    print("\n--- Per-Position Detail (sorted by |v_logit delta|) ---")
    deltas = []
    for g0, g19 in zip(gen0_results, gen19_results):
        d = abs(g19["v_logit"] - g0["v_logit"])
        deltas.append((d, g0, g19))
    deltas.sort(key=lambda x: -x[0])

    print(f"{'Label':>25} {'Cat':>12} {'G0 v':>8} {'G19 v':>8} {'Delta':>8} {'G0 k':>7} {'G19 k':>7} {'G0 ent':>7} {'G19 ent':>7}")
    for d, g0, g19 in deltas[:20]:
        print(f"{g0['label']:>25} {g0['category']:>12} {g0['v_logit']:>+8.3f} {g19['v_logit']:>+8.3f} {g19['v_logit']-g0['v_logit']:>+8.3f} {g0['k']:>7.3f} {g19['k']:>7.3f} {g0['entropy']:>7.2f} {g19['entropy']:>7.2f}")

    # --- 7. Summary diagnosis ---
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Gen 0 is zero-initialized: v_logit=0 everywhere, k=0.5 everywhere.
    # Compare gen19 absolute values rather than ratios.
    gen19_v_std = np.std(gen19_vlogits)
    gen19_v_absmax = max(abs(np.min(gen19_vlogits)), abs(np.max(gen19_vlogits)))
    ent_ratio = np.mean(gen19_ent) / np.mean(gen0_ent)
    gen19_k_std = np.std(gen19_k)
    gen19_k_mean = np.mean(gen19_k)

    print(f"\nV_logit spread (gen19 std): {gen19_v_std:.4f}, max |v_logit|: {gen19_v_absmax:.4f}")
    if gen19_v_absmax > 0.5:
        print("  -> NN has learned strong positional knowledge (v_logit > 0.5)")
    elif gen19_v_absmax > 0.1:
        print("  -> NN has learned modest positional knowledge (v_logit ~0.1-0.5)")
        print("     At k~0.05, material dominates: a 1-pawn edge (deltaM=1) contributes")
        print(f"     k*deltaM = {gen19_k_mean:.3f} vs typical |v_logit| = {np.mean(np.abs(gen19_vlogits)):.3f}")
    else:
        print("  -> NN v_logit is negligible (< 0.1) — near-zero positional knowledge")

    print(f"\nPolicy entropy ratio (gen19/gen0): {ent_ratio:.3f}")
    if ent_ratio < 0.5:
        print("  -> Policy is drastically more focused (>50% entropy reduction)")
        print("     This is the primary learned improvement — better move selection")
    elif ent_ratio < 0.7:
        print("  -> Policy is significantly more focused (>30% entropy reduction)")
    elif ent_ratio < 0.9:
        print("  -> Policy has improved moderately (10-30% entropy reduction)")
    else:
        print("  -> Policy is barely more focused than uniform")

    print(f"\nK-head mean: {gen19_k_mean:.4f} (init: 0.5), std: {gen19_k_std:.4f}")
    if gen19_k_mean < 0.2:
        print(f"  -> K dropped dramatically from 0.5 to {gen19_k_mean:.3f}")
        print("     NN learned to downweight material in favor of positional evaluation")
        if gen19_k_std > 0.01:
            print(f"     K varies by position type (std={gen19_k_std:.4f}): adapting to context")
    elif abs(gen19_k_mean - 0.5) < 0.1:
        print("  -> K is near initialization (0.5) — not adapting")

    # Overall assessment
    learned_value = gen19_v_absmax > 0.05
    learned_policy = ent_ratio < 0.85
    learned_k = abs(gen19_k_mean - 0.5) > 0.1

    print(f"\nOverall: NN learned {'value' if learned_value else 'NO value'} / "
          f"{'policy' if learned_policy else 'NO policy'} / "
          f"{'k-adaptation' if learned_k else 'NO k-adaptation'}")

    if learned_policy:
        print("\nKey finding: The NN's primary contribution is POLICY, not value.")
        print(f"  Policy entropy dropped {(1-ent_ratio)*100:.0f}% — the NN selects much better moves.")
        if learned_value and gen19_v_absmax < 0.5:
            print(f"  V_logit is modest (|max|={gen19_v_absmax:.3f}) — positional eval is secondary.")
            print(f"  With k={gen19_k_mean:.3f}, even a 1-pawn material edge ({gen19_k_mean:.3f}) is")
            print(f"  comparable to the largest v_logit ({gen19_v_absmax:.3f}).")
        print("\n  The Elo plateau likely reflects:")
        print("  - Policy saturating at 240K param capacity")
        print("  - Value head contributing little beyond material (k≈0.05)")
        print("  - Larger model should show value head improvement")


def try_plot(gen0_results, gen19_results, output_path="plateau_analysis.png"):
    """Generate comparison plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. V_logit histogram
    ax = axes[0, 0]
    gen0_v = [r["v_logit"] for r in gen0_results]
    gen19_v = [r["v_logit"] for r in gen19_results]
    bins = np.linspace(min(min(gen0_v), min(gen19_v)) - 0.1,
                       max(max(gen0_v), max(gen19_v)) + 0.1, 30)
    ax.hist(gen0_v, bins=bins, alpha=0.5, label="Gen 0", color="#2563eb")
    ax.hist(gen19_v, bins=bins, alpha=0.5, label="Gen 19", color="#dc2626")
    ax.set_xlabel("v_logit")
    ax.set_ylabel("Count")
    ax.set_title("V_logit Distribution")
    ax.legend()
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # 2. Policy entropy by category
    ax = axes[0, 1]
    categories = sorted(set(r["category"] for r in gen0_results))
    x = np.arange(len(categories))
    width = 0.35
    gen0_ent_by_cat = [np.mean([r["entropy"] for r in gen0_results if r["category"] == c]) for c in categories]
    gen19_ent_by_cat = [np.mean([r["entropy"] for r in gen19_results if r["category"] == c]) for c in categories]
    ax.bar(x - width/2, gen0_ent_by_cat, width, label="Gen 0", color="#2563eb", alpha=0.7)
    ax.bar(x + width/2, gen19_ent_by_cat, width, label="Gen 19", color="#dc2626", alpha=0.7)
    ax.set_xlabel("Position Type")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy by Position Type")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    # 3. K values by category
    ax = axes[1, 0]
    gen0_k_by_cat = [np.mean([r["k"] for r in gen0_results if r["category"] == c]) for c in categories]
    gen19_k_by_cat = [np.mean([r["k"] for r in gen19_results if r["category"] == c]) for c in categories]
    ax.bar(x - width/2, gen0_k_by_cat, width, label="Gen 0", color="#2563eb", alpha=0.7)
    ax.bar(x + width/2, gen19_k_by_cat, width, label="Gen 19", color="#dc2626", alpha=0.7)
    ax.set_xlabel("Position Type")
    ax.set_ylabel("k (material confidence)")
    ax.set_title("K-head by Position Type")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Init (0.5)")
    ax.legend()

    # 4. V_logit scatter: gen0 vs gen19
    ax = axes[1, 1]
    ax.scatter(gen0_v, gen19_v, alpha=0.6, s=30)
    lims = [min(min(gen0_v), min(gen19_v)) - 0.1, max(max(gen0_v), max(gen19_v)) + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Gen 0 v_logit")
    ax.set_ylabel("Gen 19 v_logit")
    ax.set_title("V_logit: Gen 0 vs Gen 19")
    ax.legend()
    ax.set_aspect("equal")

    plt.suptitle("Training Plateau Analysis: Gen 0 vs Gen 19", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze training plateau")
    parser.add_argument("--tiered-dir", default="runs/long_run/caissawary10",
                        help="Path to tiered training run directory")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (cpu or cuda)")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Number of residual blocks in model")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension of model")
    parser.add_argument("--output", default="plateau_analysis.png",
                        help="Output plot path")
    args = parser.parse_args()

    weights_dir = os.path.join(args.tiered_dir, "weights")

    gen0_path = os.path.join(weights_dir, "gen_0.pth")
    # Find the latest generation
    gen_files = sorted(
        [f for f in os.listdir(weights_dir) if f.startswith("gen_") and f.endswith(".pth")],
        key=lambda f: int(f.split("_")[1].split(".")[0])
    )
    latest_gen_path = os.path.join(weights_dir, gen_files[-1])
    latest_gen_num = int(gen_files[-1].split("_")[1].split(".")[0])

    print(f"Loading gen 0 from: {gen0_path}")
    print(f"Loading gen {latest_gen_num} from: {latest_gen_path}")
    print(f"Model: {args.num_blocks} blocks, {args.hidden_dim} channels")
    print(f"Device: {args.device}")
    print(f"Test positions: {len(TEST_POSITIONS)}")

    gen0_model = load_model(gen0_path, args.num_blocks, args.hidden_dim, args.device)
    gen19_model = load_model(latest_gen_path, args.num_blocks, args.hidden_dim, args.device)

    print("\nRunning inference on gen 0...")
    gen0_results = run_inference(gen0_model, TEST_POSITIONS, args.device)
    print("Running inference on gen 19...")
    gen19_results = run_inference(gen19_model, TEST_POSITIONS, args.device)

    print_comparison(gen0_results, gen19_results)
    try_plot(gen0_results, gen19_results, args.output)


if __name__ == "__main__":
    main()
