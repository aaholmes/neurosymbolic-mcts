"""Calibration test for the hybrid CPU NNUE + GPU big-net architecture.

The hybrid proposal pipelines a fast CPU evaluator with a slow GPU evaluator.
It only works if the cheap eval calibrates well to the expensive eval — without
bias that varies by position type.

We run TWO calibration comparisons:

  A. Same-evaluator depth comparison: Stockfish at low depth (fast) vs
     Stockfish at higher depth (slow). This isolates the "fast vs slow
     calibration" question from the "two different models" question. If
     this calibrates well, the architectural pattern is viable. If not,
     even ideal evaluators can't be hybridized this way.

  B. Cross-model comparison: Stockfish (fast eval) vs our Caissawary
     OracleNet (slow eval). This tests the actual proposed pairing, but
     conflates strength gap (Stockfish ~3500 Elo, Caissawary ~2500) with
     architectural calibration. Informative but should be read with caution.

Decision signal:
  - Small uniform residuals → calibration is viable, prototype the hybrid.
  - Small overall but residuals cluster by bucket → scalar calibration fails;
    distill a small net from the big net's value head instead.
  - Large or wildly inconsistent → hybrid architecture is unviable.

Output:
  - Console report with per-bucket RMSE and recommendation.
  - JSON dump of all per-position evals + calibration coefficients to /tmp/calib/results.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn
import torch.nn.functional as F
from model import SEBlock, ResBlock


class V3ValueOnly(nn.Module):
    """Minimal v3-architecture loader for the value head only (the saved
    checkpoints predate the v4 global-scalar-k refactor). Loads only the
    backbone + v_conv/v_bn/v_fc/v_out weights from a v3 checkpoint and
    returns tanh(v_logit) with q_result=0 — i.e. the network's pure
    positional value estimate, ignoring the per-position k head."""

    def __init__(self, num_blocks=6, hidden_dim=128, input_channels=17):
        super().__init__()
        self.start_conv = nn.Conv2d(input_channels, hidden_dim, 3, padding=1, bias=False)
        self.start_bn = nn.BatchNorm2d(hidden_dim)
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.v_conv = nn.Conv2d(hidden_dim, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc = nn.Linear(64, 256)   # v3 used 64; v4 widened to 65
        self.v_out = nn.Linear(256, 1)

    def forward(self, x):
        """Returns raw v_logit (positional residual). Caller composes with
        material delta: value = tanh(v_logit + k * delta_M_pawns)."""
        x = F.relu(self.start_bn(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc(v))
        v_logit = self.v_out(v)
        return v_logit.squeeze(-1)

    @staticmethod
    def load_from_checkpoint(path):
        m = V3ValueOnly()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Filter to keys this submodel cares about
        own_keys = set(m.state_dict().keys())
        filtered = {k: v for k, v in sd.items() if k in own_keys}
        missing, unexpected = m.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  WARN: missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        return m


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material_delta_pawns(board: chess.Board) -> float:
    """STM-perspective material delta in pawn units."""
    delta = 0.0
    for sq, piece in board.piece_map().items():
        v = PIECE_VALUES[piece.piece_type]
        delta += v if piece.color == board.turn else -v
    return delta


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """17x8x8 STM-perspective encoding (matches training_pipeline.ChessPosition.to_tensor)."""
    tensor = np.zeros((17, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    is_white = board.turn == chess.WHITE
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        rank, file = divmod(square, 8)
        tensor_rank = 7 - rank if is_white else rank
        is_us = piece.color == board.turn
        color_offset = 0 if is_us else 6
        tensor[color_offset + piece_map[piece.piece_type], tensor_rank, file] = 1.0
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


def random_game_positions(n_games: int, seed: int = 42):
    """Play n_games light random games, sampling positions across plies."""
    rng = random.Random(seed)
    positions = []
    for game_i in range(n_games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=True):
            legal = list(board.legal_moves)
            if not legal:
                break
            # Light bias toward captures/checks to get tactical positions too
            scored = []
            for mv in legal:
                s = 1.0
                if board.is_capture(mv):
                    s = 4.0
                elif board.gives_check(mv):
                    s = 2.0
                scored.append((mv, s))
            total = sum(s for _, s in scored)
            r = rng.random() * total
            cum = 0.0
            chosen = legal[0]
            for mv, s in scored:
                cum += s
                if r <= cum:
                    chosen = mv
                    break
            board.push(chosen)
            ply = board.ply()
            # Sample positions throughout the game (every other ply)
            if ply % 2 == 0 and ply <= 80:
                positions.append(board.copy())
            if ply > 100:
                break
    return positions


def stockfish_eval(engine: chess.engine.SimpleEngine, board: chess.Board,
                   depth: int = 10) -> int:
    """Stockfish NNUE eval in centipawns from STM perspective."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(board.turn)
    if score.is_mate():
        # Map mate scores to a large cp value
        m = score.mate()
        return 10000 if m > 0 else -10000
    return score.score(mate_score=10000)


def caissawary_eval_batch(model, boards, k_default: float = 0.326):
    """Run V3ValueOnly on a batch, then compose with material delta to get
    Caissawary's full value estimate: value = tanh(v_logit + k * delta_M_pawns).
    Returns (values_in_minus1_to_1, raw_v_logits, deltas_pawns) for diagnostic."""
    tensors = [torch.from_numpy(board_to_tensor(b)) for b in boards]
    x = torch.stack(tensors)
    model.eval()
    with torch.no_grad():
        v_logit = model(x).cpu().numpy()
    deltas = np.array([material_delta_pawns(b) for b in boards])
    composed = np.tanh(v_logit + k_default * deltas)
    return composed, v_logit, deltas


def cp_to_winprob(cp: float) -> float:
    """Convert centipawn score to win probability (Lichess-style sigmoid)."""
    return 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))


def caissa_to_winprob(v: float) -> float:
    """Caissawary tanh value [-1,1] → win prob [0,1]."""
    return 0.5 * (v + 1.0)


def fit_sigmoid_calibration(cps: np.ndarray, targets: np.ndarray):
    """Fit P_target ≈ sigmoid(a * cp + b) via least squares on the logit.
    Returns (a, b, rmse)."""
    eps = 1e-4
    targets_clipped = np.clip(targets, eps, 1.0 - eps)
    logit = np.log(targets_clipped / (1.0 - targets_clipped))  # logit(target)
    # Linear regression: logit = a*cp + b
    A = np.stack([cps, np.ones_like(cps)], axis=1)
    sol, *_ = np.linalg.lstsq(A, logit, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    pred = 1.0 / (1.0 + np.exp(-(a * cps + b)))
    rmse = float(np.sqrt(np.mean((pred - targets) ** 2)))
    return a, b, rmse, pred


def bucket_stats(name, mask, cps, targets, preds):
    if not mask.any():
        return f"  {name:24s} (n=0)"
    n = int(mask.sum())
    res = preds[mask] - targets[mask]
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mean_bias = float(np.mean(res))
    return f"  {name:24s} n={n:4d}  RMSE={rmse:.3f}  bias={mean_bias:+.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default="/tmp/calib/stockfish/stockfish-ubuntu-x86-64-avx2")
    parser.add_argument("--checkpoint",
        default="runs/long_run/scaleup_2m_tiered_propgreedy/weights/gen_46.pth")
    parser.add_argument("--n-games", type=int, default=80)
    parser.add_argument("--fast-depth", type=int, default=2,  help="Fast SF depth (proxy for NNUE-only)")
    parser.add_argument("--slow-depth", type=int, default=12, help="Slow SF depth (proxy for accurate eval)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/calib/results.json")
    args = parser.parse_args()

    print(f"=== Calibration test ===")
    print(f"  stockfish:  {args.stockfish}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  n_games:    {args.n_games}")
    print(f"  fast depth: {args.fast_depth}   slow depth: {args.slow_depth}")

    # 1. Generate positions
    t0 = time.time()
    positions = random_game_positions(args.n_games, seed=args.seed)
    print(f"  generated {len(positions)} positions in {time.time()-t0:.1f}s")

    # 2. Stockfish evals at fast and slow depth
    t0 = time.time()
    sf_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf_engine.configure({"Threads": 1, "Hash": 64})
    cps_fast, cps_slow = [], []
    for i, b in enumerate(positions):
        cps_fast.append(stockfish_eval(sf_engine, b, depth=args.fast_depth))
        cps_slow.append(stockfish_eval(sf_engine, b, depth=args.slow_depth))
        if (i + 1) % 200 == 0:
            print(f"  stockfish: {i+1}/{len(positions)} ({(time.time()-t0):.1f}s)")
    sf_engine.quit()
    cps_fast = np.array(cps_fast, dtype=np.float64)
    cps_slow = np.array(cps_slow, dtype=np.float64)
    print(f"  stockfish done in {time.time()-t0:.1f}s")

    # 3. Caissawary evals
    t0 = time.time()
    model = V3ValueOnly.load_from_checkpoint(args.checkpoint)
    model.eval()
    BATCH = 64
    caissa_values, v_logits, deltas = [], [], []
    for i in range(0, len(positions), BATCH):
        batch = positions[i:i+BATCH]
        vals, vl, dm = caissawary_eval_batch(model, batch)
        caissa_values.extend(vals.tolist())
        v_logits.extend(vl.tolist())
        deltas.extend(dm.tolist())
    caissa_values = np.array(caissa_values, dtype=np.float64)
    v_logits = np.array(v_logits, dtype=np.float64)
    deltas = np.array(deltas, dtype=np.float64)
    print(f"  caissawary done in {time.time()-t0:.1f}s")
    print(f"  diagnostic: v_logit range [{v_logits.min():.2f}, {v_logits.max():.2f}],"
          f" |material| mean {np.mean(np.abs(deltas)):.2f} pawns")

    # === Test A: SF-fast → SF-slow (same evaluator, fast vs slow) ===
    targets_A = np.array([cp_to_winprob(c) for c in cps_slow])
    a_A, b_A, rmse_A, preds_A = fit_sigmoid_calibration(cps_fast, targets_A)
    plies = np.array([p.ply() for p in positions])
    abs_fast = np.abs(cps_fast)

    print(f"\n=== Test A: Stockfish d{args.fast_depth} (fast) → Stockfish d{args.slow_depth} (slow) ===")
    print(f"  Fit:   P_slow ≈ sigmoid({a_A:.5f} * cp_fast + {b_A:.4f})")
    print(f"  Overall RMSE: {rmse_A:.3f} (probability units)")
    print(f"  Residuals by ply:")
    print(bucket_stats("opening (ply ≤ 20)",       plies <= 20,                   cps_fast, targets_A, preds_A))
    print(bucket_stats("middlegame (20 < p ≤ 50)", (plies > 20) & (plies <= 50),  cps_fast, targets_A, preds_A))
    print(bucket_stats("endgame (ply > 50)",       plies > 50,                    cps_fast, targets_A, preds_A))
    print(f"  Residuals by |cp_fast|:")
    print(bucket_stats("|cp| ≤ 50",         abs_fast <= 50,                            cps_fast, targets_A, preds_A))
    print(bucket_stats("50 < |cp| ≤ 200",   (abs_fast > 50)  & (abs_fast <= 200),     cps_fast, targets_A, preds_A))
    print(bucket_stats("200 < |cp| ≤ 1000", (abs_fast > 200) & (abs_fast <= 1000),    cps_fast, targets_A, preds_A))
    print(bucket_stats("|cp| > 1000",       abs_fast > 1000,                           cps_fast, targets_A, preds_A))
    bucket_rmses_A = [float(np.sqrt(np.mean((preds_A[m] - targets_A[m]) ** 2)))
                      for m in [plies <= 20, (plies > 20) & (plies <= 50), plies > 50] if m.any()]
    spread_A = max(bucket_rmses_A) - min(bucket_rmses_A) if bucket_rmses_A else 0.0
    print(f"  Cross-bucket RMSE spread: {spread_A:.3f}")

    # === Test B: SF-slow → Caissawary (cross-model) ===
    targets_B = np.array([caissa_to_winprob(v) for v in caissa_values])
    a_B, b_B, rmse_B, preds_B = fit_sigmoid_calibration(cps_slow, targets_B)
    abs_slow = np.abs(cps_slow)

    print(f"\n=== Test B: Stockfish d{args.slow_depth} (cp) → Caissawary OracleNet (winprob) ===")
    print(f"  (Caveat: Stockfish 17 ~3500 Elo, Caissawary v3 ~2500 Elo. Strength gap dominates.)")
    print(f"  Fit:   P_caissa ≈ sigmoid({a_B:.5f} * cp + {b_B:.4f})")
    print(f"  Overall RMSE: {rmse_B:.3f}")
    print(bucket_stats("opening (ply ≤ 20)",       plies <= 20,                  cps_slow, targets_B, preds_B))
    print(bucket_stats("middlegame (20 < p ≤ 50)", (plies > 20) & (plies <= 50), cps_slow, targets_B, preds_B))
    print(bucket_stats("endgame (ply > 50)",       plies > 50,                   cps_slow, targets_B, preds_B))

    # === Verdict ===
    # Per-magnitude breakdown for Test A: where in the eval range is the noise concentrated?
    drawish_mask = abs_fast <= 50
    midrange_mask = (abs_fast > 50) & (abs_fast <= 1000)
    rmse_drawish  = float(np.sqrt(np.mean((preds_A[drawish_mask] - targets_A[drawish_mask]) ** 2))) if drawish_mask.any() else 0
    rmse_midrange = float(np.sqrt(np.mean((preds_A[midrange_mask] - targets_A[midrange_mask]) ** 2))) if midrange_mask.any() else 0

    print(f"\n=== Verdict ===")
    print(f"  Same-evaluator (Test A) RMSE: overall {rmse_A:.3f},  drawish {rmse_drawish:.3f},  midrange {rmse_midrange:.3f}")
    print(f"  Cross-bucket spread (by ply): {spread_A:.3f}")
    print(f"")
    if rmse_A < 0.10 and spread_A < 0.05 and rmse_midrange < 0.13:
        verdict = ("GREEN: same-evaluator fast→slow calibrates tightly. The hybrid pattern is viable — "
                   "distillation may not even be necessary. Build the prototype.")
    elif rmse_A < 0.20 and spread_A < 0.08:
        verdict = ("YELLOW: calibration works at the extremes (drawish and clearly winning) but is noisy "
                   "in the tactical mid-range where guidance matters most. The hybrid pattern is *plausible* "
                   "but off-the-shelf NNUE will mislead exploration in tactical positions. The distilled-net "
                   "variant (small net trained from the big net's value head) is likely required to reach "
                   "useful strength.")
    else:
        verdict = ("RED: calibration is too noisy even for same-evaluator fast→slow. Provisional values "
                   "would corrupt MCTS statistics enough to defeat the speculation gain.")
    print(f"  {verdict}")
    print(f"\n  Test B (cross-model) RMSE {rmse_B:.3f} — dominated by Stockfish↔Caissawary strength gap, not architectural.")

    # Persist results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "n_positions":  len(positions),
        "fast_depth": args.fast_depth,
        "slow_depth": args.slow_depth,
        "test_A_same_eval": {
            "calibration_a": a_A, "calibration_b": b_A, "overall_rmse": rmse_A,
            "bucket_rmses": {
                "opening":    bucket_rmses_A[0] if len(bucket_rmses_A) > 0 else None,
                "middlegame": bucket_rmses_A[1] if len(bucket_rmses_A) > 1 else None,
                "endgame":    bucket_rmses_A[2] if len(bucket_rmses_A) > 2 else None,
            },
        },
        "test_B_cross_model": {
            "calibration_a": a_B, "calibration_b": b_B, "overall_rmse": rmse_B,
        },
        "verdict": verdict,
        "raw": [
            {"fen": positions[i].fen(), "ply": int(plies[i]),
             "sf_cp_fast": float(cps_fast[i]), "sf_cp_slow": float(cps_slow[i]),
             "caissa_value": float(caissa_values[i])}
            for i in range(len(positions))
        ],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  results saved to {args.output}")


if __name__ == "__main__":
    main()
