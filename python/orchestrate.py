"""AGZ-Style Training Orchestrator.

Config-driven loop: self-play -> buffer -> train -> evaluate -> accept/reject.
"""

import os
import sys
import json
import math
import shutil
import subprocess
import time
import argparse
from dataclasses import dataclass, field, asdict

import torch
from model import OracleNet
from replay_buffer import ReplayBuffer


@dataclass
class TrainingConfig:
    # Self-play
    games_per_generation: int = 100
    simulations_per_move: int = 800
    sims_schedule: str = ""  # e.g. "0:200,10:400,20:800" — gen:sims pairs
    enable_koth: bool = False

    # Ablation flags
    enable_tier1: bool = True
    enable_material_value: bool = True
    log_games: str = "first"

    # Replay buffer (sliding window — old data evicted FIFO)
    buffer_capacity: int = 100_000
    buffer_dir: str = "data/buffer"

    # Training
    minibatches_per_generation: int = 1000
    max_epochs: int = 10
    batch_size: int = 64
    optimizer: str = "muon"
    lr_schedule: str = ""
    initial_lr: float = 0.02

    # Evaluation (SPRT)
    eval_max_games: int = 400
    eval_simulations: int = 800
    eval_top_p_base: float = 0.95
    sprt_elo0: float = 0.0
    sprt_elo1: float = 10.0
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05

    # Parallelism
    inference_batch_size: int = 16
    game_threads: int = 0  # 0 = auto (RAYON_NUM_THREADS or rayon default)

    # Training variants
    single_variant: bool = True  # train "all" only; use --multi-variant for policy/value/all

    # Model architecture
    num_blocks: int = 6
    hidden_dim: int = 128

    # Infrastructure
    weights_dir: str = "weights"
    data_dir: str = "data"
    log_file: str = "training_log.jsonl"
    resume: bool = True
    max_generations: int = 0  # 0 = unlimited
    skip_self_play: bool = False  # skip self-play after gen 1 (eval games produce training data)

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="AGZ Training Orchestrator")
        parser.add_argument("--games-per-generation", type=int, default=100)
        parser.add_argument("--simulations-per-move", type=int, default=800)
        parser.add_argument("--sims-schedule", type=str, default="",
                            help="Ramp sims over generations, e.g. '0:200,10:400,20:800'")
        parser.add_argument("--enable-koth", action="store_true")
        parser.add_argument("--buffer-capacity", type=int, default=100_000)
        parser.add_argument("--buffer-dir", type=str, default="data/buffer")
        parser.add_argument("--minibatches-per-gen", type=int, default=1000)
        parser.add_argument("--max-epochs", type=int, default=10,
                            help="Max epochs per generation with early stopping (default: 10)")
        parser.add_argument("--n-epochs", type=int, default=None, dest="n_epochs_alias",
                            help="Backward compat alias for --max-epochs")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--optimizer", type=str, default="muon",
                            choices=["adam", "adamw", "muon"])
        parser.add_argument("--lr-schedule", type=str, default="")
        parser.add_argument("--initial-lr", type=float, default=0.02)
        parser.add_argument("--eval-max-games", type=int, default=400,
                            help="Max games for SPRT evaluation (default: 400)")
        parser.add_argument("--eval-simulations", type=int, default=800)
        parser.add_argument("--eval-top-p-base", type=float, default=0.95,
                            help="Top-p sampling base for eval move selection (default: 0.95)")
        parser.add_argument("--sprt-elo0", type=float, default=0.0,
                            help="SPRT H0 elo (default: 0.0)")
        parser.add_argument("--sprt-elo1", type=float, default=10.0,
                            help="SPRT H1 elo (default: 10.0)")
        parser.add_argument("--sprt-alpha", type=float, default=0.05,
                            help="SPRT type I error rate (default: 0.05)")
        parser.add_argument("--sprt-beta", type=float, default=0.05,
                            help="SPRT type II error rate (default: 0.05)")
        parser.add_argument("--weights-dir", type=str, default="weights")
        parser.add_argument("--data-dir", type=str, default="data")
        parser.add_argument("--log-file", type=str, default="training_log.jsonl")
        parser.add_argument("--no-resume", action="store_true")
        parser.add_argument("--max-generations", type=int, default=0,
                            help="Max generations to run (0 = unlimited)")
        parser.add_argument("--disable-tier1", action="store_true",
                            help="Disable Tier 1 safety gates (mate search + KOTH)")
        parser.add_argument("--disable-material", action="store_true",
                            help="Disable material value integration (pure AlphaZero)")
        parser.add_argument("--log-games", type=str, default="first",
                            choices=["all", "first", "none"],
                            help="Which self-play games to log (default: first)")
        parser.add_argument("--inference-batch-size", type=int, default=16,
                            help="Batch size for GPU inference server (default: 16)")
        parser.add_argument("--game-threads", type=int, default=0,
                            help="Parallel game threads for self-play/eval (0 = auto)")
        parser.add_argument("--skip-self-play", action="store_true",
                            help="Skip self-play after gen 1 (eval games produce training data)")
        parser.add_argument("--multi-variant", action="store_true",
                            help="Train all 3 variants (policy-only, value-only, all) instead of just 'all'")
        parser.add_argument("--num-blocks", type=int, default=6,
                            help="Number of residual blocks in OracleNet (default: 6)")
        parser.add_argument("--hidden-dim", type=int, default=128,
                            help="Hidden dimension of OracleNet (default: 128)")

        args = parser.parse_args()
        # Resolve max_epochs: --n-epochs alias overrides --max-epochs if provided
        max_epochs = args.n_epochs_alias if args.n_epochs_alias is not None else args.max_epochs
        return cls(
            games_per_generation=args.games_per_generation,
            simulations_per_move=args.simulations_per_move,
            sims_schedule=args.sims_schedule,
            enable_koth=args.enable_koth,
            enable_tier1=not args.disable_tier1,
            enable_material_value=not args.disable_material,
            buffer_capacity=args.buffer_capacity,
            buffer_dir=args.buffer_dir,
            minibatches_per_generation=args.minibatches_per_gen,
            max_epochs=max_epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            lr_schedule=args.lr_schedule,
            initial_lr=args.initial_lr,
            eval_max_games=args.eval_max_games,
            eval_simulations=args.eval_simulations,
            eval_top_p_base=args.eval_top_p_base,
            sprt_elo0=args.sprt_elo0,
            sprt_elo1=args.sprt_elo1,
            sprt_alpha=args.sprt_alpha,
            sprt_beta=args.sprt_beta,
            weights_dir=args.weights_dir,
            data_dir=args.data_dir,
            log_file=args.log_file,
            resume=not args.no_resume,
            max_generations=args.max_generations,
            log_games=args.log_games,
            inference_batch_size=args.inference_batch_size,
            game_threads=args.game_threads,
            single_variant=not args.multi_variant,
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            skip_self_play=args.skip_self_play,
        )


@dataclass
class OrchestratorState:
    generation: int = 0
    current_best_pth: str = ""
    current_best_pt: str = ""
    latest_pth: str = ""  # most recent candidate .pth (accepted or rejected) — train from this
    global_minibatches: int = 0
    reset_optimizer_next: bool = False
    accepted_count: int = 0  # number of models accepted
    model_elos: dict = field(default_factory=dict)  # str(accepted_count) → cumulative Elo

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        # Backward compat: old state files may not have model_elos
        if "model_elos" not in data:
            data["model_elos"] = {}
        # Backward compat: old state files may not have latest_pth
        if "latest_pth" not in data:
            data["latest_pth"] = data.get("current_best_pth", "")
        return cls(**data)


def export_model_for_rust(model, output_path):
    """Export PyTorch model to TorchScript for Rust integration.

    Traces on CUDA if available so that device-dependent ops (like
    torch.arange(..., device=x.device) in the k-head) are correctly
    baked in for GPU inference in Rust.
    """
    device = next(model.parameters()).device
    model.eval()
    example_board = torch.randn(1, 17, 8, 8, device=device)
    example_scalars = torch.randn(1, 2, device=device)  # [material, qsearch_flag]
    traced = torch.jit.trace(model, (example_board, example_scalars))
    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path} (traced on {device})")


def get_libtorch_env():
    """Get environment with LibTorch and NVIDIA CUDA libs on LD_LIBRARY_PATH."""
    import multiprocessing
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

    # Collect nvidia pip package lib dirs (cudart, cublas, cudnn, etc.)
    # These are needed for libtorch_cuda.so to load successfully.
    nvidia_dir = os.path.join(os.path.dirname(torch.__file__), "..", "nvidia")
    nvidia_lib_paths = []
    if os.path.isdir(nvidia_dir):
        for name in os.listdir(nvidia_dir):
            lib_dir = os.path.join(nvidia_dir, name, "lib")
            if os.path.isdir(lib_dir):
                nvidia_lib_paths.append(lib_dir)

    all_paths = [torch_lib_path] + nvidia_lib_paths
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(all_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LIBTORCH_USE_PYTORCH"] = "1"
    env["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
    # Ensure CUDA is visible to tch-rs (defaults to GPU 0 if not set)
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    # Leave 1 core for inference server thread + OS
    env["RAYON_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() // 2 - 1))
    return env


class Orchestrator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = OrchestratorState()
        self.state_path = os.path.join(config.data_dir, "orchestrator_state.json")
        self._last_training_losses = {}
        self._current_generation = 0

        # Ensure directories
        os.makedirs(config.weights_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.buffer_dir, exist_ok=True)

    def initialize_gen0(self):
        """Create and export a fresh gen-0 model."""
        gen0_pt = os.path.join(self.config.weights_dir, "gen_0.pt")
        gen0_pth = os.path.join(self.config.weights_dir, "gen_0.pth")

        if not os.path.exists(gen0_pt):
            print("Initializing Generation 0...")
            torch.manual_seed(0)
            model = OracleNet(num_blocks=self.config.num_blocks, hidden_dim=self.config.hidden_dim)
            # Move to CUDA so TorchScript traces device-dependent ops correctly
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            export_model_for_rust(model, gen0_pt)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "global_minibatch": 0,
            }, gen0_pth)

        self.state.current_best_pt = gen0_pt
        self.state.current_best_pth = gen0_pth
        self.state.latest_pth = gen0_pth
        self.state.model_elos = {"0": 0.0}

    def log_entry(self, entry: dict):
        """Append a JSON line to the log file."""
        os.makedirs(os.path.dirname(self.config.log_file) or ".", exist_ok=True)
        with open(self.config.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def handle_eval_result(self, accepted, generation, candidate_pt, candidate_pth, eval_results):
        """Update state based on evaluation results."""
        if accepted:
            # Copy candidate to gen_N
            gen_pt = os.path.join(self.config.weights_dir, f"gen_{generation}.pt")
            gen_pth = os.path.join(self.config.weights_dir, f"gen_{generation}.pth")
            shutil.copy2(candidate_pt, gen_pt)
            shutil.copy2(candidate_pth, gen_pth)

            self.state.current_best_pt = gen_pt
            self.state.current_best_pth = gen_pth
            self.state.latest_pth = gen_pth

            # Compute cumulative Elo for new model
            winrate = max(0.01, min(0.99, eval_results.get("winrate", 0.5)))
            elo_delta = -400 * math.log10(1.0 / winrate - 1.0)
            prev_elo = self.state.model_elos.get(str(self.state.accepted_count), 0.0)
            self.state.accepted_count += 1
            self.state.model_elos[str(self.state.accepted_count)] = prev_elo + elo_delta

            # Update latest symlink
            latest_pt = os.path.join(self.config.weights_dir, "latest.pt")
            if os.path.exists(latest_pt) or os.path.islink(latest_pt):
                os.remove(latest_pt)
            os.symlink(os.path.abspath(gen_pt), latest_pt)

            # Note: we do NOT clear the buffer on accept. Elo-weighted sampling
            # already downweights old data, and early stopping prevents overfitting.
            # Clearing causes data starvation at small data scales.

            print(f"Generation {generation} ACCEPTED (W:{eval_results['wins']} "
                  f"L:{eval_results['losses']} D:{eval_results['draws']} "
                  f"Elo: {self.state.model_elos[str(self.state.accepted_count)]:.1f})")
        else:
            # Rejected — keep candidate as training base for next generation
            self.state.latest_pth = candidate_pth
            print(f"Generation {generation} REJECTED (W:{eval_results['wins']} "
                  f"L:{eval_results['losses']} D:{eval_results['draws']})")

    def _get_sims_for_gen(self, generation):
        """Resolve simulations count for a generation from schedule or default."""
        if not self.config.sims_schedule:
            return self.config.simulations_per_move
        # Parse "0:200,10:400,20:800" into sorted list of (gen_threshold, sims)
        entries = []
        for part in self.config.sims_schedule.split(","):
            gen_str, sims_str = part.strip().split(":")
            entries.append((int(gen_str), int(sims_str)))
        entries.sort()
        # Find the last entry whose threshold <= generation
        sims = entries[0][1]  # default to first entry
        for threshold, s in entries:
            if generation >= threshold:
                sims = s
            else:
                break
        return sims

    def run_self_play(self, generation):
        """Run self-play games and return the output directory."""
        data_dir = os.path.join(self.config.data_dir, f"gen_{generation}")
        os.makedirs(data_dir, exist_ok=True)

        sims = self._get_sims_for_gen(generation)
        seed_offset = (generation - 1) * self.config.games_per_generation
        cmd = [
            "cargo", "run", "--release", "--features", "neural", "--bin", "self_play", "--",
            str(self.config.games_per_generation),
            str(sims),
            data_dir,
            self.state.current_best_pt,
            "true" if self.config.enable_koth else "false",
            str(self.config.enable_tier1).lower(),
            str(self.config.enable_material_value).lower(),
            self.config.log_games,
            "--batch-size", str(self.config.inference_batch_size),
            "--seed-offset", str(seed_offset),
        ]
        if self.config.game_threads > 0:
            cmd.extend(["--threads", str(self.config.game_threads)])

        print(f"Self-play: {self.config.games_per_generation} games, "
              f"{sims} sims, "
              f"batch_size={self.config.inference_batch_size}...")
        result = subprocess.run(
            cmd, env=get_libtorch_env(),
            capture_output=True, text=True, check=True,
        )

        # Save game logs (stdout = moves/results, stderr = training sample details)
        log_path = os.path.join(data_dir, "self_play_games.txt")
        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- Training Sample Details (stderr) ---\n")
                f.write(result.stderr)
        if result.stdout:
            print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        if result.stderr:
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
                  file=sys.stderr)

        return data_dir

    def clear_buffer(self):
        """Clear all data from the replay buffer (called on model acceptance)."""
        buf = ReplayBuffer(
            capacity_positions=self.config.buffer_capacity,
            buffer_dir=self.config.buffer_dir,
        )
        buf.load_manifest()
        for entry in buf.entries:
            try:
                os.remove(entry["path"])
            except FileNotFoundError:
                pass
        buf.entries = []
        buf.save_manifest()
        print("Buffer cleared (new model accepted)")

    def update_buffer(self, game_data_dir):
        """Add new games to replay buffer and evict if over capacity."""
        current_elo = self.state.model_elos.get(str(self.state.accepted_count), 0.0)
        buf = ReplayBuffer(
            capacity_positions=self.config.buffer_capacity,
            buffer_dir=self.config.buffer_dir,
        )
        buf.load_manifest()
        added = buf.add_games(game_data_dir, model_elo=current_elo)
        buf.evict_oldest()
        total = buf.total_positions()
        print(f"Buffer: +{added} positions (elo={current_elo:.1f}), {total} total")
        return total

    def _add_eval_data_to_buffer(self, bin_dir, model_elo):
        """Add eval training data to the replay buffer."""
        buf = ReplayBuffer(
            capacity_positions=self.config.buffer_capacity,
            buffer_dir=self.config.buffer_dir,
        )
        buf.load_manifest()
        added = buf.add_games(bin_dir, model_elo=model_elo)
        buf.evict_oldest()
        return added

    def _get_buffer_size(self):
        """Get current buffer size without adding any data."""
        buf = ReplayBuffer(
            capacity_positions=self.config.buffer_capacity,
            buffer_dir=self.config.buffer_dir,
        )
        buf.load_manifest()
        return buf.total_positions()

    def _compute_adaptive_minibatches(self, buffer_positions):
        """Compute minibatches targeting ~1.5 epochs over the buffer.

        Prevents overfitting by scaling training to buffer size rather than
        using a fixed count. Early generations with small buffers get fewer
        steps; later generations with large buffers get more.
        """
        target = max(100, int(3.0 * buffer_positions / self.config.batch_size))
        actual = min(target, self.config.minibatches_per_generation)
        return actual

    def run_training(self, generation, buffer_positions=None, train_heads="all", suffix=""):
        """Train on replay buffer, return (candidate_pth, candidate_pt).

        train_heads: 'all', 'policy' (freeze value+k), or 'value' (freeze policy).
        suffix: appended to filename, e.g. '_policy' -> candidate_3_policy.pth
        """
        candidate_pth = os.path.join(self.config.weights_dir, f"candidate_{generation}{suffix}.pth")
        candidate_pt = os.path.join(self.config.weights_dir, f"candidate_{generation}{suffix}.pt")

        lr = self.config.initial_lr

        resume_path = self.state.current_best_pth
        cmd = [
            "python3", "python/train.py",
            "--buffer-dir", self.config.buffer_dir,
            self.config.buffer_dir,   # data_dir (ignored when --buffer-dir set)
            candidate_pth,             # output_path
            resume_path,               # resume_path (latest candidate, not necessarily best)
            "--optimizer", self.config.optimizer,
            "--lr", str(lr),
            "--batch-size", str(self.config.batch_size),
            "--use-epochs",
            "--max-epochs", str(self.config.max_epochs),
        ]
        if self.config.lr_schedule:
            cmd.extend(["--lr-schedule", self.config.lr_schedule])

        if not self.config.enable_material_value:
            cmd.append("--disable-material")

        cmd.extend(["--num-blocks", str(self.config.num_blocks)])
        cmd.extend(["--hidden-dim", str(self.config.hidden_dim)])

        if train_heads != "all":
            cmd.extend(["--train-heads", train_heads])

        print(f"Training ({train_heads}): up to {self.config.max_epochs} epoch(s), "
              f"optimizer={self.config.optimizer}, lr={lr}, resume={resume_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Save full training output
        data_dir = os.path.join(self.config.data_dir, f"gen_{generation}")
        os.makedirs(data_dir, exist_ok=True)
        stats_path = os.path.join(data_dir, f"training_stats{suffix}.txt")
        with open(stats_path, "w") as f:
            f.write(result.stdout)

        # Store training config used for logging
        self._last_training_losses = {"max_epochs": self.config.max_epochs}

        # Parse BEST_EPOCH and VAL_LOSS from train.py output
        import re
        for line in result.stdout.splitlines():
            m_be = re.match(r'BEST_EPOCH=(\d+)', line)
            if m_be:
                self._last_training_losses["best_epoch"] = int(m_be.group(1))
            m_vl = re.match(r'VAL_LOSS=([\d.]+)', line)
            if m_vl:
                self._last_training_losses["val_loss"] = float(m_vl.group(1))

        for line in reversed(result.stdout.splitlines()):
            if "Loss=" in line and "P=" in line and "V=" in line:
                m_loss = re.search(r'Loss=([\d.]+)', line)
                m_ploss = re.search(r'P=([\d.]+)', line)
                m_vloss = re.search(r'V=([\d.]+)', line)
                m_k = re.search(r'K=([\d.]+)', line)
                if m_loss:
                    self._last_training_losses["loss"] = float(m_loss.group(1))
                if m_ploss:
                    self._last_training_losses["policy_loss"] = float(m_ploss.group(1))
                if m_vloss:
                    self._last_training_losses["value_loss"] = float(m_vloss.group(1))
                if m_k:
                    self._last_training_losses["k_mean"] = float(m_k.group(1))
                break
        # Print training summary for visibility
        if result.stdout:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        # Export to TorchScript
        print("Exporting candidate model...")
        model = OracleNet(num_blocks=self.config.num_blocks, hidden_dim=self.config.hidden_dim)
        checkpoint = torch.load(candidate_pth, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        # Move to CUDA so TorchScript traces device-dependent ops correctly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        export_model_for_rust(model, candidate_pt)

        return candidate_pth, candidate_pt

    def run_evaluation(self, candidate_pt, generation=None, suffix=""):
        """Evaluate candidate vs current best using SPRT. Returns (accepted, results_dict)."""
        eval_sims = self.config.eval_simulations
        if generation is not None and self.config.sims_schedule:
            eval_sims = self._get_sims_for_gen(generation)
        cmd = [
            "cargo", "run", "--release", "--features", "neural",
            "--bin", "evaluate_models", "--",
            candidate_pt,
            self.state.current_best_pt,
            str(self.config.eval_max_games),
            str(eval_sims),
            "--sprt",
            "--elo0", str(self.config.sprt_elo0),
            "--elo1", str(self.config.sprt_elo1),
            "--sprt-alpha", str(self.config.sprt_alpha),
            "--sprt-beta", str(self.config.sprt_beta),
        ]
        if self.config.enable_koth:
            cmd.append("--enable-koth")
        if not self.config.enable_tier1:
            cmd.append("--disable-tier1")
        if not self.config.enable_material_value:
            cmd.append("--disable-material")
        cmd.extend(["--batch-size", str(self.config.inference_batch_size)])
        cmd.extend(["--top-p-base", str(self.config.eval_top_p_base)])
        if self.config.game_threads > 0:
            cmd.extend(["--threads", str(self.config.game_threads)])
        if generation is not None:
            # Same seed offset for all variants so games are directly comparable
            eval_seed_offset = generation * self.config.eval_max_games
            cmd.extend(["--seed-offset", str(eval_seed_offset)])

        # Save training data from eval games for buffer ingestion
        eval_data_dir = os.path.join(self.config.data_dir, f"gen_{self._current_generation}", "eval_data")
        cmd.extend(["--save-training-data", eval_data_dir])

        print(f"Evaluating: up to {self.config.eval_max_games} games @ {eval_sims} sims (SPRT "
              f"elo0={self.config.sprt_elo0}, elo1={self.config.sprt_elo1}, "
              f"alpha={self.config.sprt_alpha}, beta={self.config.sprt_beta})")

        result = subprocess.run(
            cmd, env=get_libtorch_env(),
            capture_output=True, text=True,
        )

        # Save eval game logs
        data_dir = os.path.join(self.config.data_dir, f"gen_{self._current_generation}")
        os.makedirs(data_dir, exist_ok=True)
        eval_log_path = os.path.join(data_dir, f"eval_games{suffix}.txt")
        with open(eval_log_path, "w") as f:
            if result.stderr:
                f.write(result.stderr)
            f.write("\n--- RESULTS ---\n")
            f.write(result.stdout)

        # Parse stdout: "WINS=X LOSSES=Y DRAWS=Z WINRATE=0.XX ACCEPTED=true/false
        #                 GAMES_PLAYED=N LLR=Y.YY SPRT_RESULT=H1/H0/inconclusive"
        output = result.stdout.strip()
        parts = {}
        for token in output.split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v

        wins = int(parts.get("WINS", 0))
        losses = int(parts.get("LOSSES", 0))
        draws = int(parts.get("DRAWS", 0))
        winrate = float(parts.get("WINRATE", 0.0))
        games_played = int(parts.get("GAMES_PLAYED", wins + losses + draws))
        llr = float(parts.get("LLR", 0.0))
        sprt_result = parts.get("SPRT_RESULT", "inconclusive")

        # Accept only if SPRT decided H1
        accepted = sprt_result == "H1"

        return accepted, {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": winrate,
            "games_played": games_played,
            "llr": llr,
            "sprt_result": sprt_result,
        }

    def _write_overview(self, generation, buffer_size, accepted, eval_results):
        """Write per-generation overview file with config, timing, and results."""
        data_dir = os.path.join(self.config.data_dir, f"gen_{generation}")
        os.makedirs(data_dir, exist_ok=True)
        overview_path = os.path.join(data_dir, "overview.txt")

        lines = [
            f"Generation {generation}",
            f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "",
            "=== Hyperparameters ===",
        ]
        for k, v in asdict(self.config).items():
            lines.append(f"  {k}: {v}")

        lines.extend([
            "",
            "=== Buffer ===",
            f"  Size after update: {buffer_size}",
            "",
            "=== Training ===",
        ])
        if self._last_training_losses:
            for k, v in self._last_training_losses.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  (no losses parsed)")

        lines.extend([
            "",
            "=== Evaluation ===",
            f"  Wins: {eval_results['wins']}",
            f"  Losses: {eval_results['losses']}",
            f"  Draws: {eval_results['draws']}",
            f"  Winrate: {eval_results['winrate']:.4f}",
            f"  Result: {'ACCEPTED' if accepted else 'REJECTED'}",
            "",
            "=== Current Best ===",
            f"  {self.state.current_best_pt}",
        ])

        with open(overview_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def save_state(self):
        """Save orchestrator state for resumability."""
        self.state.save(self.state_path)

    def load_state(self):
        """Load orchestrator state if it exists."""
        if os.path.exists(self.state_path):
            self.state = OrchestratorState.load(self.state_path)
            return True
        return False

    def run(self):
        """Main AGZ training loop."""
        # Initialize or resume
        if self.config.resume and self.load_state():
            print(f"Resuming from generation {self.state.generation}")
            start_gen = self.state.generation + 1
        else:
            self.initialize_gen0()
            start_gen = 1
            self.save_state()

        generation = start_gen
        while self.config.max_generations == 0 or generation <= start_gen + self.config.max_generations - 1:
            print(f"\n{'='*60}")
            print(f"=== Generation {generation} ===")
            print(f"{'='*60}")

            self._current_generation = generation

            # 1. Self-play (skip after gen 1 if configured)
            if not self.config.skip_self_play or generation <= 1:
                game_data_dir = self.run_self_play(generation)
                buffer_size = self.update_buffer(game_data_dir)
            else:
                buffer_size = self._get_buffer_size()

            # 3. Train variant(s) & evaluate each via SPRT
            if self.config.single_variant:
                variants = [("all", "")]
            else:
                variants = [
                    ("policy", "_policy"),
                    ("value", "_value"),
                    ("all", ""),
                ]

            best_variant = None  # (accepted, train_heads, pth, pt, eval_results, training_losses)

            for train_heads, suffix in variants:
                print(f"\n--- Variant: train_heads={train_heads} ---")
                candidate_pth, candidate_pt = self.run_training(
                    generation, buffer_positions=buffer_size,
                    train_heads=train_heads, suffix=suffix,
                )
                variant_training_losses = dict(self._last_training_losses)

                accepted, eval_results = self.run_evaluation(candidate_pt, generation=generation, suffix=suffix)
                print(f"  {train_heads}: W:{eval_results['wins']} L:{eval_results['losses']} "
                      f"D:{eval_results['draws']} WR:{eval_results['winrate']:.3f} "
                      f"LLR:{eval_results.get('llr', 'N/A')} -> "
                      f"{'ACCEPTED' if accepted else 'rejected'}")

                if accepted:
                    # Pick the accepted variant with highest winrate
                    if best_variant is None or eval_results["winrate"] > best_variant[4]["winrate"]:
                        best_variant = (True, train_heads, candidate_pth, candidate_pt,
                                        eval_results, variant_training_losses)

            # 4. Accept best passing variant, or reject all
            if best_variant:
                accepted, train_heads, candidate_pth, candidate_pt, eval_results, training_losses = best_variant
                print(f"\n>>> Accepting variant: train_heads={train_heads} "
                      f"(WR={eval_results['winrate']:.3f})")
                self._last_training_losses = training_losses
            else:
                # No variant passed — use the 'all' variant for logging
                accepted = False
                train_heads = "all"
                # candidate_pth/pt already set from last iteration (the 'all' variant)
                self._last_training_losses = variant_training_losses
                print(f"\n>>> No variant passed SPRT")

            # Capture current Elo before it may change
            current_elo = self.state.model_elos.get(str(self.state.accepted_count), 0.0)

            self.handle_eval_result(
                accepted=accepted,
                generation=generation,
                candidate_pt=candidate_pt,
                candidate_pth=candidate_pth,
                eval_results=eval_results,
            )

            # 4b. Ingest eval game training data into buffer
            eval_data_dir = os.path.join(self.config.data_dir, f"gen_{generation}", "eval_data")
            current_eval_dir = os.path.join(eval_data_dir, "current")
            candidate_eval_dir = os.path.join(eval_data_dir, "candidate")
            eval_added = 0

            # Compute Elo delta from measured winrate
            winrate_clamped = max(0.01, min(0.99, eval_results.get("winrate", 0.5)))
            elo_delta = -400 * math.log10(1.0 / winrate_clamped - 1.0)

            if accepted:
                # Winner = candidate (new_elo), Loser = current (current_elo)
                winner_elo = self.state.model_elos.get(str(self.state.accepted_count), 0.0)
                loser_elo = current_elo
                winner_dir, loser_dir = candidate_eval_dir, current_eval_dir
            else:
                # Don't clear — accumulate between accepts
                # Winner = current, Loser = candidate (capped at current_elo)
                winner_elo = current_elo
                loser_elo = current_elo + min(0.0, elo_delta)
                winner_dir, loser_dir = current_eval_dir, candidate_eval_dir

            # Ingest both sides
            if os.path.isdir(winner_dir):
                eval_added += self._add_eval_data_to_buffer(winner_dir, winner_elo)
            if os.path.isdir(loser_dir):
                eval_added += self._add_eval_data_to_buffer(loser_dir, loser_elo)

            if eval_added > 0:
                print(f"Buffer: +{eval_added} eval positions ingested")
                buffer_size += eval_added

            # 5. Log
            log = {
                "gen": generation,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "games_generated": self.config.games_per_generation if (not self.config.skip_self_play or generation <= 1) else 0,
                "simulations": self._get_sims_for_gen(generation),
                "buffer_size": buffer_size,
                "eval_wins": eval_results["wins"],
                "eval_losses": eval_results["losses"],
                "eval_draws": eval_results["draws"],
                "eval_winrate": eval_results["winrate"],
                "eval_games_played": eval_results.get("games_played"),
                "eval_llr": eval_results.get("llr"),
                "eval_sprt_result": eval_results.get("sprt_result"),
                "accepted": accepted,
                "accepted_variant": train_heads if accepted else None,
                "current_best": self.state.current_best_pt,
                "current_model_elo": self.state.model_elos.get(str(self.state.accepted_count), 0.0),
            }
            if self._last_training_losses:
                log["training_loss"] = self._last_training_losses.get("loss")
                log["training_policy_loss"] = self._last_training_losses.get("policy_loss")
                log["training_value_loss"] = self._last_training_losses.get("value_loss")
                log["training_k_mean"] = self._last_training_losses.get("k_mean")
                log["max_epochs"] = self._last_training_losses.get("max_epochs")
                log["best_epoch"] = self._last_training_losses.get("best_epoch")
                log["val_loss"] = self._last_training_losses.get("val_loss")
            self.log_entry(log)

            # 6. Write overview file
            self._write_overview(generation, buffer_size, accepted, eval_results)

            # 8. Save state
            self.state.generation = generation
            self.save_state()

            generation += 1


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    orch = Orchestrator(config)
    orch.run()
