"""AGZ-Style Training Orchestrator.

Config-driven loop: self-play -> buffer -> train -> evaluate -> accept/reject.
"""

import os
import sys
import json
import shutil
import subprocess
import time
import argparse
from dataclasses import dataclass, field, asdict

import torch
from model import LogosNet
from replay_buffer import ReplayBuffer


@dataclass
class TrainingConfig:
    # Self-play
    games_per_generation: int = 100
    simulations_per_move: int = 800
    enable_koth: bool = False

    # Ablation flags
    enable_tier1: bool = True
    enable_material_value: bool = True
    log_games: str = "first"

    # Replay buffer (sliding window — old data evicted FIFO)
    buffer_capacity: int = 100_000
    buffer_dir: str = "data/buffer"
    sampling_half_life: int = 20_000  # recency weighting; 0 = uniform sampling

    # Training
    minibatches_per_generation: int = 1000
    batch_size: int = 64
    optimizer: str = "muon"
    lr_schedule: str = ""
    initial_lr: float = 0.02

    # Evaluation (SPRT)
    eval_max_games: int = 400
    eval_simulations: int = 800
    sprt_elo0: float = 0.0
    sprt_elo1: float = 10.0
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05

    # Parallelism
    inference_batch_size: int = 16
    game_threads: int = 0  # 0 = auto (RAYON_NUM_THREADS or rayon default)

    # Infrastructure
    weights_dir: str = "weights"
    data_dir: str = "data"
    log_file: str = "training_log.jsonl"
    resume: bool = True
    max_generations: int = 0  # 0 = unlimited

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="AGZ Training Orchestrator")
        parser.add_argument("--games-per-generation", type=int, default=100)
        parser.add_argument("--simulations-per-move", type=int, default=800)
        parser.add_argument("--enable-koth", action="store_true")
        parser.add_argument("--buffer-capacity", type=int, default=100_000)
        parser.add_argument("--buffer-dir", type=str, default="data/buffer")
        parser.add_argument("--sampling-half-life", type=int, default=20_000,
                            help="Recency sampling half-life in positions (0 = uniform)")
        parser.add_argument("--minibatches-per-gen", type=int, default=1000)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--optimizer", type=str, default="muon",
                            choices=["adam", "adamw", "muon"])
        parser.add_argument("--lr-schedule", type=str, default="")
        parser.add_argument("--initial-lr", type=float, default=0.02)
        parser.add_argument("--eval-max-games", type=int, default=400,
                            help="Max games for SPRT evaluation (default: 400)")
        parser.add_argument("--eval-simulations", type=int, default=800)
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

        args = parser.parse_args()
        return cls(
            games_per_generation=args.games_per_generation,
            simulations_per_move=args.simulations_per_move,
            enable_koth=args.enable_koth,
            enable_tier1=not args.disable_tier1,
            enable_material_value=not args.disable_material,
            buffer_capacity=args.buffer_capacity,
            buffer_dir=args.buffer_dir,
            sampling_half_life=args.sampling_half_life,
            minibatches_per_generation=args.minibatches_per_gen,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            lr_schedule=args.lr_schedule,
            initial_lr=args.initial_lr,
            eval_max_games=args.eval_max_games,
            eval_simulations=args.eval_simulations,
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
        )


@dataclass
class OrchestratorState:
    generation: int = 0
    current_best_pth: str = ""
    current_best_pt: str = ""
    global_minibatches: int = 0
    reset_optimizer_next: bool = False

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def export_model_for_rust(model, output_path):
    """Export PyTorch model to TorchScript for Rust integration."""
    model.eval()
    example_board = torch.randn(1, 17, 8, 8)
    example_material = torch.randn(1, 1)
    traced = torch.jit.trace(model, (example_board, example_material))
    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path}")


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
            model = LogosNet()
            export_model_for_rust(model, gen0_pt)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "global_minibatch": 0,
            }, gen0_pth)

        self.state.current_best_pt = gen0_pt
        self.state.current_best_pth = gen0_pth

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

            # Update latest symlink
            latest_pt = os.path.join(self.config.weights_dir, "latest.pt")
            if os.path.exists(latest_pt) or os.path.islink(latest_pt):
                os.remove(latest_pt)
            os.symlink(os.path.abspath(gen_pt), latest_pt)

            print(f"Generation {generation} ACCEPTED (W:{eval_results['wins']} "
                  f"L:{eval_results['losses']} D:{eval_results['draws']})")
        else:
            print(f"Generation {generation} REJECTED (W:{eval_results['wins']} "
                  f"L:{eval_results['losses']} D:{eval_results['draws']})")

    def run_self_play(self, generation):
        """Run self-play games and return the output directory."""
        data_dir = os.path.join(self.config.data_dir, f"gen_{generation}")
        os.makedirs(data_dir, exist_ok=True)

        cmd = [
            "cargo", "run", "--release", "--features", "neural", "--bin", "self_play", "--",
            str(self.config.games_per_generation),
            str(self.config.simulations_per_move),
            data_dir,
            self.state.current_best_pt,
            "true" if self.config.enable_koth else "false",
            str(self.config.enable_tier1).lower(),
            str(self.config.enable_material_value).lower(),
            self.config.log_games,
            "--batch-size", str(self.config.inference_batch_size),
        ]
        if self.config.game_threads > 0:
            cmd.extend(["--threads", str(self.config.game_threads)])

        print(f"Self-play: {self.config.games_per_generation} games, "
              f"{self.config.simulations_per_move} sims, "
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
        buf = ReplayBuffer(
            capacity_positions=self.config.buffer_capacity,
            buffer_dir=self.config.buffer_dir,
        )
        buf.load_manifest()
        added = buf.add_games(game_data_dir)
        buf.evict_oldest()
        total = buf.total_positions()
        print(f"Buffer: +{added} positions, {total} total")
        return total

    def _compute_adaptive_minibatches(self, buffer_positions):
        """Compute minibatches targeting ~1.5 epochs over the buffer.

        Prevents overfitting by scaling training to buffer size rather than
        using a fixed count. Early generations with small buffers get fewer
        steps; later generations with large buffers get more.
        """
        target = max(100, int(1.5 * buffer_positions / self.config.batch_size))
        actual = min(target, self.config.minibatches_per_generation)
        return actual

    def run_training(self, generation, buffer_positions=None):
        """Train on replay buffer, return (candidate_pth, candidate_pt)."""
        candidate_pth = os.path.join(self.config.weights_dir, f"candidate_{generation}.pth")
        candidate_pt = os.path.join(self.config.weights_dir, f"candidate_{generation}.pt")

        # Adaptive minibatches: scale to buffer size to prevent overfitting
        if buffer_positions is not None and buffer_positions > 0:
            minibatches = self._compute_adaptive_minibatches(buffer_positions)
            effective_epochs = (minibatches * self.config.batch_size) / buffer_positions
            print(f"Adaptive training: {minibatches} minibatches "
                  f"(~{effective_epochs:.1f} epochs over {buffer_positions} positions)")
        else:
            minibatches = self.config.minibatches_per_generation

        lr = self.config.initial_lr

        cmd = [
            "python3", "python/train.py",
            "--buffer-dir", self.config.buffer_dir,
            self.config.buffer_dir,   # data_dir (ignored when --buffer-dir set)
            candidate_pth,             # output_path
            self.state.current_best_pth,  # resume_path
            "--optimizer", self.config.optimizer,
            "--lr", str(lr),
            "--batch-size", str(self.config.batch_size),
            "--minibatches", str(minibatches),
        ]
        if self.config.lr_schedule:
            cmd.extend(["--lr-schedule", self.config.lr_schedule])

        cmd.extend(["--sampling-half-life", str(self.config.sampling_half_life)])

        if not self.config.enable_material_value:
            cmd.append("--disable-material")

        print(f"Training: {minibatches} minibatches, "
              f"optimizer={self.config.optimizer}, lr={lr}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Save full training output
        data_dir = os.path.join(self.config.data_dir, f"gen_{generation}")
        os.makedirs(data_dir, exist_ok=True)
        stats_path = os.path.join(data_dir, "training_stats.txt")
        with open(stats_path, "w") as f:
            f.write(result.stdout)

        # Store actual minibatches used for logging
        self._last_training_losses = {"actual_minibatches": minibatches}
        for line in reversed(result.stdout.splitlines()):
            if "Loss=" in line and "P=" in line and "V=" in line:
                import re
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
        model = LogosNet()
        checkpoint = torch.load(candidate_pth, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        export_model_for_rust(model, candidate_pt)

        return candidate_pth, candidate_pt

    def run_evaluation(self, candidate_pt):
        """Evaluate candidate vs current best using SPRT. Returns (accepted, results_dict)."""
        cmd = [
            "cargo", "run", "--release", "--features", "neural",
            "--bin", "evaluate_models", "--",
            candidate_pt,
            self.state.current_best_pt,
            str(self.config.eval_max_games),
            str(self.config.eval_simulations),
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
        if self.config.game_threads > 0:
            cmd.extend(["--threads", str(self.config.game_threads)])

        print(f"Evaluating: up to {self.config.eval_max_games} games (SPRT "
              f"elo0={self.config.sprt_elo0}, elo1={self.config.sprt_elo1}, "
              f"alpha={self.config.sprt_alpha}, beta={self.config.sprt_beta})")

        result = subprocess.run(
            cmd, env=get_libtorch_env(),
            capture_output=True, text=True,
        )

        # Save eval game logs
        data_dir = os.path.join(self.config.data_dir, f"gen_{self._current_generation}")
        os.makedirs(data_dir, exist_ok=True)
        eval_log_path = os.path.join(data_dir, "eval_games.txt")
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

            # 1. Self-play
            game_data_dir = self.run_self_play(generation)

            # 2. Buffer update
            buffer_size = self.update_buffer(game_data_dir)

            # 3. Train (adaptive minibatches based on buffer size)
            candidate_pth, candidate_pt = self.run_training(generation, buffer_positions=buffer_size)

            # 4. Evaluate
            accepted, eval_results = self.run_evaluation(candidate_pt)

            # 5. Accept or reject
            self.handle_eval_result(
                accepted=accepted,
                generation=generation,
                candidate_pt=candidate_pt,
                candidate_pth=candidate_pth,
                eval_results=eval_results,
            )

            # 6. Log
            log = {
                "gen": generation,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "games_generated": self.config.games_per_generation,
                "buffer_size": buffer_size,
                "eval_wins": eval_results["wins"],
                "eval_losses": eval_results["losses"],
                "eval_draws": eval_results["draws"],
                "eval_winrate": eval_results["winrate"],
                "eval_games_played": eval_results.get("games_played"),
                "eval_llr": eval_results.get("llr"),
                "eval_sprt_result": eval_results.get("sprt_result"),
                "accepted": accepted,
                "current_best": self.state.current_best_pt,
            }
            if self._last_training_losses:
                log["training_loss"] = self._last_training_losses.get("loss")
                log["training_policy_loss"] = self._last_training_losses.get("policy_loss")
                log["training_value_loss"] = self._last_training_losses.get("value_loss")
                log["training_k_mean"] = self._last_training_losses.get("k_mean")
                log["actual_minibatches"] = self._last_training_losses.get("actual_minibatches")
            self.log_entry(log)

            # 7. Write overview file
            self._write_overview(generation, buffer_size, accepted, eval_results)

            # 8. Save state
            self.state.generation = generation
            self.save_state()

            generation += 1


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    orch = Orchestrator(config)
    orch.run()
