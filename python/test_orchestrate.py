"""Tests for the AGZ orchestrator."""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

from orchestrate import TrainingConfig, OrchestratorState, Orchestrator
from replay_buffer import SAMPLE_SIZE_FLOATS


def make_fake_bin(path, num_positions):
    """Create a fake .bin file."""
    data = np.random.randn(num_positions, SAMPLE_SIZE_FLOATS).astype(np.float32)
    data.tofile(path)


@pytest.fixture
def tmp_workspace():
    """Create a temporary workspace directory."""
    workspace = tempfile.mkdtemp(prefix="orch_test_")
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


class TestTrainingConfig:
    def test_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.games_per_generation == 100
        assert cfg.simulations_per_move == 800
        assert cfg.enable_koth is False
        assert cfg.buffer_capacity == 500_000
        assert cfg.minibatches_per_generation == 1000
        assert cfg.batch_size == 64
        assert cfg.optimizer == "muon"
        assert cfg.initial_lr == 0.02
        assert cfg.eval_games == 100
        assert cfg.eval_simulations == 800
        assert cfg.acceptance_threshold == 0.55
        assert cfg.resume is True

    def test_config_from_args(self):
        test_args = [
            "orchestrate.py",
            "--games-per-generation", "500",
            "--optimizer", "adam",
            "--eval-games", "200",
            "--acceptance-threshold", "0.60",
            "--buffer-capacity", "1000000",
        ]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.games_per_generation == 500
        assert cfg.optimizer == "adam"
        assert cfg.eval_games == 200
        assert cfg.acceptance_threshold == 0.60
        assert cfg.buffer_capacity == 1000000


class TestOrchestratorState:
    def test_orchestrator_state_save_load_roundtrip(self, tmp_workspace):
        state = OrchestratorState(
            generation=5,
            current_best_pth="weights/gen_4.pth",
            current_best_pt="weights/gen_4.pt",
            global_minibatches=5000,
        )
        state_path = os.path.join(tmp_workspace, "state.json")
        state.save(state_path)

        loaded = OrchestratorState.load(state_path)
        assert loaded.generation == 5
        assert loaded.current_best_pth == "weights/gen_4.pth"
        assert loaded.current_best_pt == "weights/gen_4.pt"
        assert loaded.global_minibatches == 5000

    def test_resume_skips_completed_generation(self, tmp_workspace):
        state = OrchestratorState(
            generation=5,
            current_best_pth="weights/gen_4.pth",
            current_best_pt="weights/gen_4.pt",
            global_minibatches=5000,
        )
        state_path = os.path.join(tmp_workspace, "state.json")
        state.save(state_path)

        loaded = OrchestratorState.load(state_path)
        # The orchestrator should start from generation 6
        assert loaded.generation + 1 == 6


class TestOrchestrator:
    def test_initialize_gen0_creates_model(self, tmp_workspace):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
        )
        orch = Orchestrator(cfg)
        orch.initialize_gen0()

        gen0_pt = os.path.join(cfg.weights_dir, "gen_0.pt")
        gen0_pth = os.path.join(cfg.weights_dir, "gen_0.pth")
        assert os.path.exists(gen0_pt)
        assert os.path.exists(gen0_pth)

    def test_log_entry_format(self, tmp_workspace):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
        )
        orch = Orchestrator(cfg)

        entry = {
            "gen": 5,
            "timestamp": "2024-01-01T00:00:00",
            "games_generated": 100,
            "buffer_size": 45000,
            "train_loss": 2.31,
            "eval_wins": 58,
            "eval_losses": 35,
            "eval_draws": 7,
            "eval_winrate": 0.615,
            "accepted": True,
            "current_best": "weights/gen_5.pt",
        }
        orch.log_entry(entry)

        with open(cfg.log_file) as f:
            line = f.readline().strip()
            loaded = json.loads(line)
        assert loaded["gen"] == 5
        assert loaded["accepted"] is True
        assert loaded["eval_winrate"] == 0.615

    def test_rejected_model_keeps_current_best(self, tmp_workspace):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
        )
        orch = Orchestrator(cfg)
        orch.state.current_best_pt = "weights/gen_3.pt"
        orch.state.current_best_pth = "weights/gen_3.pth"

        # Simulate rejection
        orch.handle_eval_result(
            accepted=False,
            generation=4,
            candidate_pt="weights/candidate_4.pt",
            candidate_pth="weights/candidate_4.pth",
            eval_results={"wins": 40, "losses": 50, "draws": 10},
        )
        assert orch.state.current_best_pt == "weights/gen_3.pt"

    def test_accepted_model_updates_current_best(self, tmp_workspace):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
        )
        os.makedirs(cfg.weights_dir, exist_ok=True)
        orch = Orchestrator(cfg)
        orch.state.current_best_pt = "weights/gen_3.pt"
        orch.state.current_best_pth = "weights/gen_3.pth"

        # Create fake candidate files
        candidate_pt = os.path.join(cfg.weights_dir, "candidate_4.pt")
        candidate_pth = os.path.join(cfg.weights_dir, "candidate_4.pth")
        with open(candidate_pt, "w") as f:
            f.write("fake")
        with open(candidate_pth, "w") as f:
            f.write("fake")

        orch.handle_eval_result(
            accepted=True,
            generation=4,
            candidate_pt=candidate_pt,
            candidate_pth=candidate_pth,
            eval_results={"wins": 60, "losses": 30, "draws": 10},
        )
        assert orch.state.current_best_pt == os.path.join(cfg.weights_dir, "gen_4.pt")
        assert orch.state.current_best_pth == os.path.join(cfg.weights_dir, "gen_4.pth")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
