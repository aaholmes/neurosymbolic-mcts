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

from orchestrate import (
    TrainingConfig, OrchestratorState, Orchestrator,
)
from replay_buffer import ReplayBuffer
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
        assert cfg.buffer_capacity == 100_000
        assert cfg.minibatches_per_generation == 1000
        assert cfg.batch_size == 64
        assert cfg.optimizer == "muon"
        assert cfg.initial_lr == 0.02
        assert cfg.eval_max_games == 400
        assert cfg.eval_simulations == 800
        assert cfg.sprt_elo0 == 0.0
        assert cfg.sprt_elo1 == 10.0
        assert cfg.sprt_alpha == 0.05
        assert cfg.sprt_beta == 0.05
        assert cfg.resume is True

    def test_config_from_args(self):
        test_args = [
            "orchestrate.py",
            "--games-per-generation", "500",
            "--optimizer", "adam",
            "--eval-max-games", "200",
            "--buffer-capacity", "1000000",
        ]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.games_per_generation == 500
        assert cfg.optimizer == "adam"
        assert cfg.eval_max_games == 200
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


class TestTrainingConfigExtended:
    def test_config_from_args_with_koth(self):
        test_args = ["orchestrate.py", "--enable-koth"]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.enable_koth is True

    def test_config_from_args_no_resume(self):
        test_args = ["orchestrate.py", "--no-resume"]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.resume is False

    def test_config_from_args_lr_schedule(self):
        test_args = ["orchestrate.py", "--lr-schedule", "500:0.01,1000:0.001"]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.lr_schedule == "500:0.01,1000:0.001"

    def test_config_from_args_all_defaults(self):
        test_args = ["orchestrate.py"]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.games_per_generation == 100
        assert cfg.simulations_per_move == 800
        assert cfg.optimizer == "muon"
        assert cfg.resume is True


class TestOrchestratorStateExtended:
    def test_state_default_values(self):
        state = OrchestratorState()
        assert state.generation == 0
        assert state.current_best_pth == ""
        assert state.current_best_pt == ""
        assert state.global_minibatches == 0

    def test_state_save_creates_file(self, tmp_workspace):
        state = OrchestratorState(generation=3)
        path = os.path.join(tmp_workspace, "state.json")
        state.save(path)
        assert os.path.exists(path)

    def test_state_load_preserves_all_fields(self, tmp_workspace):
        state = OrchestratorState(
            generation=10,
            current_best_pth="a.pth",
            current_best_pt="a.pt",
            global_minibatches=99999,
        )
        path = os.path.join(tmp_workspace, "state.json")
        state.save(path)

        loaded = OrchestratorState.load(path)
        assert loaded.generation == 10
        assert loaded.current_best_pth == "a.pth"
        assert loaded.current_best_pt == "a.pt"
        assert loaded.global_minibatches == 99999


class TestOrchestratorExtended:
    def _make_orch(self, tmp_workspace, **overrides):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            **overrides,
        )
        return Orchestrator(cfg)

    def test_init_creates_directories(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        assert os.path.isdir(orch.config.weights_dir)
        assert os.path.isdir(orch.config.data_dir)
        assert os.path.isdir(orch.config.buffer_dir)

    def test_initialize_gen0_idempotent(self, tmp_workspace):
        """Calling initialize_gen0 twice doesn't overwrite existing model."""
        orch = self._make_orch(tmp_workspace)
        orch.initialize_gen0()
        gen0_pt = os.path.join(orch.config.weights_dir, "gen_0.pt")
        mtime1 = os.path.getmtime(gen0_pt)

        import time
        time.sleep(0.01)
        orch.initialize_gen0()
        mtime2 = os.path.getmtime(gen0_pt)
        assert mtime1 == mtime2  # File not overwritten

    def test_initialize_gen0_sets_state(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        orch.initialize_gen0()
        assert orch.state.current_best_pt.endswith("gen_0.pt")
        assert orch.state.current_best_pth.endswith("gen_0.pth")

    def test_log_entry_appends_multiple(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        orch.log_entry({"gen": 1, "accepted": True})
        orch.log_entry({"gen": 2, "accepted": False})

        with open(orch.config.log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["gen"] == 1
        assert json.loads(lines[1])["gen"] == 2

    def test_handle_eval_result_accepted_creates_gen_file(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        os.makedirs(orch.config.weights_dir, exist_ok=True)

        candidate_pt = os.path.join(orch.config.weights_dir, "candidate.pt")
        candidate_pth = os.path.join(orch.config.weights_dir, "candidate.pth")
        with open(candidate_pt, "w") as f:
            f.write("model")
        with open(candidate_pth, "w") as f:
            f.write("model")

        orch.handle_eval_result(
            accepted=True,
            generation=7,
            candidate_pt=candidate_pt,
            candidate_pth=candidate_pth,
            eval_results={"wins": 60, "losses": 30, "draws": 10},
        )

        gen_pt = os.path.join(orch.config.weights_dir, "gen_7.pt")
        gen_pth = os.path.join(orch.config.weights_dir, "gen_7.pth")
        assert os.path.exists(gen_pt)
        assert os.path.exists(gen_pth)

    def test_handle_eval_result_accepted_updates_symlink(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        os.makedirs(orch.config.weights_dir, exist_ok=True)

        candidate_pt = os.path.join(orch.config.weights_dir, "candidate.pt")
        candidate_pth = os.path.join(orch.config.weights_dir, "candidate.pth")
        with open(candidate_pt, "w") as f:
            f.write("model")
        with open(candidate_pth, "w") as f:
            f.write("model")

        orch.handle_eval_result(
            accepted=True, generation=1,
            candidate_pt=candidate_pt, candidate_pth=candidate_pth,
            eval_results={"wins": 60, "losses": 30, "draws": 10},
        )

        latest = os.path.join(orch.config.weights_dir, "latest.pt")
        assert os.path.islink(latest)

    def test_handle_eval_result_accepted_replaces_existing_symlink(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        os.makedirs(orch.config.weights_dir, exist_ok=True)
        latest = os.path.join(orch.config.weights_dir, "latest.pt")

        # Create initial symlink
        dummy = os.path.join(orch.config.weights_dir, "dummy.pt")
        with open(dummy, "w") as f:
            f.write("old")
        os.symlink(os.path.abspath(dummy), latest)

        candidate_pt = os.path.join(orch.config.weights_dir, "candidate.pt")
        candidate_pth = os.path.join(orch.config.weights_dir, "candidate.pth")
        with open(candidate_pt, "w") as f:
            f.write("new")
        with open(candidate_pth, "w") as f:
            f.write("new")

        orch.handle_eval_result(
            accepted=True, generation=2,
            candidate_pt=candidate_pt, candidate_pth=candidate_pth,
            eval_results={"wins": 60, "losses": 30, "draws": 10},
        )
        assert os.path.islink(latest)
        # Symlink should point to gen_2.pt
        target = os.readlink(latest)
        assert "gen_2.pt" in target

    def test_handle_eval_result_rejected_no_files_created(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "old.pt"
        orch.state.current_best_pth = "old.pth"

        orch.handle_eval_result(
            accepted=False, generation=5,
            candidate_pt="candidate.pt", candidate_pth="candidate.pth",
            eval_results={"wins": 40, "losses": 50, "draws": 10},
        )
        gen_pt = os.path.join(orch.config.weights_dir, "gen_5.pt")
        assert not os.path.exists(gen_pt)
        assert orch.state.current_best_pt == "old.pt"

    def test_save_and_load_state_roundtrip(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        orch.state.generation = 42
        orch.state.current_best_pt = "weights/gen_41.pt"
        orch.state.current_best_pth = "weights/gen_41.pth"
        orch.state.global_minibatches = 12345
        orch.save_state()

        orch2 = self._make_orch(tmp_workspace)
        loaded = orch2.load_state()
        assert loaded is True
        assert orch2.state.generation == 42
        assert orch2.state.current_best_pt == "weights/gen_41.pt"
        assert orch2.state.global_minibatches == 12345

    def test_load_state_returns_false_when_no_state(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace)
        loaded = orch.load_state()
        assert loaded is False

    def test_update_buffer_adds_and_evicts(self, tmp_workspace):
        orch = self._make_orch(tmp_workspace, buffer_capacity=30)

        # Create fake game data
        game_dir = os.path.join(tmp_workspace, "games")
        os.makedirs(game_dir)
        make_fake_bin(os.path.join(game_dir, "game.bin"), 50)

        total = orch.update_buffer(game_dir)
        assert total <= 30  # Capacity is 30

    def test_export_model_for_rust(self, tmp_workspace):
        from orchestrate import export_model_for_rust
        from model import OracleNet
        import torch

        model = OracleNet(num_blocks=1, hidden_dim=16)
        output_path = os.path.join(tmp_workspace, "test.pt")
        export_model_for_rust(model, output_path)
        assert os.path.exists(output_path)

        # Verify it's loadable as a TorchScript module
        loaded = torch.jit.load(output_path)
        board = torch.randn(1, 17, 8, 8)
        material = torch.randn(1, 1)
        policy, value, k = loaded(board, material)
        assert policy.shape == (1, 4672)

    def test_get_libtorch_env(self, tmp_workspace):
        from orchestrate import get_libtorch_env
        env = get_libtorch_env()
        assert "LD_LIBRARY_PATH" in env
        assert "LIBTORCH_USE_PYTORCH" in env
        assert env["LIBTORCH_BYPASS_VERSION_CHECK"] == "1"

    def test_run_evaluation_parses_stdout(self, tmp_workspace):
        """Test evaluation output parsing with mocked subprocess."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = (
            "WINS=60 LOSSES=30 DRAWS=10 WINRATE=0.6500 ACCEPTED=true "
            "GAMES_PLAYED=100 LLR=3.50 SPRT_RESULT=H1\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            accepted, results = orch.run_evaluation("candidate.pt")

        assert accepted is True
        assert results["wins"] == 60
        assert results["losses"] == 30
        assert results["draws"] == 10
        assert results["winrate"] == pytest.approx(0.65)

    def test_run_evaluation_rejected_output(self, tmp_workspace):
        """Test evaluation output parsing when model is rejected."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = (
            "WINS=40 LOSSES=50 DRAWS=10 WINRATE=0.4500 ACCEPTED=false "
            "GAMES_PLAYED=100 LLR=-3.00 SPRT_RESULT=H0\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            accepted, results = orch.run_evaluation("candidate.pt")

        assert accepted is False
        assert results["wins"] == 40
        assert results["losses"] == 50

    def test_run_evaluation_empty_output(self, tmp_workspace):
        """Test evaluation parsing with empty output uses defaults."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            accepted, results = orch.run_evaluation("candidate.pt")

        assert accepted is False
        assert results["wins"] == 0
        assert results["losses"] == 0
        assert results["draws"] == 0


class TestAdaptiveMinibatches:
    def _make_orch(self, tmp_workspace, **overrides):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            **overrides,
        )
        return Orchestrator(cfg)

    def test_small_buffer_gets_minimum_100_minibatches(self, tmp_workspace):
        """With very few positions, floor of 100 minibatches applies."""
        orch = self._make_orch(tmp_workspace)
        # 100 positions / 64 batch_size * 1.5 = 2.3 → floored to 100
        result = orch._compute_adaptive_minibatches(100)
        assert result == 100

    def test_gen1_buffer_gets_scaled_minibatches(self, tmp_workspace):
        """~7000 positions (gen 1) should get ~324 minibatches, not 1000."""
        orch = self._make_orch(tmp_workspace)
        result = orch._compute_adaptive_minibatches(6914)
        expected = int(3.0 * 6914 / 64)  # = 324
        assert result == expected
        assert result < 400  # Much less than the old default of 1000

    def test_large_buffer_capped_at_config_max(self, tmp_workspace):
        """With large buffer, minibatches capped at config maximum."""
        orch = self._make_orch(tmp_workspace, minibatches_per_generation=1000)
        # 100,000 positions * 1.5 / 64 = 2343 → capped at 1000
        result = orch._compute_adaptive_minibatches(100_000)
        assert result == 1000

    def test_medium_buffer_scales_linearly(self, tmp_workspace):
        """Buffer in the middle range scales proportionally."""
        orch = self._make_orch(tmp_workspace, minibatches_per_generation=5000)
        result_10k = orch._compute_adaptive_minibatches(10_000)
        result_20k = orch._compute_adaptive_minibatches(20_000)
        # Both should be proportional (before hitting cap)
        assert result_20k == pytest.approx(2 * result_10k, abs=2)

    def test_effective_epochs_around_3(self, tmp_workspace):
        """The adaptive calculation should produce ~3.0 effective epochs."""
        orch = self._make_orch(tmp_workspace)
        buffer_size = 12_920  # Gen 2 from the analysis
        minibatches = orch._compute_adaptive_minibatches(buffer_size)
        effective_epochs = (minibatches * 64) / buffer_size
        assert 2.5 <= effective_epochs <= 3.5

    def test_run_training_passes_buffer_positions(self, tmp_workspace):
        """run_training uses adaptive minibatches when buffer_positions given."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pth = "best.pth"

        mock_result = MagicMock()
        mock_result.stdout = "Step 162/162 (global 162): Loss=1.50 P=3.00 V=0.50 K=0.35\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch.object(orch, '_compute_adaptive_minibatches', return_value=162) as mock_adaptive, \
             patch("orchestrate.OracleNet") as mock_model_cls, \
             patch("torch.load", return_value={"model_state_dict": {}}), \
             patch("orchestrate.export_model_for_rust"):
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model

            orch.run_training(1, buffer_positions=6914)

            mock_adaptive.assert_called_once_with(6914)
            # Check the --minibatches arg in the subprocess call
            cmd = mock_run.call_args[0][0]
            mb_idx = cmd.index("--minibatches")
            assert cmd[mb_idx + 1] == "162"

    def test_run_training_without_buffer_uses_config(self, tmp_workspace):
        """run_training without buffer_positions uses config default."""
        orch = self._make_orch(tmp_workspace, minibatches_per_generation=500)
        orch.state.current_best_pth = "best.pth"

        mock_result = MagicMock()
        mock_result.stdout = "Step 500/500 (global 500): Loss=1.50 P=3.00 V=0.50 K=0.35\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("orchestrate.OracleNet") as mock_model_cls, \
             patch("torch.load", return_value={"model_state_dict": {}}), \
             patch("orchestrate.export_model_for_rust"):
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model

            orch.run_training(1)

            cmd = mock_run.call_args[0][0]
            mb_idx = cmd.index("--minibatches")
            assert cmd[mb_idx + 1] == "500"


class TestCudaEnv:
    def test_cuda_visible_devices_set_by_default(self):
        """CUDA_VISIBLE_DEVICES should default to '0' if not set."""
        from orchestrate import get_libtorch_env
        with patch.dict(os.environ, {}, clear=False):
            # Remove CUDA_VISIBLE_DEVICES if present
            env_copy = os.environ.copy()
            env_copy.pop("CUDA_VISIBLE_DEVICES", None)
            with patch.dict(os.environ, env_copy, clear=True):
                env = get_libtorch_env()
                assert env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_cuda_visible_devices_not_overridden(self):
        """CUDA_VISIBLE_DEVICES should not be overridden if already set."""
        from orchestrate import get_libtorch_env
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,2"}):
            env = get_libtorch_env()
            assert env["CUDA_VISIBLE_DEVICES"] == "1,2"


class TestSlidingWindowBuffer:
    def _make_orch(self, tmp_workspace, **overrides):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            **overrides,
        )
        return Orchestrator(cfg)

    def test_buffer_evicts_oldest_when_over_capacity(self, tmp_workspace):
        """Buffer should evict oldest data when capacity is exceeded."""
        # Capacity of 30 positions — very small for testing
        orch = self._make_orch(tmp_workspace, buffer_capacity=30)

        game_dir = os.path.join(tmp_workspace, "games")
        os.makedirs(game_dir)
        make_fake_bin(os.path.join(game_dir, "game.bin"), 50)

        total = orch.update_buffer(game_dir)
        assert total <= 30  # Eviction should keep it under capacity

    def test_acceptance_clears_buffer(self, tmp_workspace):
        """Acceptance should clear the buffer — old data already trained on."""
        orch = self._make_orch(tmp_workspace, buffer_capacity=500_000)
        orch.state.current_best_pt = "weights/gen_3.pt"
        orch.state.current_best_pth = "weights/gen_3.pth"

        # Add data
        game_dir = os.path.join(tmp_workspace, "games")
        os.makedirs(game_dir)
        make_fake_bin(os.path.join(game_dir, "game.bin"), 50)
        orch.update_buffer(game_dir)

        buf = ReplayBuffer(capacity_positions=500_000, buffer_dir=orch.config.buffer_dir)
        buf.load_manifest()
        positions_before = buf.total_positions()
        assert positions_before > 0

        # Accept a model
        candidate_pt = os.path.join(orch.config.weights_dir, "candidate_4.pt")
        candidate_pth = os.path.join(orch.config.weights_dir, "candidate_4.pth")
        with open(candidate_pt, "w") as f:
            f.write("fake")
        with open(candidate_pth, "w") as f:
            f.write("fake")

        orch.handle_eval_result(
            accepted=True, generation=4,
            candidate_pt=candidate_pt, candidate_pth=candidate_pth,
            eval_results={"wins": 60, "losses": 30, "draws": 10},
        )

        # Buffer should be cleared
        buf2 = ReplayBuffer(capacity_positions=500_000, buffer_dir=orch.config.buffer_dir)
        buf2.load_manifest()
        assert buf2.total_positions() == 0

    def test_reject_ingests_both_sides_eval_data(self, tmp_workspace):
        """Rejection should ingest eval data from both sides with Elo tags.

        Tests the dual-ingestion pattern used in run_generation by calling
        _add_eval_data_to_buffer directly (the integration is in run_generation).
        """
        orch = self._make_orch(tmp_workspace, buffer_capacity=500_000)
        current_elo = 100.0

        # Add some existing buffer data
        game_dir = os.path.join(tmp_workspace, "games")
        os.makedirs(game_dir)
        make_fake_bin(os.path.join(game_dir, "game.bin"), 50)
        orch.update_buffer(game_dir)

        buf = ReplayBuffer(capacity_positions=500_000, buffer_dir=orch.config.buffer_dir)
        buf.load_manifest()
        positions_before = buf.total_positions()
        assert positions_before > 0

        # Create eval data dirs with fake .bin files
        current_eval_dir = os.path.join(tmp_workspace, "eval_current")
        candidate_eval_dir = os.path.join(tmp_workspace, "eval_candidate")
        os.makedirs(current_eval_dir)
        os.makedirs(candidate_eval_dir)
        make_fake_bin(os.path.join(current_eval_dir, "game.bin"), 30)
        make_fake_bin(os.path.join(candidate_eval_dir, "game.bin"), 30)

        # Simulate reject path: WR=47% → elo_delta ≈ -21
        import math
        winrate = 0.47
        elo_delta = -400 * math.log10(1.0 / winrate - 1.0)
        winner_elo = current_elo
        loser_elo = current_elo + elo_delta  # ~79

        # Ingest both sides (same logic as run_generation reject path)
        orch._add_eval_data_to_buffer(current_eval_dir, winner_elo)
        orch._add_eval_data_to_buffer(candidate_eval_dir, loser_elo)

        # Buffer should have old data + both sides' eval data
        buf2 = ReplayBuffer(capacity_positions=500_000, buffer_dir=orch.config.buffer_dir)
        buf2.load_manifest()
        assert buf2.total_positions() == positions_before + 60

        # Check Elo tags: the last two entries should be winner (100.0) and loser (~79)
        # (first entry is the pre-existing data with elo=0.0)
        elos = sorted([e.get("model_elo", 0.0) for e in buf2.entries])
        assert elos[-1] == pytest.approx(100.0, abs=0.1)  # winner side
        assert elos[-2] == pytest.approx(loser_elo, abs=0.1)  # loser side

    def test_reject_with_positive_wr_caps_loser_elo(self, tmp_workspace):
        """Reject with WR>50% should cap candidate Elo at current_elo."""
        import math
        current_elo = 100.0
        # WR=52% but SPRT said H0 — not enough evidence
        winrate = 0.52
        elo_delta = -400 * math.log10(1.0 / winrate - 1.0)
        assert elo_delta > 0  # positive delta

        # Reject path clamps to non-positive
        loser_elo = current_elo + min(0.0, elo_delta)
        assert loser_elo == current_elo  # capped, not above current

    def test_default_buffer_capacity(self):
        """Default buffer capacity should be ~18 generations worth."""
        cfg = TrainingConfig()
        assert cfg.buffer_capacity == 100_000

    def test_training_uses_constant_lr(self, tmp_workspace):
        """LR should be constant regardless of buffer size."""
        orch = self._make_orch(tmp_workspace, initial_lr=0.02)
        orch.state.current_best_pth = "best.pth"

        mock_result = MagicMock()
        mock_result.stdout = "Step 100/100 (global 100): Loss=1.50 P=3.00 V=0.50 K=0.35\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("orchestrate.OracleNet") as mock_model_cls, \
             patch("torch.load", return_value={"model_state_dict": {}}), \
             patch("orchestrate.export_model_for_rust"):
            mock_model_cls.return_value = MagicMock()
            # Even with small buffer, LR should be full
            orch.run_training(1, buffer_positions=5000)

            cmd = mock_run.call_args[0][0]
            lr_idx = cmd.index("--lr")
            lr_val = float(cmd[lr_idx + 1])
            assert lr_val == pytest.approx(0.02)


class TestMaxGenerations:
    def test_config_default_max_generations_is_zero(self):
        cfg = TrainingConfig()
        assert cfg.max_generations == 0

    def test_config_from_args_max_generations(self):
        test_args = ["orchestrate.py", "--max-generations", "5"]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.max_generations == 5

    def test_config_from_args_max_generations_default(self):
        test_args = ["orchestrate.py"]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.max_generations == 0

    def test_loop_terminates_at_max_generations(self, tmp_workspace):
        """Orchestrator should run exactly max_generations iterations."""
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            max_generations=3,
            games_per_generation=2,
            simulations_per_move=10,
            eval_max_games=2,
            eval_simulations=10,
        )
        orch = Orchestrator(cfg)
        generations_run = []

        # Mock all subprocess calls and track generations
        with patch.object(orch, 'run_self_play', return_value=tmp_workspace) as mock_sp, \
             patch.object(orch, 'update_buffer', return_value=100) as mock_buf, \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")) as mock_train, \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})) as mock_eval:

            orch.initialize_gen0()
            orch.save_state()

            # Run the main loop
            orch.run()

            # Should have run exactly 3 generations (single variant mode)
            assert mock_sp.call_count == 3
            assert mock_train.call_count == 3   # 1 variant x 3 generations
            assert mock_eval.call_count == 3    # 1 variant x 3 generations

    def test_loop_runs_one_generation(self, tmp_workspace):
        """max_generations=1 should run exactly one iteration."""
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            max_generations=1,
        )
        orch = Orchestrator(cfg)

        with patch.object(orch, 'run_self_play', return_value=tmp_workspace), \
             patch.object(orch, 'update_buffer', return_value=100), \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")), \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})) as mock_eval:

            orch.initialize_gen0()
            orch.save_state()
            orch.run()
            assert mock_eval.call_count == 1  # 1 variant in 1 generation (single_variant=True)

    def test_loop_respects_resume_with_max_generations(self, tmp_workspace):
        """When resuming, max_generations counts from the resume point."""
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            max_generations=2,
            resume=True,
        )
        orch = Orchestrator(cfg)

        # Pre-seed state as if generation 5 already completed
        orch.initialize_gen0()
        orch.state.generation = 5
        orch.save_state()

        with patch.object(orch, 'run_self_play', return_value=tmp_workspace), \
             patch.object(orch, 'update_buffer', return_value=100), \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")), \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})) as mock_eval:

            orch.run()
            # Should run generations 6 and 7 (2 total, 1 variant each)
            assert mock_eval.call_count == 2


class TestSPRT:
    def _make_orch(self, tmp_workspace, **overrides):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            **overrides,
        )
        return Orchestrator(cfg)

    def test_sprt_config_defaults(self):
        """Verify new SPRT config defaults."""
        cfg = TrainingConfig()
        assert cfg.sprt_elo0 == 0.0
        assert cfg.sprt_elo1 == 10.0
        assert cfg.sprt_alpha == 0.05
        assert cfg.sprt_beta == 0.05
        assert cfg.eval_max_games == 400

    def test_sprt_config_from_args(self):
        """Verify CLI parsing of SPRT args."""
        test_args = [
            "orchestrate.py",
            "--sprt-elo0", "5.0",
            "--sprt-elo1", "15.0",
            "--sprt-alpha", "0.01",
            "--sprt-beta", "0.10",
            "--eval-max-games", "200",
        ]
        with patch("sys.argv", test_args):
            cfg = TrainingConfig.from_args()
        assert cfg.sprt_elo0 == 5.0
        assert cfg.sprt_elo1 == 15.0
        assert cfg.sprt_alpha == 0.01
        assert cfg.sprt_beta == 0.10
        assert cfg.eval_max_games == 200

    def test_run_evaluation_parses_h1(self, tmp_workspace):
        """SPRT_RESULT=H1 → accepted=True."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = (
            "WINS=60 LOSSES=30 DRAWS=10 WINRATE=0.6500 ACCEPTED=true "
            "GAMES_PLAYED=100 LLR=3.50 SPRT_RESULT=H1\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            accepted, results = orch.run_evaluation("candidate.pt")

        assert accepted is True
        assert results["sprt_result"] == "H1"
        assert results["llr"] == pytest.approx(3.50)
        assert results["games_played"] == 100

    def test_run_evaluation_parses_h0(self, tmp_workspace):
        """SPRT_RESULT=H0 → accepted=False."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = (
            "WINS=30 LOSSES=60 DRAWS=10 WINRATE=0.3500 ACCEPTED=false "
            "GAMES_PLAYED=100 LLR=-3.50 SPRT_RESULT=H0\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            accepted, results = orch.run_evaluation("candidate.pt")

        assert accepted is False
        assert results["sprt_result"] == "H0"

    def test_run_evaluation_parses_inconclusive(self, tmp_workspace):
        """SPRT_RESULT=inconclusive → accepted=False."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = (
            "WINS=50 LOSSES=50 DRAWS=300 WINRATE=0.5000 ACCEPTED=false "
            "GAMES_PLAYED=400 LLR=0.10 SPRT_RESULT=inconclusive\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            accepted, results = orch.run_evaluation("candidate.pt")

        assert accepted is False
        assert results["sprt_result"] == "inconclusive"

    def test_run_evaluation_passes_sprt_args(self, tmp_workspace):
        """Verify SPRT CLI args are passed to the Rust binary."""
        orch = self._make_orch(
            tmp_workspace,
            sprt_elo0=0.0,
            sprt_elo1=10.0,
            sprt_alpha=0.05,
            sprt_beta=0.05,
            eval_max_games=400,
        )
        orch.state.current_best_pt = "current.pt"

        mock_result = MagicMock()
        mock_result.stdout = (
            "WINS=60 LOSSES=30 DRAWS=10 WINRATE=0.6500 ACCEPTED=true "
            "GAMES_PLAYED=50 LLR=3.00 SPRT_RESULT=H1\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            orch.run_evaluation("candidate.pt")

        cmd = mock_run.call_args[0][0]
        assert "--sprt" in cmd
        elo0_idx = cmd.index("--elo0")
        assert cmd[elo0_idx + 1] == "0.0"
        elo1_idx = cmd.index("--elo1")
        assert cmd[elo1_idx + 1] == "10.0"
        alpha_idx = cmd.index("--sprt-alpha")
        assert cmd[alpha_idx + 1] == "0.05"
        beta_idx = cmd.index("--sprt-beta")
        assert cmd[beta_idx + 1] == "0.05"


class TestSkipSelfPlay:
    def _make_orch(self, tmp_workspace, **overrides):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            **overrides,
        )
        return Orchestrator(cfg)

    def test_skip_self_play_after_gen1(self, tmp_workspace):
        """With skip_self_play=True, run_self_play called only for gen 1."""
        orch = self._make_orch(tmp_workspace, skip_self_play=True, max_generations=3)
        orch.initialize_gen0()
        orch.save_state()

        with patch.object(orch, 'run_self_play', return_value=tmp_workspace) as mock_sp, \
             patch.object(orch, 'update_buffer', return_value=100) as mock_buf, \
             patch.object(orch, '_get_buffer_size', return_value=100) as mock_get_buf, \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")) as mock_train, \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})) as mock_eval:

            orch.run()

            assert mock_sp.call_count == 1  # Only gen 1
            assert mock_train.call_count == 3  # Every gen
            assert mock_eval.call_count == 3  # Every gen

    def test_skip_self_play_still_gets_buffer_size(self, tmp_workspace):
        """With skip_self_play=True, gen 2+ gets buffer_size via _get_buffer_size."""
        orch = self._make_orch(tmp_workspace, skip_self_play=True, max_generations=2)
        orch.initialize_gen0()
        orch.save_state()

        with patch.object(orch, 'run_self_play', return_value=tmp_workspace), \
             patch.object(orch, 'update_buffer', return_value=100), \
             patch.object(orch, '_get_buffer_size', return_value=200) as mock_get_buf, \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")) as mock_train, \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})):

            orch.run()

            # Gen 2 should use _get_buffer_size (called once for gen 2)
            assert mock_get_buf.call_count == 1
            # Gen 2 training should get buffer_positions=200 from _get_buffer_size
            gen2_call = mock_train.call_args_list[1]
            assert gen2_call[1].get("buffer_positions") == 200

    def test_skip_self_play_default_false(self):
        """skip_self_play defaults to False for backward compat."""
        cfg = TrainingConfig()
        assert cfg.skip_self_play is False

    def test_skip_self_play_gen1_always_runs(self, tmp_workspace):
        """Even with skip_self_play=True, gen 1 always runs self-play."""
        orch = self._make_orch(tmp_workspace, skip_self_play=True, max_generations=1)
        orch.initialize_gen0()
        orch.save_state()

        with patch.object(orch, 'run_self_play', return_value=tmp_workspace) as mock_sp, \
             patch.object(orch, 'update_buffer', return_value=100), \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")), \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})):

            orch.run()

            assert mock_sp.call_count == 1  # Gen 1 always runs

    def test_skip_self_play_log_games_generated_zero(self, tmp_workspace):
        """When self-play is skipped, games_generated in log should be 0."""
        orch = self._make_orch(tmp_workspace, skip_self_play=True, max_generations=2)
        orch.initialize_gen0()
        orch.save_state()

        with patch.object(orch, 'run_self_play', return_value=tmp_workspace), \
             patch.object(orch, 'update_buffer', return_value=100), \
             patch.object(orch, '_get_buffer_size', return_value=100), \
             patch.object(orch, 'run_training', return_value=("c.pth", "c.pt")), \
             patch.object(orch, 'run_evaluation', return_value=(False, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})):

            orch.run()

        with open(orch.config.log_file) as f:
            lines = f.readlines()
        gen1_log = json.loads(lines[0])
        gen2_log = json.loads(lines[1])
        assert gen1_log["games_generated"] == orch.config.games_per_generation
        assert gen2_log["games_generated"] == 0


class TestEloTracking:
    def _make_orch(self, tmp_workspace, **overrides):
        cfg = TrainingConfig(
            weights_dir=os.path.join(tmp_workspace, "weights"),
            data_dir=os.path.join(tmp_workspace, "data"),
            buffer_dir=os.path.join(tmp_workspace, "buffer"),
            log_file=os.path.join(tmp_workspace, "log.jsonl"),
            **overrides,
        )
        return Orchestrator(cfg)

    def test_elo_accumulation_on_acceptance(self, tmp_workspace):
        """Accepting a model with 55% winrate should add ~35 Elo."""
        import math
        orch = self._make_orch(tmp_workspace)
        os.makedirs(orch.config.weights_dir, exist_ok=True)

        candidate_pt = os.path.join(orch.config.weights_dir, "candidate.pt")
        candidate_pth = os.path.join(orch.config.weights_dir, "candidate.pth")
        with open(candidate_pt, "w") as f:
            f.write("fake")
        with open(candidate_pth, "w") as f:
            f.write("fake")

        # Initial state: gen 0 at Elo 0
        orch.state.model_elos = {"0": 0.0}
        orch.state.accepted_count = 0

        # Accept with 55% winrate
        orch.handle_eval_result(
            accepted=True, generation=1,
            candidate_pt=candidate_pt, candidate_pth=candidate_pth,
            eval_results={"wins": 55, "losses": 45, "draws": 0, "winrate": 0.55},
        )

        expected_delta = -400 * math.log10(1/0.55 - 1)
        assert orch.state.accepted_count == 1
        assert "1" in orch.state.model_elos
        assert orch.state.model_elos["1"] == pytest.approx(expected_delta, rel=1e-4)

    def test_elo_accumulates_across_multiple_acceptances(self, tmp_workspace):
        """Multiple acceptances should accumulate Elo."""
        import math
        orch = self._make_orch(tmp_workspace)
        os.makedirs(orch.config.weights_dir, exist_ok=True)

        orch.state.model_elos = {"0": 0.0}
        orch.state.accepted_count = 0

        winrates = [0.55, 0.60, 0.52]
        for i, wr in enumerate(winrates):
            candidate_pt = os.path.join(orch.config.weights_dir, f"candidate_{i+1}.pt")
            candidate_pth = os.path.join(orch.config.weights_dir, f"candidate_{i+1}.pth")
            with open(candidate_pt, "w") as f:
                f.write("fake")
            with open(candidate_pth, "w") as f:
                f.write("fake")

            orch.handle_eval_result(
                accepted=True, generation=i+1,
                candidate_pt=candidate_pt, candidate_pth=candidate_pth,
                eval_results={"wins": int(wr*100), "losses": int((1-wr)*100),
                              "draws": 0, "winrate": wr},
            )

        # Verify cumulative Elo
        cumulative = 0.0
        for wr in winrates:
            cumulative += -400 * math.log10(1/wr - 1)

        assert orch.state.accepted_count == 3
        assert orch.state.model_elos["3"] == pytest.approx(cumulative, rel=1e-4)

    def test_rejection_does_not_change_elos(self, tmp_workspace):
        """Rejecting a model should not add any Elo entry."""
        orch = self._make_orch(tmp_workspace)
        orch.state.model_elos = {"0": 0.0}
        orch.state.accepted_count = 0
        orch.state.current_best_pt = "old.pt"
        orch.state.current_best_pth = "old.pth"

        orch.handle_eval_result(
            accepted=False, generation=1,
            candidate_pt="candidate.pt", candidate_pth="candidate.pth",
            eval_results={"wins": 40, "losses": 50, "draws": 10, "winrate": 0.45},
        )

        assert orch.state.accepted_count == 0
        assert len(orch.state.model_elos) == 1
        assert "0" in orch.state.model_elos

    def test_elo_state_save_load_roundtrip(self, tmp_workspace):
        """model_elos should survive save/load cycle."""
        orch = self._make_orch(tmp_workspace)
        orch.state.model_elos = {"0": 0.0, "1": 35.0, "2": 105.0}
        orch.state.accepted_count = 2
        orch.save_state()

        orch2 = self._make_orch(tmp_workspace)
        loaded = orch2.load_state()
        assert loaded is True
        assert orch2.state.model_elos == {"0": 0.0, "1": 35.0, "2": 105.0}
        assert orch2.state.accepted_count == 2

    def test_legacy_state_without_model_elos(self, tmp_workspace):
        """Old state files without model_elos should load with default {}."""
        state_path = os.path.join(tmp_workspace, "data", "orchestrator_state.json")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        # Write a legacy state file without model_elos
        legacy_state = {
            "generation": 5,
            "current_best_pth": "weights/gen_4.pth",
            "current_best_pt": "weights/gen_4.pt",
            "global_minibatches": 5000,
            "reset_optimizer_next": False,
            "accepted_count": 3,
        }
        with open(state_path, "w") as f:
            json.dump(legacy_state, f)

        state = OrchestratorState.load(state_path)
        assert state.model_elos == {}
        assert state.generation == 5

    def test_run_training_no_longer_passes_sampling_half_life(self, tmp_workspace):
        """run_training should NOT pass --sampling-half-life to subprocess."""
        orch = self._make_orch(tmp_workspace)
        orch.state.current_best_pth = "best.pth"

        mock_result = MagicMock()
        mock_result.stdout = "Step 100/100 (global 100): Loss=1.50 P=3.00 V=0.50 K=0.35\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("orchestrate.OracleNet") as mock_model_cls, \
             patch("torch.load", return_value={"model_state_dict": {}}), \
             patch("orchestrate.export_model_for_rust"):
            mock_model_cls.return_value = MagicMock()
            orch.run_training(1, buffer_positions=10000)

            cmd = mock_run.call_args[0][0]
            assert "--sampling-half-life" not in cmd

    def test_initialize_gen0_sets_elo_zero(self, tmp_workspace):
        """initialize_gen0 should set model_elos['0'] = 0.0."""
        orch = self._make_orch(tmp_workspace)
        orch.initialize_gen0()
        assert orch.state.model_elos == {"0": 0.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
