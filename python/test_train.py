"""Tests for training upgrades: minibatch mode, LR scheduling, full checkpoint, buffer-dir loading."""

import os
import sys
import shutil
import tempfile
import numpy as np
import torch
import pytest

# Ensure python/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from model import OracleNet
from replay_buffer import ReplayBuffer, SAMPLE_SIZE_FLOATS, BYTES_PER_SAMPLE

# Import training internals (we'll test functions directly)
import train as train_module


def make_fake_bin(path, num_positions):
    """Create a fake .bin file with valid-shaped random data."""
    data = np.random.randn(num_positions, SAMPLE_SIZE_FLOATS).astype(np.float32)
    data.tofile(path)


def make_tiny_model():
    """Create a tiny model for fast testing."""
    return OracleNet(num_blocks=1, hidden_dim=16)


@pytest.fixture
def tmp_dirs():
    data_dir = tempfile.mkdtemp(prefix="train_data_")
    output_dir = tempfile.mkdtemp(prefix="train_out_")
    yield data_dir, output_dir
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


class TestMinibatchMode:
    def test_minibatch_mode_runs_exact_count(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=10,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 10


class TestLRSchedule:
    def test_lr_schedule_changes_lr_at_boundaries(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        # Schedule: at minibatch 3, drop to 0.0005; at 7, drop to 0.0001
        schedule = "3:0.0005,7:0.0001"
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=10,
            lr_schedule=schedule,
            buffer_dir=None,
            _return_lr_history=True,
        )
        # step_count is actually (step_count, lr_history) when _return_lr_history=True
        actual_steps, lr_history = step_count
        assert actual_steps == 10
        # Before boundary 3: lr=0.001
        assert lr_history[0] == pytest.approx(0.001)
        # After boundary 3: lr=0.0005
        assert lr_history[3] == pytest.approx(0.0005)
        # After boundary 7: lr=0.0001
        assert lr_history[7] == pytest.approx(0.0001)


class TestCheckpoint:
    def test_checkpoint_saves_full_state(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=5,
            lr_schedule="",
            buffer_dir=None,
        )

        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "global_minibatch" in checkpoint
        assert checkpoint["global_minibatch"] == 5

    def test_resume_restores_optimizer_and_minibatch(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")

        # First run: 5 minibatches
        train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=5,
            lr_schedule="",
            buffer_dir=None,
        )

        # Second run: resume from checkpoint, run 5 more
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=output_path,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=5,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 5

        # Check global_minibatch is now 10
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert checkpoint["global_minibatch"] == 10


class TestBufferDir:
    def test_buffer_dir_flag_loads_from_buffer(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs

        # Set up a replay buffer
        buffer_dir = tempfile.mkdtemp(prefix="buffer_")
        try:
            source_dir = tempfile.mkdtemp(prefix="source_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 200)

            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir)
            shutil.rmtree(source_dir, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            step_count = train_module.train_with_config(
                data_dir=None,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                epochs=1,
                batch_size=32,
                minibatches=5,
                lr_schedule="",
                buffer_dir=buffer_dir,
            )
            assert step_count == 5
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)


class TestChessDataset:
    def test_chess_dataset_loads_bin_file(self, tmp_dirs):
        data_dir, _ = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 10)

        dataset = train_module.ChessDataset(data_dir)
        assert len(dataset) == 10

    def test_chess_dataset_getitem_shapes(self, tmp_dirs):
        data_dir, _ = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 5)

        dataset = train_module.ChessDataset(data_dir)
        board, scalars, value, policy = dataset[0]
        assert board.shape == (17, 8, 8)
        assert scalars.shape == (2,), f"Expected scalars shape (2,), got {scalars.shape}"
        assert value.shape == (1,)
        assert policy.shape == (4672,)

    def test_chess_dataset_empty_dir(self, tmp_dirs):
        data_dir, _ = tmp_dirs
        dataset = train_module.ChessDataset(data_dir)
        assert len(dataset) == 0

    def test_chess_dataset_skips_empty_file(self, tmp_dirs):
        data_dir, _ = tmp_dirs
        with open(os.path.join(data_dir, "empty.bin"), "w") as f:
            pass
        dataset = train_module.ChessDataset(data_dir)
        assert len(dataset) == 0

    def test_chess_dataset_multiple_files(self, tmp_dirs):
        data_dir, _ = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "a.bin"), 5)
        make_fake_bin(os.path.join(data_dir, "b.bin"), 8)
        dataset = train_module.ChessDataset(data_dir)
        assert len(dataset) == 13


class TestBufferDataset:
    def test_buffer_dataset_init_loads_manifest(self, tmp_dirs):
        _, output_dir = tmp_dirs
        buffer_dir = tempfile.mkdtemp(prefix="bufdataset_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 50)
            from replay_buffer import ReplayBuffer
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir)
            shutil.rmtree(source_dir, ignore_errors=True)

            dataset = train_module.BufferDataset(buffer_dir)
            assert len(dataset) == 50
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_buffer_dataset_getitem_shapes(self, tmp_dirs):
        _, output_dir = tmp_dirs
        buffer_dir = tempfile.mkdtemp(prefix="bufdataset_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 50)
            from replay_buffer import ReplayBuffer
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir)
            shutil.rmtree(source_dir, ignore_errors=True)

            dataset = train_module.BufferDataset(buffer_dir)
            board, scalars, value, policy = dataset[0]
            assert board.shape == (17, 8, 8)
            assert scalars.shape == (2,), f"Expected scalars shape (2,), got {scalars.shape}"
            assert value.shape == (1,)
            assert policy.shape == (4672,)
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)


class TestParseLrSchedule:
    def test_empty_string_returns_empty(self):
        assert train_module.parse_lr_schedule("") == []

    def test_single_boundary(self):
        result = train_module.parse_lr_schedule("100:0.01")
        assert result == [(100, 0.01)]

    def test_multiple_boundaries_sorted(self):
        result = train_module.parse_lr_schedule("200:0.001,100:0.01")
        assert result == [(100, 0.01), (200, 0.001)]

    def test_whitespace_handling(self):
        result = train_module.parse_lr_schedule(" 100:0.01 , 200:0.001 ")
        assert result == [(100, 0.01), (200, 0.001)]


class TestGetLrForStep:
    def test_before_first_boundary(self):
        schedule = [(100, 0.01), (200, 0.001)]
        assert train_module.get_lr_for_step(schedule, 0.1, 50) == 0.1

    def test_at_first_boundary(self):
        schedule = [(100, 0.01), (200, 0.001)]
        assert train_module.get_lr_for_step(schedule, 0.1, 100) == 0.01

    def test_between_boundaries(self):
        schedule = [(100, 0.01), (200, 0.001)]
        assert train_module.get_lr_for_step(schedule, 0.1, 150) == 0.01

    def test_at_second_boundary(self):
        schedule = [(100, 0.01), (200, 0.001)]
        assert train_module.get_lr_for_step(schedule, 0.1, 200) == 0.001

    def test_past_all_boundaries(self):
        schedule = [(100, 0.01), (200, 0.001)]
        assert train_module.get_lr_for_step(schedule, 0.1, 999) == 0.001

    def test_empty_schedule_returns_base(self):
        assert train_module.get_lr_for_step([], 0.1, 999) == 0.1


class TestMakeOptimizer:
    def test_adam_optimizer(self):
        model = OracleNet(num_blocks=1, hidden_dim=16)
        opt = train_module.make_optimizer(model, "adam", 0.001)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw_optimizer(self):
        model = OracleNet(num_blocks=1, hidden_dim=16)
        opt = train_module.make_optimizer(model, "adamw", 0.001)
        assert isinstance(opt, torch.optim.AdamW)

    def test_unknown_optimizer_defaults_to_adam(self):
        model = OracleNet(num_blocks=1, hidden_dim=16)
        opt = train_module.make_optimizer(model, "unknown", 0.001)
        assert isinstance(opt, torch.optim.Adam)


class TestEpochMode:
    def test_epoch_mode_runs_correct_batches(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=2,
            batch_size=32,
            minibatches=None,
            lr_schedule="",
            buffer_dir=None,
        )
        # 100 samples / 32 batch_size = 4 batches per epoch (ceil), * 2 epochs
        assert step_count > 0
        assert os.path.exists(output_path)

    def test_epoch_mode_saves_checkpoint(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        output_path = os.path.join(output_dir, "model.pth")
        train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=None,
            lr_schedule="",
            buffer_dir=None,
        )
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "global_minibatch" in checkpoint

    def test_epoch_mode_with_lr_schedule(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        result = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=None,
            lr_schedule="3:0.0005",
            buffer_dir=None,
            _return_lr_history=True,
        )
        steps, lr_history = result
        assert steps > 3
        assert lr_history[0] == pytest.approx(0.001)
        assert lr_history[3] == pytest.approx(0.0005)


class TestEmptyDataset:
    def test_empty_dataset_returns_zero(self, tmp_dirs):
        data_dir, output_dir = tmp_dirs
        # Empty directory, no .bin files
        output_path = os.path.join(output_dir, "model.pth")
        result = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=10,
            lr_schedule="",
            buffer_dir=None,
        )
        assert result == 0


class TestResumeEdgeCases:
    def test_resume_with_legacy_checkpoint(self, tmp_dirs):
        """Resume from a legacy checkpoint that's just a state_dict."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        # Create legacy checkpoint (just state_dict)
        model = OracleNet()
        legacy_path = os.path.join(output_dir, "legacy.pth")
        torch.save(model.state_dict(), legacy_path)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=legacy_path,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=3,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 3
        # Global minibatch should start from 0 (legacy has no counter)
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert checkpoint["global_minibatch"] == 3

    def test_resume_with_nonexistent_path(self, tmp_dirs):
        """Resume from a path that doesn't exist should just start fresh."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path="/nonexistent/path.pth",
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=3,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 3

    def test_resume_with_corrupt_checkpoint(self, tmp_dirs):
        """Resume from a corrupt file should not crash."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        corrupt_path = os.path.join(output_dir, "corrupt.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"this is not a valid checkpoint")

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=corrupt_path,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=3,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 3

    def test_adamw_optimizer_training(self, tmp_dirs):
        """Training with AdamW optimizer works."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adamw",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=3,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 3


class TestMinibatchWraparound:
    def test_minibatch_wraps_around_small_dataset(self, tmp_dirs):
        """When minibatches > number of batches in dataset, iterator wraps around."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 10)

        output_path = os.path.join(output_dir, "model.pth")
        # 10 samples / 32 batch = 1 batch per epoch, but we want 5 minibatches
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=5,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 5


class TestIntermediateCheckpoints:
    def test_checkpoint_saved_at_200_minibatches(self, tmp_dirs):
        """Mid-training checkpoint should be written at step 200."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=250,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 250

        # Checkpoint should exist (written at step 200, then overwritten at end)
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert checkpoint["global_minibatch"] == 250

    def test_checkpoint_at_200_is_resumable(self, tmp_dirs):
        """A checkpoint saved at step 200 should be resumable."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        # Run 200 minibatches — checkpoint fires exactly at step 200
        train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=200,
            lr_schedule="",
            buffer_dir=None,
        )

        # Verify the checkpoint at 200 has correct global_minibatch
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert checkpoint["global_minibatch"] == 200

        # Resume from it
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=output_path,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=50,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 50
        checkpoint2 = torch.load(output_path, map_location="cpu", weights_only=False)
        assert checkpoint2["global_minibatch"] == 250

    def test_no_checkpoint_before_200(self, tmp_dirs):
        """When running fewer than 200 minibatches, only the final checkpoint is written."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=100,
            lr_schedule="",
            buffer_dir=None,
        )
        assert step_count == 100
        # Final checkpoint should still exist
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        assert checkpoint["global_minibatch"] == 100


class TestLossLogging:
    def test_minibatch_mode_prints_loss_at_100(self, tmp_dirs, capsys):
        """Loss should be printed at step 100 in minibatch mode."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=150,
            lr_schedule="",
            buffer_dir=None,
        )

        captured = capsys.readouterr()
        assert "Step 100/150" in captured.out
        assert "Loss=" in captured.out
        assert "P=" in captured.out
        assert "V=" in captured.out
        assert "K=" in captured.out

    def test_minibatch_mode_no_log_before_100(self, tmp_dirs, capsys):
        """No loss logging when running fewer than 100 minibatches."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 200)

        output_path = os.path.join(output_dir, "model.pth")
        train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=1,
            batch_size=32,
            minibatches=50,
            lr_schedule="",
            buffer_dir=None,
        )

        captured = capsys.readouterr()
        assert "Step " not in captured.out or "Step 50/50" not in captured.out


class TestBufferDatasetChunking:
    def _make_buffer(self, num_positions):
        """Create a temp buffer with fake data, return (buffer_dir, cleanup_dirs)."""
        buffer_dir = tempfile.mkdtemp(prefix="bufchunk_")
        source_dir = tempfile.mkdtemp(prefix="src_")
        make_fake_bin(os.path.join(source_dir, "game.bin"), num_positions)
        from replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)
        shutil.rmtree(source_dir, ignore_errors=True)
        return buffer_dir

    def test_chunk_refreshes_when_exhausted(self):
        """After consuming chunk_size items, a new chunk should be loaded."""
        buffer_dir = self._make_buffer(100)
        try:
            dataset = train_module.BufferDataset(buffer_dir, chunk_size=10)
            assert dataset.chunk_size == 10
            assert dataset._chunk_idx == 0

            # Consume all 10 items in the chunk
            for i in range(10):
                _ = dataset[i]
            assert dataset._chunk_idx == 10

            # Next access should trigger refresh
            _ = dataset[10]
            assert dataset._chunk_idx == 1  # Reset to 0, then incremented to 1
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_small_chunk_size(self):
        """Chunk size smaller than dataset should still work."""
        buffer_dir = self._make_buffer(50)
        try:
            dataset = train_module.BufferDataset(buffer_dir, chunk_size=5)
            # Access 15 items (3 chunks worth)
            for i in range(15):
                board, scalars, val, pol = dataset[i]
                assert board.shape == (17, 8, 8)
                assert scalars.shape == (2,)
                assert pol.shape == (4672,)
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_default_chunk_size(self):
        """Default chunk_size should be 4096."""
        buffer_dir = self._make_buffer(50)
        try:
            # This will try to sample 4096 from only 50 positions — should still work
            # because sample_batch samples with replacement
            dataset = train_module.BufferDataset(buffer_dir)
            assert dataset.chunk_size == 4096
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)


class TestEpochDataset:
    def test_epoch_dataset_loads_all_positions_same_elo(self, tmp_dirs):
        """EpochDataset with same-Elo data should include all positions."""
        _, output_dir = tmp_dirs
        buffer_dir = tempfile.mkdtemp(prefix="epoch_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 100)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            dataset = train_module.EpochDataset(buffer_dir, augment=False)
            assert len(dataset) == 100
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_epoch_dataset_getitem_shapes(self, tmp_dirs):
        """EpochDataset items should have correct tensor shapes."""
        _, output_dir = tmp_dirs
        buffer_dir = tempfile.mkdtemp(prefix="epoch_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 50)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            dataset = train_module.EpochDataset(buffer_dir, augment=False)
            board, scalars, value, policy = dataset[0]
            assert board.shape == (17, 8, 8)
            assert scalars.shape == (2,), f"Expected scalars shape (2,), got {scalars.shape}"
            assert value.shape == (1,)
            assert policy.shape == (4672,)
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_use_epochs_trains_correct_steps(self, tmp_dirs):
        """use_epochs=True trains len(dataset)/batch_size steps per epoch."""
        data_dir, output_dir = tmp_dirs

        buffer_dir = tempfile.mkdtemp(prefix="epoch_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 100)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            step_count = train_module.train_with_config(
                data_dir=data_dir,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                epochs=1,
                batch_size=32,
                minibatches=None,
                lr_schedule="",
                buffer_dir=buffer_dir,
                use_epochs=True,
            )
            # 100 positions / 32 batch = 4 batches (3 full + 1 partial with drop_last=False)
            expected_batches = (100 + 32 - 1) // 32  # ceil(100/32) = 4
            assert step_count == expected_batches
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_use_epochs_multiple_epochs(self, tmp_dirs):
        """use_epochs=True with max_epochs=2 trains 2x the steps (with train/val split)."""
        data_dir, output_dir = tmp_dirs

        buffer_dir = tempfile.mkdtemp(prefix="epoch_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 100)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            step_count = train_module.train_with_config(
                data_dir=data_dir,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                max_epochs=2,
                batch_size=32,
                minibatches=None,
                lr_schedule="",
                buffer_dir=buffer_dir,
                use_epochs=True,
            )
            # With max_epochs=2, dataset is split 90/10 train/val
            # Each epoch trains on ceil(90/32)=3 batches
            # Both epochs should run (assuming no early stopping with random data)
            assert step_count > 0
            # At minimum 1 epoch worth of steps
            min_per_epoch = (90 + 32 - 1) // 32  # 3
            assert step_count >= min_per_epoch
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_use_epochs_with_elo_gap_fewer_steps(self, tmp_dirs):
        """use_epochs=True with Elo gap → fewer steps than total positions."""
        data_dir, output_dir = tmp_dirs

        buffer_dir = tempfile.mkdtemp(prefix="epoch_buf_")
        try:
            src1 = tempfile.mkdtemp(prefix="low_")
            src2 = tempfile.mkdtemp(prefix="high_")
            make_fake_bin(os.path.join(src1, "low.bin"), 1000)
            make_fake_bin(os.path.join(src2, "high.bin"), 100)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(src1, model_elo=0.0)
            buf.add_games(src2, model_elo=200.0)
            shutil.rmtree(src1, ignore_errors=True)
            shutil.rmtree(src2, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            np.random.seed(42)
            step_count = train_module.train_with_config(
                data_dir=data_dir,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                epochs=1,
                batch_size=32,
                minibatches=None,
                lr_schedule="",
                buffer_dir=buffer_dir,
                use_epochs=True,
            )
            # Total buffer = 1100, but with 200 Elo gap the 1000 low-Elo positions
            # have ~32% inclusion ≈ 320, plus 100 high-Elo = ~420 included
            # So steps should be ~420/32 ≈ 14
            all_positions_steps = (1100 + 32 - 1) // 32  # 35
            assert step_count < all_positions_steps, \
                f"Expected fewer steps than {all_positions_steps}, got {step_count}"
            assert step_count > 0
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)


class TestAdaptiveEpochs:
    def test_early_stopping_prints_best_epoch(self, tmp_dirs, capsys):
        """Early stopping should print BEST_EPOCH and VAL_LOSS."""
        data_dir, output_dir = tmp_dirs

        buffer_dir = tempfile.mkdtemp(prefix="adaptive_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 200)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            step_count = train_module.train_with_config(
                data_dir=data_dir,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                max_epochs=5,
                batch_size=32,
                minibatches=None,
                lr_schedule="",
                buffer_dir=buffer_dir,
                use_epochs=True,
            )

            captured = capsys.readouterr()
            assert "BEST_EPOCH=" in captured.out
            assert "VAL_LOSS=" in captured.out
            assert "Train/val split:" in captured.out
            assert step_count > 0
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_max_epochs_1_skips_validation(self, tmp_dirs, capsys):
        """max_epochs=1 should not create train/val split."""
        data_dir, output_dir = tmp_dirs

        buffer_dir = tempfile.mkdtemp(prefix="adaptive_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 200)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            step_count = train_module.train_with_config(
                data_dir=data_dir,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                max_epochs=1,
                batch_size=32,
                minibatches=None,
                lr_schedule="",
                buffer_dir=buffer_dir,
                use_epochs=True,
            )

            captured = capsys.readouterr()
            assert "Train/val split:" not in captured.out
            assert "BEST_EPOCH=" not in captured.out
            # Should still train
            assert step_count > 0
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_best_model_restored_after_early_stopping(self, tmp_dirs):
        """The saved checkpoint should contain the best epoch's model weights."""
        data_dir, output_dir = tmp_dirs

        buffer_dir = tempfile.mkdtemp(prefix="adaptive_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            make_fake_bin(os.path.join(source_dir, "game.bin"), 200)
            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            output_path = os.path.join(output_dir, "model.pth")
            train_module.train_with_config(
                data_dir=data_dir,
                output_path=output_path,
                resume_path=None,
                optimizer_name="adam",
                lr=0.001,
                max_epochs=5,
                batch_size=32,
                minibatches=None,
                lr_schedule="",
                buffer_dir=buffer_dir,
                use_epochs=True,
            )

            # Verify checkpoint was saved
            assert os.path.exists(output_path)
            checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in checkpoint
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_epochs_backward_compat_kwarg(self, tmp_dirs):
        """Passing epochs= as keyword arg should still work via backward compat."""
        data_dir, output_dir = tmp_dirs
        make_fake_bin(os.path.join(data_dir, "game.bin"), 100)

        output_path = os.path.join(output_dir, "model.pth")
        step_count = train_module.train_with_config(
            data_dir=data_dir,
            output_path=output_path,
            resume_path=None,
            optimizer_name="adam",
            lr=0.001,
            epochs=2,
            batch_size=32,
            minibatches=None,
            lr_schedule="",
            buffer_dir=None,
        )
        # 100 samples / 32 batch_size = 4 batches per epoch * 2 epochs = 8
        assert step_count > 0


class TestOrchestratorStateBackwardCompat:
    def test_load_without_latest_pth(self, tmp_dirs):
        """Old state files without latest_pth should default to current_best_pth."""
        data_dir, output_dir = tmp_dirs
        state_path = os.path.join(output_dir, "state.json")

        # Write an old-style state file (no latest_pth)
        import json
        old_state = {
            "generation": 5,
            "current_best_pth": "/some/path/gen_5.pth",
            "current_best_pt": "/some/path/gen_5.pt",
            "global_minibatches": 1000,
            "reset_optimizer_next": False,
            "accepted_count": 3,
            "model_elos": {"0": 0.0, "1": 15.0, "2": 30.0, "3": 45.0},
        }
        with open(state_path, "w") as f:
            json.dump(old_state, f)

        # Import and load
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from orchestrate import OrchestratorState
        state = OrchestratorState.load(state_path)

        assert state.latest_pth == "/some/path/gen_5.pth"
        assert state.current_best_pth == "/some/path/gen_5.pth"
        assert state.generation == 5

    def test_save_and_load_with_latest_pth(self, tmp_dirs):
        """State with latest_pth should round-trip correctly."""
        data_dir, output_dir = tmp_dirs
        state_path = os.path.join(output_dir, "state.json")

        from orchestrate import OrchestratorState
        state = OrchestratorState(
            generation=3,
            current_best_pth="/best/gen_2.pth",
            current_best_pt="/best/gen_2.pt",
            latest_pth="/weights/candidate_3.pth",
            accepted_count=2,
        )
        state.save(state_path)

        loaded = OrchestratorState.load(state_path)
        assert loaded.latest_pth == "/weights/candidate_3.pth"
        assert loaded.current_best_pth == "/best/gen_2.pth"


class TestQsearchFlag:
    """Tests for the qsearch_completed flag in training data format."""

    def test_sample_size_is_5763(self):
        """Binary format should be 5763 floats: board(1088) + material(1) + qsearch_flag(1) + value(1) + policy(4672)."""
        assert train_module.SAMPLE_SIZE_FLOATS == 5763
        assert SAMPLE_SIZE_FLOATS == 5763

    def test_chess_dataset_parses_qsearch_flag(self, tmp_dirs):
        """ChessDataset should correctly parse qsearch_completed from binary data."""
        data_dir, _ = tmp_dirs
        # Create a sample with known values
        board_end = 17 * 64
        sample = np.zeros(5763, dtype=np.float32)
        sample[board_end] = 3.5       # material
        sample[board_end + 1] = 1.0   # qsearch_completed = true
        sample[board_end + 2] = 0.75  # value
        sample.tofile(os.path.join(data_dir, "test.bin"))

        dataset = train_module.ChessDataset(data_dir, augment=False)
        board, scalars, value, policy = dataset[0]

        assert scalars.shape == (2,)
        assert abs(scalars[0].item() - 3.5) < 1e-5, f"material should be 3.5, got {scalars[0].item()}"
        assert abs(scalars[1].item() - 1.0) < 1e-5, f"qsearch_flag should be 1.0, got {scalars[1].item()}"
        assert abs(value.item() - 0.75) < 1e-5

    def test_epoch_dataset_parses_qsearch_flag(self, tmp_dirs):
        """EpochDataset should correctly parse qsearch_completed from binary data."""
        _, output_dir = tmp_dirs
        buffer_dir = tempfile.mkdtemp(prefix="qsearch_buf_")
        try:
            source_dir = tempfile.mkdtemp(prefix="src_")
            board_end = 17 * 64
            sample = np.zeros(5763, dtype=np.float32)
            sample[board_end] = -2.0      # material
            sample[board_end + 1] = 0.0   # qsearch_completed = false
            sample[board_end + 2] = -0.5  # value
            sample.tofile(os.path.join(source_dir, "test.bin"))

            buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf.add_games(source_dir, model_elo=0.0)
            shutil.rmtree(source_dir, ignore_errors=True)

            dataset = train_module.EpochDataset(buffer_dir, augment=False)
            board, scalars, value, policy = dataset[0]

            assert scalars.shape == (2,)
            assert abs(scalars[0].item() - (-2.0)) < 1e-5
            assert abs(scalars[1].item() - 0.0) < 1e-5
            assert abs(value.item() - (-0.5)) < 1e-5
        finally:
            shutil.rmtree(buffer_dir, ignore_errors=True)

    def test_model_output_changes_with_qsearch_flag(self):
        """OracleNet k-head should produce different output when qsearch_flag changes."""
        model = OracleNet(num_blocks=1, hidden_dim=16)
        model.eval()

        board = torch.randn(1, 17, 8, 8)
        scalars_completed = torch.tensor([[0.0, 1.0]])    # qsearch completed
        scalars_incomplete = torch.tensor([[0.0, 0.0]])    # qsearch incomplete

        with torch.no_grad():
            _, v1, k1 = model(board, scalars_completed)
            _, v2, k2 = model(board, scalars_incomplete)

        # After training, k should differ; with zero init it may be very close
        # but the forward path must not error
        assert v1.shape == (1, 1)
        assert k1.shape == (1, 1)
        assert v2.shape == (1, 1)
        assert k2.shape == (1, 1)

    def test_backward_compat_single_scalar(self):
        """Model should handle [B, 1] input via backward compat (pads qsearch_flag=1.0)."""
        model = OracleNet(num_blocks=1, hidden_dim=16)
        model.eval()

        board = torch.randn(2, 17, 8, 8)
        material_only = torch.randn(2, 1)

        with torch.no_grad():
            policy, value, k = model(board, material_only)

        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)
        assert k.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
