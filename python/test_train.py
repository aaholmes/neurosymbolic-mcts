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

from model import LogosNet
from replay_buffer import ReplayBuffer, SAMPLE_SIZE_FLOATS, BYTES_PER_SAMPLE

# Import training internals (we'll test functions directly)
import train as train_module


def make_fake_bin(path, num_positions):
    """Create a fake .bin file with valid-shaped random data."""
    data = np.random.randn(num_positions, SAMPLE_SIZE_FLOATS).astype(np.float32)
    data.tofile(path)


def make_tiny_model():
    """Create a tiny model for fast testing."""
    return LogosNet(num_blocks=1, hidden_dim=16)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
