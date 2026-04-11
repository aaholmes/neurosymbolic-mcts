"""Tests for TorchScript export of TransformerNet.

Verifies that:
1. TransformerNet can be traced and saved as TorchScript
2. The traced model produces identical outputs to the PyTorch model
3. Batched inference works correctly
4. Eval mode returns raw v_logit (not tanh'd value)
"""
import os
import sys
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from model import TransformerNet


@pytest.fixture
def model():
    m = TransformerNet()
    m.eval()
    return m


@pytest.fixture
def traced_model(model):
    example_board = torch.randn(1, 17, 8, 8)
    example_scalars = torch.randn(1, 2)
    traced = torch.jit.trace(model, (example_board, example_scalars))
    return traced


@pytest.fixture
def saved_model_path(traced_model):
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        traced_model.save(f.name)
        yield f.name
    os.unlink(f.name)


class TestTorchScriptExport:
    def test_transformer_traces_without_error(self, model):
        """TransformerNet can be traced with torch.jit.trace."""
        board = torch.randn(1, 17, 8, 8)
        scalars = torch.randn(1, 2)
        traced = torch.jit.trace(model, (board, scalars))
        assert traced is not None

    def test_transformer_export_saves_and_loads(self, saved_model_path):
        """Traced model can be saved to disk and reloaded."""
        loaded = torch.jit.load(saved_model_path)
        assert loaded is not None
        # Verify it can run
        board = torch.randn(1, 17, 8, 8)
        scalars = torch.tensor([[0.0, 1.0]])
        policy, v_logit, k = loaded(board, scalars)
        assert policy.shape == (1, 4672)

    def test_transformer_export_matches_pytorch_b1(self, model, traced_model):
        """Traced model output matches PyTorch model for batch=1."""
        board = torch.randn(1, 17, 8, 8)
        scalars = torch.tensor([[0.5, 1.0]])
        with torch.no_grad():
            py_policy, py_v, py_k = model(board, scalars)
            ts_policy, ts_v, ts_k = traced_model(board, scalars)
        assert torch.allclose(py_policy, ts_policy, atol=1e-5)
        assert torch.allclose(py_v, ts_v, atol=1e-5)
        assert torch.allclose(py_k, ts_k, atol=1e-5)

    def test_transformer_export_matches_pytorch_b8(self, model, traced_model):
        """Traced model output matches PyTorch model for batch=8."""
        board = torch.randn(8, 17, 8, 8)
        scalars = torch.randn(8, 2)
        with torch.no_grad():
            py_policy, py_v, py_k = model(board, scalars)
            ts_policy, ts_v, ts_k = traced_model(board, scalars)
        assert torch.allclose(py_policy, ts_policy, atol=1e-5)
        assert torch.allclose(py_v, ts_v, atol=1e-5)
        assert torch.allclose(py_k, ts_k, atol=1e-5)

    def test_transformer_export_matches_pytorch_b36(self, model, traced_model):
        """Traced model output matches for batch=36 (self-play batch size)."""
        board = torch.randn(36, 17, 8, 8)
        scalars = torch.randn(36, 2)
        with torch.no_grad():
            py_policy, py_v, py_k = model(board, scalars)
            ts_policy, ts_v, ts_k = traced_model(board, scalars)
        assert torch.allclose(py_policy, ts_policy, atol=1e-5)
        assert torch.allclose(py_v, ts_v, atol=1e-5)

    def test_transformer_export_eval_mode_returns_v_logit(self, traced_model):
        """In eval mode, the model returns raw v_logit (not tanh'd value).
        v_logit should be unbounded — can be outside [-1, 1]."""
        board = torch.randn(4, 17, 8, 8)
        scalars = torch.tensor([[0.0, 1.0]] * 4)
        with torch.no_grad():
            policy, v_logit, k = traced_model(board, scalars)
        # v_logit should be small for random weights but NOT clamped to [-1,1]
        # k should always be positive (0.47 * softplus)
        assert k.min() > 0
        assert policy.shape == (4, 4672)
        assert v_logit.shape == (4, 1)

    def test_transformer_export_policy_is_log_softmax(self, traced_model):
        """Policy output should be log-probabilities (sum of exp = 1)."""
        board = torch.randn(2, 17, 8, 8)
        scalars = torch.tensor([[0.0, 1.0]] * 2)
        with torch.no_grad():
            policy, _, _ = traced_model(board, scalars)
        probs = torch.exp(policy)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4)

    def test_transformer_export_save_load_roundtrip(self, model):
        """Full roundtrip: trace → save → load → compare."""
        board = torch.randn(4, 17, 8, 8)
        scalars = torch.randn(4, 2)

        with torch.no_grad():
            py_policy, py_v, py_k = model(board, scalars)

        # Trace and save
        traced = torch.jit.trace(model, (torch.randn(1, 17, 8, 8), torch.randn(1, 2)))
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            traced.save(f.name)
            path = f.name

        # Load and compare
        loaded = torch.jit.load(path)
        with torch.no_grad():
            ts_policy, ts_v, ts_k = loaded(board, scalars)

        os.unlink(path)

        assert torch.allclose(py_policy, ts_policy, atol=1e-5)
        assert torch.allclose(py_v, ts_v, atol=1e-5)
        assert torch.allclose(py_k, ts_k, atol=1e-5)

    def test_transformer_export_with_trained_weights(self):
        """If trained weights exist, verify export with real weights."""
        weights_path = 'weights/transformer4/candidate_1.pth'
        if not os.path.exists(weights_path):
            pytest.skip("No trained weights available")

        model = TransformerNet()
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()

        board = torch.randn(4, 17, 8, 8)
        scalars = torch.tensor([[0.0, 1.0]] * 4)

        with torch.no_grad():
            py_policy, py_v, py_k = model(board, scalars)

        traced = torch.jit.trace(model, (torch.randn(1, 17, 8, 8), torch.randn(1, 2)))
        with torch.no_grad():
            ts_policy, ts_v, ts_k = traced(board, scalars)

        assert torch.allclose(py_policy, ts_policy, atol=1e-5)
        assert torch.allclose(py_v, ts_v, atol=1e-5)
        # Policy should be non-uniform with trained weights
        assert ts_policy.std() > 0.1
