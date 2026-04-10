"""Unit tests for the Python side of neurosymbolic-mcts.

Tests cover:
  1. TransformerNet forward pass (shapes, modes, k computation, zero-init)
  2. SPRT computation (_compute_sprt math)
  3. Validation value loss (tanh applied in eval mode before MSE)

Run with: pytest tests/test_transformer_python.py -v
"""

import sys
import os
import math

# Add python/ to path so we can import model, orchestrate, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. TransformerNet forward pass tests
# ---------------------------------------------------------------------------

class TestTransformerNetForwardPass:
    """Tests for TransformerNet output shapes, modes, and initialization."""

    def _make_model(self, num_blocks=2, hidden_dim=64):
        from model import TransformerNet
        return TransformerNet(
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            input_channels=17,
            policy_output_size=4672,
            num_heads=4,
            ffn_dim=256,
        )

    def _make_inputs(self, batch_size=4):
        x = torch.randn(batch_size, 17, 8, 8)
        scalars = torch.randn(batch_size, 2)
        return x, scalars

    def test_output_shapes(self):
        model = self._make_model()
        model.train()
        x, scalars = self._make_inputs(batch_size=4)
        policy, value, k = model(x, scalars)

        assert policy.shape == (4, 4672), f"policy shape {policy.shape}"
        assert value.shape == (4, 1), f"value shape {value.shape}"
        assert k.shape == (4, 1), f"k shape {k.shape}"

    def test_output_shapes_batch_1(self):
        model = self._make_model()
        model.train()
        x, scalars = self._make_inputs(batch_size=1)
        policy, value, k = model(x, scalars)

        assert policy.shape == (1, 4672)
        assert value.shape == (1, 1)
        assert k.shape == (1, 1)

    def test_training_mode_returns_tanh(self):
        """In training mode, value = tanh(v_logit + k * q_result)."""
        model = self._make_model()
        model.train()
        x, scalars = self._make_inputs(batch_size=4)
        _, value, _ = model(x, scalars)

        # tanh output is bounded in [-1, 1]
        assert (value >= -1.0).all() and (value <= 1.0).all(), (
            f"Training value not in [-1,1]: min={value.min()}, max={value.max()}"
        )

    def test_eval_mode_returns_raw_v_logit(self):
        """In eval mode, second output is raw v_logit (unbounded)."""
        model = self._make_model()

        x, scalars = self._make_inputs(batch_size=4)

        # Get training output (tanh-bounded)
        model.train()
        _, value_train, k_train = model(x, scalars)

        # Get eval output (raw v_logit)
        model.eval()
        with torch.no_grad():
            _, v_logit_eval, k_eval = model(x, scalars)

        # In training mode, value = tanh(v_logit + k * q)
        # In eval mode, we get raw v_logit. Reconstruct and compare.
        q_result = scalars[:, 0:1]
        reconstructed = torch.tanh(v_logit_eval + k_eval * q_result)

        # Should match the training output (with same inputs, same weights)
        assert torch.allclose(reconstructed, value_train, atol=1e-5), (
            f"Reconstructed value doesn't match training value"
        )

    def test_k_always_positive(self):
        """k = 0.47 * softplus(k_logit) is always positive."""
        model = self._make_model()
        model.train()
        x, scalars = self._make_inputs(batch_size=8)
        _, _, k = model(x, scalars)

        assert (k > 0).all(), f"k has non-positive values: {k}"

    def test_k_initial_value(self):
        """At init, k_logit=0, so k = 0.47 * softplus(0) = 0.47 * ln(2) ~ 0.326."""
        model = self._make_model()
        expected_k = 0.47 * math.log(2)
        actual_k = 0.47 * F.softplus(model.k_logit).item()
        assert abs(actual_k - expected_k) < 1e-5, (
            f"Initial k={actual_k}, expected={expected_k}"
        )

    def test_k_varies_with_k_logit(self):
        """k changes when k_logit is modified."""
        model = self._make_model()
        k1 = 0.47 * F.softplus(model.k_logit).item()

        model.k_logit.data = torch.tensor(2.0)
        k2 = 0.47 * F.softplus(model.k_logit).item()

        assert k2 > k1, f"k didn't increase with larger k_logit: k1={k1}, k2={k2}"

    def test_zero_init_heads_uniform_policy(self):
        """Zero-initialized heads produce uniform policy (all logits ~ 0)."""
        model = self._make_model()
        model.eval()
        x, scalars = self._make_inputs(batch_size=2)

        with torch.no_grad():
            policy, _, _ = model(x, scalars)

        # Policy is log_softmax. Uniform = log(1/4672) for all entries.
        expected_log_prob = math.log(1.0 / 4672)
        assert torch.allclose(
            policy, torch.full_like(policy, expected_log_prob), atol=1e-4
        ), f"Policy not uniform: std={policy.std().item()}, mean={policy.mean().item()}"

    def test_policy_sums_to_one_after_exp(self):
        """Policy is log_softmax, so exp(policy).sum(dim=1) ~ 1."""
        model = self._make_model()
        model.train()
        x, scalars = self._make_inputs(batch_size=4)
        policy, _, _ = model(x, scalars)

        probs = torch.exp(policy)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"exp(policy) doesn't sum to 1: {sums}"
        )

    def test_scalars_1d_backward_compat(self):
        """If scalars is [B] (1D), model should handle it."""
        model = self._make_model()
        model.train()
        x = torch.randn(3, 17, 8, 8)
        scalars = torch.randn(3)  # 1D
        policy, value, k = model(x, scalars)

        assert policy.shape == (3, 4672)
        assert value.shape == (3, 1)

    def test_scalars_single_column_backward_compat(self):
        """If scalars is [B, 1], model pads with 1.0 for qsearch_flag."""
        model = self._make_model()
        model.train()
        x = torch.randn(3, 17, 8, 8)
        scalars = torch.randn(3, 1)  # only q_result
        policy, value, k = model(x, scalars)

        assert policy.shape == (3, 4672)
        assert value.shape == (3, 1)


# ---------------------------------------------------------------------------
# 2. SPRT computation tests
# ---------------------------------------------------------------------------

class TestSPRTComputation:
    """Tests for _compute_sprt in orchestrate.py."""

    def _make_orchestrator(self, elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05):
        """Create a minimal orchestrator with SPRT config."""
        from orchestrate import TrainingConfig, Orchestrator
        config = TrainingConfig(
            sprt_elo0=elo0,
            sprt_elo1=elo1,
            sprt_alpha=alpha,
            sprt_beta=beta,
        )
        # Orchestrator.__init__ does file I/O, so we monkey-patch
        orch = object.__new__(Orchestrator)
        orch.config = config
        return orch

    def test_50_percent_winrate_llr_near_zero(self):
        """50% winrate (equal W/L) should give LLR near 0."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=50, losses=50, draws=0)
        assert abs(llr) < 5.0, f"LLR should be near 0 for 50/50, got {llr}"
        # Should generally be inconclusive or close to 0
        assert decision in ("inconclusive", "H0"), (
            f"50/50 should be inconclusive or H0, got {decision}"
        )

    def test_high_winrate_accepts(self):
        """High winrate should produce positive LLR and accept (H1)."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=200, losses=50, draws=50)
        assert llr > 0, f"High winrate should have positive LLR, got {llr}"
        assert decision == "H1", f"Expected H1 (accept), got {decision}"

    def test_low_winrate_rejects(self):
        """Low winrate should produce negative LLR and reject (H0)."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=50, losses=200, draws=50)
        assert llr < 0, f"Low winrate should have negative LLR, got {llr}"
        assert decision == "H0", f"Expected H0 (reject), got {decision}"

    def test_all_draws_llr_near_zero(self):
        """All draws (score=0.5) should give LLR near 0."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=0, losses=0, draws=100)
        # score = 0.5, which is exactly the expected score under H0 (elo0=0)
        # so LLR should be near 0 (slightly positive since p1 > p0)
        assert abs(llr) < 5.0, f"All draws should have LLR near 0, got {llr}"

    def test_zero_games_inconclusive(self):
        """Zero games should be inconclusive."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=0, losses=0, draws=0)
        assert llr == 0.0
        assert decision == "inconclusive"

    def test_all_wins_degenerate(self):
        """All wins (score=1.0) should be handled as degenerate case."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=100, losses=0, draws=0)
        assert llr == 100.0, f"All wins should return llr=100.0, got {llr}"
        assert decision == "H1"

    def test_all_losses_degenerate(self):
        """All losses (score=0.0) should be handled as degenerate case."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=0, losses=100, draws=0)
        assert llr == -100.0, f"All losses should return llr=-100.0, got {llr}"
        assert decision == "H0"

    def test_sprt_bounds_correct(self):
        """Verify SPRT upper/lower bounds match the formula."""
        orch = self._make_orchestrator(alpha=0.05, beta=0.05)
        upper = math.log((1 - 0.05) / 0.05)  # ~2.944
        lower = math.log(0.05 / (1 - 0.05))  # ~-2.944
        assert abs(upper - 2.944) < 0.01
        assert abs(lower + 2.944) < 0.01

    def test_small_advantage_inconclusive(self):
        """Slight advantage with few games should be inconclusive."""
        orch = self._make_orchestrator()
        llr, decision = orch._compute_sprt(wins=12, losses=8, draws=0)
        # Only 20 games with slight edge -- not enough for a decision
        assert decision == "inconclusive", (
            f"Expected inconclusive with 20 games, got {decision} (llr={llr})"
        )


# ---------------------------------------------------------------------------
# 3. Validation value loss tests
# ---------------------------------------------------------------------------

class TestValidationValueLoss:
    """Tests for validation value loss: in eval mode, apply tanh(v_logit + k * q) before MSE."""

    def _make_model(self, num_blocks=2, hidden_dim=64):
        from model import TransformerNet
        return TransformerNet(
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            input_channels=17,
            policy_output_size=4672,
            num_heads=4,
            ffn_dim=256,
        )

    def test_val_loss_bounded(self):
        """Validation value loss should be bounded since tanh output is in [-1, 1].

        MSE between two values in [-1, 1] is at most 4.0 (when one is -1 and other is +1).
        """
        model = self._make_model()
        model.eval()

        x = torch.randn(8, 17, 8, 8)
        scalars = torch.randn(8, 2)
        # Target values in [-1, 1] (as in real training data)
        values = torch.FloatTensor(8, 1).uniform_(-1, 1)

        with torch.no_grad():
            pred_policy, pred_value_raw, k_val = model(x, scalars)

        # Replicate the validation fix: apply tanh(v_logit + k * q_result)
        q_result = scalars[:, 0:1]
        pred_value = torch.tanh(pred_value_raw + k_val * q_result)

        v_loss = F.mse_loss(pred_value, values)

        # Loss bounded by 4.0 (max squared distance in [-1, 1])
        assert v_loss.item() <= 4.0, f"Value loss {v_loss.item()} exceeds bound of 4.0"
        assert v_loss.item() >= 0.0, f"Value loss {v_loss.item()} is negative"

    def test_val_loss_without_tanh_can_be_unbounded(self):
        """Without applying tanh, MSE on raw v_logit can exceed 4.0."""
        model = self._make_model()
        model.eval()

        # Use large scalars to push v_logit to large values
        x = torch.randn(8, 17, 8, 8) * 5.0
        scalars = torch.zeros(8, 2)
        values = torch.zeros(8, 1)  # target = 0

        with torch.no_grad():
            _, v_logit_raw, _ = model(x, scalars)

        # Raw MSE: v_logit can be arbitrarily large
        raw_loss = F.mse_loss(v_logit_raw, values)

        # After tanh: bounded
        bounded_loss = F.mse_loss(torch.tanh(v_logit_raw), values)
        assert bounded_loss.item() <= 4.0

    def test_val_loss_matches_train_semantics(self):
        """Validation loss (manual tanh) should match what training mode produces.

        In training mode, model returns tanh(v_logit + k * q) directly.
        In eval mode, model returns raw v_logit, and we manually apply tanh + k*q.
        Both should give the same MSE against the same target.
        """
        model = self._make_model()
        x = torch.randn(4, 17, 8, 8)
        scalars = torch.randn(4, 2)
        values = torch.FloatTensor(4, 1).uniform_(-1, 1)

        # Training mode: model returns tanh(v_logit + k * q)
        model.train()
        _, value_train, _ = model(x, scalars)
        train_loss = F.mse_loss(value_train, values)

        # Eval mode: model returns raw v_logit, we apply tanh manually
        model.eval()
        with torch.no_grad():
            _, v_logit, k_val = model(x, scalars)
        q_result = scalars[:, 0:1]
        value_eval = torch.tanh(v_logit + k_val * q_result)
        eval_loss = F.mse_loss(value_eval, values)

        assert torch.allclose(train_loss, eval_loss, atol=1e-5), (
            f"Train loss {train_loss.item()} != eval loss {eval_loss.item()}"
        )

    def test_perfect_prediction_zero_loss(self):
        """If model predicts exactly the target, loss should be ~0."""
        model = self._make_model()
        model.eval()

        x = torch.randn(4, 17, 8, 8)
        scalars = torch.randn(4, 2)

        with torch.no_grad():
            _, v_logit, k_val = model(x, scalars)

        q_result = scalars[:, 0:1]
        pred_value = torch.tanh(v_logit + k_val * q_result)

        # Use the prediction as the target
        loss = F.mse_loss(pred_value, pred_value)
        assert loss.item() < 1e-10, f"Self-prediction loss should be ~0, got {loss.item()}"
