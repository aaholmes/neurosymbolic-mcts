"""
Muon Optimizer Implementation

Momentum-based optimizer with Newton-Schulz orthogonalization.
Based on: https://github.com/KellerJordan/modded-nanogpt/blob/master/muon.py

Applies orthogonalization to 2D+ weight tensors (linear, conv layers).
Falls back to AdamW for 1D parameters (biases, norms).
"""
import torch
from torch.optim.optimizer import Optimizer


class Muon(Optimizer):
    """
    Muon optimizer with momentum-based orthogonalization.

    Args:
        params: model parameters
        lr: learning rate (default: 0.02, higher than Adam)
        momentum: momentum coefficient (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz orthogonalization steps (default: 5)
        backend_lr: learning rate for backend AdamW (1D params) (default: 1e-3)
        backend_weight_decay: weight decay for backend AdamW (default: 1e-4)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, backend_lr=1e-3, backend_weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        super().__init__(params, defaults)

        # Separate 1D params for backend AdamW optimizer
        backend_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim < 2:
                    backend_params.append(p)

        if backend_params:
            self.backend = torch.optim.AdamW(
                backend_params, lr=backend_lr, weight_decay=backend_weight_decay
            )
        else:
            self.backend = None

    @staticmethod
    def _newton_schulz(G, steps=5):
        """Newton-Schulz iteration for approximate orthogonalization.

        Iteratively refines G toward the closest orthogonal matrix using:
            X_{k+1} = a*X_k + b*(X_k @ X_k^T) @ X_k + c*(X_k @ X_k^T)^2 @ X_k
        """
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)

        # Normalize to unit spectral norm for convergence
        X = G.float()
        X /= (X.norm() + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Skip 1D params — handled by backend AdamW
                if p.ndim < 2:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf.clone()

                # Apply Newton-Schulz orthogonalization
                if p.ndim == 2:
                    # Linear layers: orthogonalize directly
                    update = self._newton_schulz(update, ns_steps)
                elif p.ndim == 4:
                    # Conv layers: flatten to 2D, orthogonalize, reshape back
                    shape = update.shape
                    update_2d = update.flatten(1)
                    update_2d = self._newton_schulz(update_2d, ns_steps)
                    update = update_2d.view(shape)

                p.add_(update, alpha=-lr)

        # Step backend optimizer for 1D params (biases, norms)
        if self.backend is not None:
            self.backend.step()

        return loss
