"""
Write golden fixtures for the Rust (candle) SeresNet port.

A tiny SeresNet (2 blocks, hidden 32) with every parameter randomized,
including the heads that SeresNet zero-initializes and the BatchNorm running
statistics. With the default init the policy is uniform and v_logit is 0, so
a broken port would still match.

Usage:
    python python/export_parity_fixture.py [out_dir]   # default tests/fixtures
"""

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
from model import SeresNet  # noqa: E402

NUM_BLOCKS = 2
HIDDEN_DIM = 32
BATCH = 16


def randomized_model(seed=0):
    torch.manual_seed(seed)
    model = SeresNet(num_blocks=NUM_BLOCKS, hidden_dim=HIDDEN_DIM)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 0:
                p.copy_(torch.randn(()))
            elif "bn" in name:
                # BatchNorm affine parameters: scale near 1, shift near 0
                p.copy_(1.0 + 0.2 * torch.randn_like(p) if name.endswith("weight")
                        else 0.1 * torch.randn_like(p))
            else:
                fan_in = p[0].numel() if p.dim() > 1 else p.numel()
                p.copy_(torch.randn_like(p) / fan_in ** 0.5)
        for name, buf in model.named_buffers():
            if name.endswith("running_mean"):
                buf.copy_(0.1 * torch.randn_like(buf))
            elif name.endswith("running_var"):
                buf.copy_(0.5 + torch.rand_like(buf))
    model.eval()
    return model


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
    os.makedirs(out_dir, exist_ok=True)

    model = randomized_model()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}

    g = torch.Generator().manual_seed(1)
    x = (torch.rand(BATCH, 17, 8, 8, generator=g) < 0.15).float()
    scalars = torch.stack([
        torch.randn(BATCH, generator=g) * 3.0,                 # q_result
        (torch.rand(BATCH, generator=g) < 0.5).float(),        # qsearch flag
    ], dim=1)

    with torch.no_grad():
        log_policy, v_logit, k = model(x, scalars)

    save_file(state, os.path.join(out_dir, "seresnet_tiny.safetensors"))
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {}},
               os.path.join(out_dir, "seresnet_tiny.pth"))
    save_file({
        "input": x.contiguous(),
        "scalars": scalars.contiguous(),
        "log_policy": log_policy.contiguous(),
        "v_logit": v_logit.contiguous(),
        "k": k.contiguous(),
    }, os.path.join(out_dir, "seresnet_tiny_io.safetensors"))

    p = log_policy.exp()
    print(f"wrote fixtures to {out_dir}: policy max prob {p.max():.4f}, "
          f"v_logit range [{v_logit.min():.3f}, {v_logit.max():.3f}], k {k[0, 0]:.4f}")


if __name__ == "__main__":
    main()
