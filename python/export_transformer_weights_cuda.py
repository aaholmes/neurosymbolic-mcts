"""
Export TransformerNet weights to the flat binary format consumed by
cuda/transformer_weights.cu (load_transformer_weights).

Usage:
    python python/export_transformer_weights_cuda.py <checkpoint.pt> <output.bin>
    python python/export_transformer_weights_cuda.py <checkpoint.pt> <output.bin> --verify

The output file is a raw float32 binary dump of the TransformerWeights C struct,
with fields in the exact order defined in cuda/transformer_weights.cuh.
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

import os
sys.path.insert(0, os.path.dirname(__file__))
from model import TransformerNet

# Architecture constants (must match cuda/transformer_weights.cuh)
NN_HIDDEN_DIM      = 128
NN_INPUT_CHANNELS  = 17
TF_NUM_LAYERS      = 12   # must match cuda/transformer_weights.cuh
TF_NUM_HEADS       = 4
TF_HEAD_DIM        = 32
TF_FFN_DIM         = 512
TF_VALUE_FC1_OUT   = 256
NN_POLICY_PLANES   = 73


def t(tensor):
    """Detach, move to CPU, cast to float32, flatten to 1-D numpy array."""
    return tensor.detach().cpu().float().numpy().ravel()


def ln_fields(ln):
    """Emit LayerNorm arrays: weight (γ), bias (β)."""
    return [t(ln.weight), t(ln.bias)]


def build_parts(model):
    """Build flat weight array matching TransformerWeights struct layout."""
    parts = []

    # --- Input projection ---
    parts.append(t(model.input_proj.weight))      # [128, 17]
    parts.append(t(model.input_proj.bias))         # [128]
    parts.append(t(model.pos_embedding))           # [64, 128]

    # --- Transformer blocks ---
    for block in model.blocks:
        # Pre-attention LayerNorm
        parts.extend(ln_fields(block.ln1))

        # QKV projection (fused)
        parts.append(t(block.qkv.weight))          # [384, 128]
        parts.append(t(block.qkv.bias))            # [384]

        # Output projection
        parts.append(t(block.out_proj.weight))     # [128, 128]
        parts.append(t(block.out_proj.bias))       # [128]

        # Pre-FFN LayerNorm
        parts.extend(ln_fields(block.ln2))

        # FFN
        parts.append(t(block.ffn1.weight))         # [512, 128]
        parts.append(t(block.ffn1.bias))           # [512]
        parts.append(t(block.ffn2.weight))         # [128, 512]
        parts.append(t(block.ffn2.bias))           # [128]

    # --- Policy head ---
    parts.extend(ln_fields(model.p_ln))
    parts.append(t(model.p_head.weight))           # [73, 128]
    parts.append(t(model.p_head.bias))             # [73]

    # --- Value head ---
    parts.extend(ln_fields(model.v_ln))
    parts.append(t(model.v_fc1.weight))            # [256, 129]
    parts.append(t(model.v_fc1.bias))              # [256]
    parts.append(t(model.v_fc2.weight.reshape(-1)))  # [256]
    parts.append(t(model.v_fc2.bias))              # [1]

    # --- K scalar ---
    parts.append(t(model.k_logit.unsqueeze(0)))    # [1]

    return parts


def compute_expected_floats():
    """Compute expected number of floats matching TransformerWeights struct."""
    per_block = (
        2 * NN_HIDDEN_DIM                              # ln1: weight + bias
        + NN_HIDDEN_DIM * 3 * NN_HIDDEN_DIM            # qkv_weight
        + 3 * NN_HIDDEN_DIM                             # qkv_bias
        + NN_HIDDEN_DIM * NN_HIDDEN_DIM                 # out_proj_weight
        + NN_HIDDEN_DIM                                 # out_proj_bias
        + 2 * NN_HIDDEN_DIM                             # ln2: weight + bias
        + NN_HIDDEN_DIM * TF_FFN_DIM                    # ffn1_weight
        + TF_FFN_DIM                                    # ffn1_bias
        + TF_FFN_DIM * NN_HIDDEN_DIM                    # ffn2_weight
        + NN_HIDDEN_DIM                                 # ffn2_bias
    )
    total = (
        NN_HIDDEN_DIM * NN_INPUT_CHANNELS               # input_proj_weight
        + NN_HIDDEN_DIM                                  # input_proj_bias
        + 64 * NN_HIDDEN_DIM                             # pos_embedding
        + TF_NUM_LAYERS * per_block
        + 2 * NN_HIDDEN_DIM                              # p_ln
        + NN_POLICY_PLANES * NN_HIDDEN_DIM               # p_head_weight
        + NN_POLICY_PLANES                               # p_head_bias
        + 2 * NN_HIDDEN_DIM                              # v_ln
        + TF_VALUE_FC1_OUT * (NN_HIDDEN_DIM + 1)         # v_fc1_weight
        + TF_VALUE_FC1_OUT                               # v_fc1_bias
        + TF_VALUE_FC1_OUT                               # v_fc2_weight
        + 1                                              # v_fc2_bias
        + 1                                              # k_logit
    )
    return total


def main():
    parser = argparse.ArgumentParser(description="Export TransformerNet weights for CUDA kernel")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("output", help="Output binary file path (.bin)")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        state_dict = raw.get("model_state_dict", raw)
    else:
        state_dict = raw.state_dict()

    model = TransformerNet(num_blocks=args.num_blocks, hidden_dim=args.hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

    k_val = 0.47 * float(F.softplus(model.k_logit.detach()))
    print(f"  k_logit = {float(model.k_logit):.6f}  →  k = {k_val:.6f}")

    parts = build_parts(model)
    all_weights = np.concatenate(parts).astype(np.float32)

    expected = compute_expected_floats()
    if len(all_weights) != expected:
        print(f"ERROR: float count {len(all_weights)} != expected {expected}")
        sys.exit(1)

    all_weights.tofile(args.output)
    size_bytes = len(all_weights) * 4
    print(f"Written: {args.output}  ({size_bytes:,} bytes = {size_bytes/1024/1024:.2f} MB)")

    if args.verify:
        readback = np.fromfile(args.output, dtype=np.float32)
        assert len(readback) == expected
        assert np.allclose(readback, all_weights, atol=0)
        print("Verify: readback matches.")


if __name__ == "__main__":
    main()
