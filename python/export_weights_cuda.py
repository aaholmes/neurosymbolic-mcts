"""
Export trained OracleNet weights to the flat binary format consumed by
cuda/nn_weights.cu (load_nn_weights).

Usage:
    python python/export_weights_cuda.py <checkpoint.pt> <output.bin>
    python python/export_weights_cuda.py <checkpoint.pt> <output.bin> --verify

The output file is a raw float32 binary dump of the OracleNetWeights C struct,
with fields in the exact order defined in cuda/nn_weights.cuh.
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Add python/ to path so we can import model.py regardless of cwd
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import OracleNet


# ---------------------------------------------------------------------------
# Architecture constants (must match cuda/nn_weights.cuh)
# ---------------------------------------------------------------------------
NN_HIDDEN_DIM      = 128
NN_INPUT_CHANNELS  = 17
NN_NUM_BLOCKS      = 6
NN_SE_INNER        = NN_HIDDEN_DIM // 16   # 8
NN_POLICY_PLANES   = 73
NN_VALUE_FC1_IN    = 65   # 64 + 1 (q_result)
NN_VALUE_FC1_OUT   = 256

EXPECTED_FLOATS = (
    NN_HIDDEN_DIM * NN_INPUT_CHANNELS * 3 * 3  # start_conv_weight
    + 4 * NN_HIDDEN_DIM                         # start_bn
    + NN_NUM_BLOCKS * (
        NN_HIDDEN_DIM * NN_HIDDEN_DIM * 3 * 3   # conv1_weight
        + 4 * NN_HIDDEN_DIM                      # bn1
        + NN_HIDDEN_DIM * NN_HIDDEN_DIM * 3 * 3  # conv2_weight
        + 4 * NN_HIDDEN_DIM                      # bn2
        + NN_SE_INNER * NN_HIDDEN_DIM            # se.fc1_weight
        + NN_HIDDEN_DIM * NN_SE_INNER            # se.fc2_weight
    )
    + NN_HIDDEN_DIM * NN_HIDDEN_DIM * 3 * 3     # p_conv_weight
    + 4 * NN_HIDDEN_DIM                          # p_bn
    + NN_POLICY_PLANES * NN_HIDDEN_DIM           # p_head_weight
    + NN_POLICY_PLANES                           # p_head_bias
    + NN_HIDDEN_DIM                              # v_conv_weight
    + 1                                          # v_conv_bias
    + 4 * 1                                      # v_bn
    + NN_VALUE_FC1_OUT * NN_VALUE_FC1_IN         # v_fc_weight
    + NN_VALUE_FC1_OUT                           # v_fc_bias
    + NN_VALUE_FC1_OUT                           # v_out_weight
    + 1                                          # v_out_bias
    + 1                                          # k_logit
)
EXPECTED_BYTES = EXPECTED_FLOATS * 4


def t(tensor):
    """Detach, move to CPU, cast to float32, flatten to 1-D numpy array."""
    return tensor.detach().cpu().float().numpy().ravel()


def bn_fields(bn):
    """Emit BN arrays in struct order: weight (γ), bias (β), running_mean, running_var."""
    return [t(bn.weight), t(bn.bias), t(bn.running_mean), t(bn.running_var)]


def build_parts(model):
    parts = []

    # --- Input conv + BN ---
    parts.append(t(model.start_conv.weight))          # [128, 17, 3, 3]
    parts.extend(bn_fields(model.start_bn))

    # --- 6 Residual blocks ---
    for i, block in enumerate(model.res_blocks):
        parts.append(t(block.conv1.weight))            # [128, 128, 3, 3]
        parts.extend(bn_fields(block.bn1))
        parts.append(t(block.conv2.weight))            # [128, 128, 3, 3]
        parts.extend(bn_fields(block.bn2))
        parts.append(t(block.se.fc[0].weight))         # [8, 128]
        parts.append(t(block.se.fc[2].weight))         # [128, 8]

    # --- Policy head ---
    parts.append(t(model.p_conv.weight))               # [128, 128, 3, 3]
    parts.extend(bn_fields(model.p_bn))
    parts.append(t(model.p_head.weight.reshape(NN_POLICY_PLANES, NN_HIDDEN_DIM)))  # [73,128]
    parts.append(t(model.p_head.bias))                 # [73]

    # --- Value head ---
    parts.append(t(model.v_conv.weight.reshape(NN_HIDDEN_DIM)))  # [128]
    parts.append(t(model.v_conv.bias))                 # [1]
    parts.extend(bn_fields(model.v_bn))
    parts.append(t(model.v_fc.weight))                 # [256, 65]
    parts.append(t(model.v_fc.bias))                   # [256]
    parts.append(t(model.v_out.weight.reshape(NN_VALUE_FC1_OUT)))  # [256]
    parts.append(t(model.v_out.bias))                  # [1]

    # --- K scalar ---
    parts.append(t(model.k_logit.unsqueeze(0)))        # [1]

    return parts


def verify(output_path, model):
    """Read back the binary and spot-check a few fields against the model."""
    raw = np.fromfile(output_path, dtype=np.float32)
    assert len(raw) == EXPECTED_FLOATS, \
        f"Readback size mismatch: {len(raw)} vs {EXPECTED_FLOATS}"

    errors = 0

    def check(name, arr_from_file, tensor):
        nonlocal errors
        expected = tensor.detach().cpu().float().numpy().ravel()
        if not np.allclose(arr_from_file, expected, atol=0):
            print(f"  FAIL {name}: first mismatch at index "
                  f"{np.argmax(arr_from_file != expected)}")
            errors += 1
        else:
            print(f"  OK   {name}")

    # Compute byte offsets for spot-check fields
    offset = 0

    # start_conv_weight
    n = NN_HIDDEN_DIM * NN_INPUT_CHANNELS * 3 * 3
    check("start_conv.weight", raw[offset:offset+n], model.start_conv.weight)
    offset += n

    # start_bn weight (γ)
    check("start_bn.weight", raw[offset:offset+NN_HIDDEN_DIM], model.start_bn.weight)
    offset += NN_HIDDEN_DIM
    # start_bn bias (β)
    check("start_bn.bias", raw[offset:offset+NN_HIDDEN_DIM], model.start_bn.bias)
    offset += NN_HIDDEN_DIM
    # start_bn running_mean
    check("start_bn.running_mean", raw[offset:offset+NN_HIDDEN_DIM], model.start_bn.running_mean)
    offset += NN_HIDDEN_DIM
    # start_bn running_var
    check("start_bn.running_var", raw[offset:offset+NN_HIDDEN_DIM], model.start_bn.running_var)
    offset += NN_HIDDEN_DIM

    # k_logit is the last float
    check("k_logit", raw[-1:], model.k_logit.unsqueeze(0))

    # v_fc_bias is 257 floats from the end (v_out_weight[256] + v_out_bias[1] + k_logit[1])
    vfc_bias_end = EXPECTED_FLOATS - NN_VALUE_FC1_OUT - 1 - 1
    vfc_bias_start = vfc_bias_end - NN_VALUE_FC1_OUT
    check("v_fc.bias", raw[vfc_bias_start:vfc_bias_end], model.v_fc.bias)

    if errors == 0:
        print("All spot-checks passed.")
    else:
        print(f"{errors} spot-check(s) FAILED.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export OracleNet weights for CUDA kernel")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("output", help="Output binary file path (.bin)")
    parser.add_argument("--verify", action="store_true",
                        help="Read back output and verify against model tensors")
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    if args.num_blocks != 6 or args.hidden_dim != 128:
        print("Warning: CUDA struct is hard-coded for num_blocks=6, hidden_dim=128.")
        print("Non-default architectures will produce a size-mismatched file.")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        state_dict = raw.get("model_state_dict", raw)
    else:
        # TorchScript archive
        print("  Detected TorchScript archive, extracting state_dict...")
        state_dict = raw.state_dict()

    model = OracleNet(num_blocks=args.num_blocks, hidden_dim=args.hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

    k_val = 0.47 * float(F.softplus(model.k_logit.detach()))
    print(f"  k_logit = {float(model.k_logit):.6f}  →  k = {k_val:.6f}")

    # Build flat array
    parts = build_parts(model)
    all_weights = np.concatenate(parts).astype(np.float32)

    if len(all_weights) != EXPECTED_FLOATS:
        print(f"ERROR: float count {len(all_weights)} != expected {EXPECTED_FLOATS}")
        sys.exit(1)

    # Write
    all_weights.tofile(args.output)
    size_bytes = len(all_weights) * 4
    print(f"Written: {args.output}  ({size_bytes:,} bytes = {size_bytes/1024/1024:.2f} MB)")

    if args.verify:
        print("Verifying readback...")
        verify(args.output, model)


if __name__ == "__main__":
    main()
