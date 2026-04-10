#!/usr/bin/env python3
"""Generate reference data for CUDA transformer regression tests.

Usage: python3 scripts/gen_transformer_reference.py [weights.pth]

Generates binary files in cuda/test/reference_data/ that the CUDA test
test_transformer_vs_pytorch compares against. Rerun this whenever the
model architecture changes or you want to update the reference weights.
"""
import sys, os, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from model import TransformerNet

def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'weights/transformer4/candidate_1.pth'
    out_dir = 'cuda/test/reference_data'
    os.makedirs(out_dir, exist_ok=True)

    model = TransformerNet()
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    # Build starting position encoding (must match tf_board_to_tokens)
    # 17 channels: 0-5 STM pieces, 6-11 opp pieces, 12 EP, 13-16 castling
    tokens = np.zeros((64, 17), dtype=np.float32)
    for ch, squares in {
        0: [8,9,10,11,12,13,14,15],     # white pawns
        1: [1, 6], 2: [2, 5],           # white knights, bishops
        3: [0, 7], 4: [3], 5: [4],      # white rooks, queen, king
        6: [48,49,50,51,52,53,54,55],    # black pawns
        7: [57, 62], 8: [58, 61],        # black knights, bishops
        9: [56, 63], 10: [59], 11: [60], # black rooks, queen, king
    }.items():
        for sq in squares:
            tokens[sq, ch] = 1.0
    for sq in range(64):
        for c in [13, 14, 15, 16]:
            tokens[sq, c] = 1.0

    tokens.tofile(f'{out_dir}/startpos_tokens.bin')

    with torch.no_grad():
        t = torch.from_numpy(tokens).unsqueeze(0)

        after_proj = model.input_proj(t)
        after_proj.numpy()[0].tofile(f'{out_dir}/startpos_after_proj.bin')

        h = after_proj + model.pos_embedding
        h.numpy()[0].tofile(f'{out_dir}/startpos_after_posemb.bin')

        for i, block in enumerate(model.blocks):
            h = block(h)
            if i == 0:
                h.numpy()[0].tofile(f'{out_dir}/startpos_after_block0.bin')
        h.numpy()[0].tofile(f'{out_dir}/startpos_after_block5.bin')

        p = model.p_ln(h)
        p = model.p_head(p).reshape(1, -1)
        torch.log_softmax(p, dim=1).numpy()[0].tofile(f'{out_dir}/startpos_policy.bin')

        v = model.v_ln(h)
        v_pool = v.mean(dim=1)
        k_val = (0.47 * torch.log(1 + torch.exp(model.k_logit))).item()

        vals = []
        for q in [0.0, 3.0]:
            q_t = torch.tensor([[q]])
            v_feat = torch.cat([v_pool, q_t], dim=1)
            v_logit = model.v_fc2(torch.relu(model.v_fc1(v_feat))).item()
            value = float(np.tanh(v_logit + k_val * q))
            vals.extend([v_logit, value])
            print(f'q={q}: v_logit={v_logit:.8f}, value={value:.8f}, k={k_val:.8f}')

        np.array([vals[0], vals[1], k_val, vals[2], vals[3]], dtype=np.float32).tofile(
            f'{out_dir}/startpos_value.bin')

    print(f'\nReference data written to {out_dir}/:')
    for f in sorted(os.listdir(out_dir)):
        print(f'  {f}: {os.path.getsize(os.path.join(out_dir, f))} bytes')

if __name__ == '__main__':
    main()
