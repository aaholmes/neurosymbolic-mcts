import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import OracleNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def index_to_uci(idx):
    """
    Decodes a flat policy index (0..4671) into a UCI move string.
    Implements a subset of the AlphaZero mapping (Queen slides) for demonstration.
    """
    if idx >= 4672: return "invalid"
    
    src = idx // 73
    plane = idx % 73
    
    src_rank = src // 8
    src_file = src % 8
    
    # Directions: N, NE, E, SE, S, SW, W, NW
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    if plane < 56:
        # Queen Move
        direction_idx = plane // 7
        distance = (plane % 7) + 1
        
        dx, dy = directions[direction_idx]
        
        dst_rank = src_rank + dy * distance
        dst_file = src_file + dx * distance
        
        if 0 <= dst_rank < 8 and 0 <= dst_file < 8:
            return f"{chr(ord('a')+src_file)}{src_rank+1}{chr(ord('a')+dst_file)}{dst_rank+1}"
            
    elif plane < 64:
        # Knight Move (56..63)
        knight_moves = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        k_idx = plane - 56
        dx, dy = knight_moves[k_idx]
        
        dst_rank = src_rank + dy
        dst_file = src_file + dx
        
        if 0 <= dst_rank < 8 and 0 <= dst_file < 8:
            return f"{chr(ord('a')+src_file)}{src_rank+1}{chr(ord('a')+dst_file)}{dst_rank+1}"
            
    # Underpromotions (64..72) are skipped for this simple decoder
    return f"idx{idx}"

def test_model_shapes():
    print(f"\nTesting OracleNet...")
    batch_size = 4
    # Random chess board input: [Batch, Channels, Height, Width]
    dummy_input = torch.randn(batch_size, 17, 8, 8)
    dummy_scalars = torch.randn(batch_size, 2)  # [material, qsearch_flag]

    try:
        model = OracleNet()
        policy, value, k = model(dummy_input, dummy_scalars)
        
        # Check Policy Shape
        # OracleNet uses 4672 for policy output size
        assert policy.shape == (batch_size, 4672), f"Policy shape mismatch! Expected ({batch_size}, 4672), got {policy.shape}"
        
        # Check Value Shape
        assert value.shape == (batch_size, 1), f"Value shape mismatch! Expected ({batch_size}, 1), got {value.shape}"
        
        # Check K Shape
        assert k.shape == (batch_size, 1), f"K shape mismatch! Expected ({batch_size}, 1), got {k.shape}"
        
        print(f"✅ Shapes Correct. Params: {count_parameters(model):,}")
        return model
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_overfitting():
    print(f"\nRunning Overfitting Test on OracleNet...")
    model = OracleNet()
    model.train() # Explicitly set to train mode
    
    # Use Adam for faster convergence in overfitting test
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create a fixed batch of random data
    batch_size = 4 # Smaller batch to make memorization easier
    inputs = torch.randn(batch_size, 17, 8, 8)
    materials = torch.randn(batch_size, 2)  # [material, qsearch_flag]
    
    # Target Policy: Random probabilities
    target_policy = torch.randn(batch_size, 4672).softmax(dim=1)
    
    # Target Value: Random values between -1 and 1
    target_value = torch.tanh(torch.randn(batch_size, 1))

    final_val_loss = 0
    
    print("  Iter | Total  | PolLoss | ValLoss | Avg K")
    print("  -----+--------+---------+---------+-------")
    
    for i in range(201):
        optimizer.zero_grad()
        
        # Forward pass: OracleNet returns (policy, value, k)
        pred_policy, pred_value, k = model(inputs, materials)
        
        # Loss = MSE(Value) + CrossEntropy(Policy)
        # Since pred_policy is log_softmax, we use negative sum product
        loss_value = F.mse_loss(pred_value, target_value)
        loss_policy = -torch.sum(target_policy * pred_policy) / batch_size
        
        loss = loss_value + loss_policy
        loss.backward()
        optimizer.step()
        
        if i == 200: 
            final_val_loss = loss_value.item()
        
        if i % 20 == 0:
            print(f"  {i:4d} | {loss.item():.4f} | {loss_policy.item():.4f}  | {loss_value.item():.4f} | {k.mean().item():.4f}")

    # Success criteria: Value loss should drop significantly (indicating gradient flow)
    if final_val_loss < 0.1:
        print(f"✅ Overfitting Successful (Value Loss dropped to {final_val_loss:.4f})")
    else:
        print(f"⚠️  Overfitting Warning: Value Loss didn't drop enough ({final_val_loss:.4f}). Check learning rate or gradient flow.")

def get_starting_position_tensor():
    # Returns [1, 17, 8, 8] tensor representing starting position
    board = torch.zeros(1, 17, 8, 8)
    
    # Ranks in tensor: 0=Rank 8 ... 7=Rank 1 (Flipped)
    
    # White Pieces (Rank 1 -> Tensor Row 7)
    # P=0, N=1, B=2, R=3, Q=4, K=5
    board[0, 3, 7, 0] = 1 # R
    board[0, 1, 7, 1] = 1 # N
    board[0, 2, 7, 2] = 1 # B
    board[0, 4, 7, 3] = 1 # Q
    board[0, 5, 7, 4] = 1 # K
    board[0, 2, 7, 5] = 1 # B
    board[0, 1, 7, 6] = 1 # N
    board[0, 3, 7, 7] = 1 # R
    
    # White Pawns (Rank 2 -> Tensor Row 6)
    board[0, 0, 6, :] = 1
    
    # Black Pieces (Rank 8 -> Tensor Row 0)
    # Offset by 6: P=6, N=7, B=8, R=9, Q=10, K=11
    board[0, 9, 0, 0] = 1 # r
    board[0, 7, 0, 1] = 1 # n
    board[0, 8, 0, 2] = 1 # b
    board[0, 10, 0, 3] = 1 # q
    board[0, 11, 0, 4] = 1 # k
    board[0, 8, 0, 5] = 1 # b
    board[0, 7, 0, 6] = 1 # n
    board[0, 9, 0, 7] = 1 # r
    
    # Black Pawns (Rank 7 -> Tensor Row 1)
    board[0, 6, 1, :] = 1
    
    # Castling Rights (All True)
    board[0, 13:17, :, :] = 1
    
    return board

def print_tensor_as_board(tensor):
    # tensor shape: [1, 17, 8, 8]
    # We only care about first 12 channels for visualization
    print("Board State (Us vs Them perspective):")
    print("  +-----------------+")
    
    # Mapping: Channel -> Char
    # 0-5: Us P, N, B, R, Q, K
    # 6-11: Them p, n, b, r, q, k
    chars = "PNBRQKpnbrqk"
    
    for row in range(8): # Tensor row 0 to 7 (Rank 8 to 1)
        rank_label = 8 - row
        line = f"{rank_label} | "
        for col in range(8):
            char = "."
            for ch in range(12):
                if tensor[0, ch, row, col] > 0.5:
                    char = chars[ch]
                    break # Assume only one piece per square
            line += f"{char} "
        line += "|"
        print(line)
    
    print("  +-----------------+")
    print("    a b c d e f g h")

def test_no_k_head_layers():
    """K-head layers removed, scalar k_logit added."""
    model = OracleNet()
    # Old layers must be gone
    assert not hasattr(model, 'k_stm_patch_fc')
    assert not hasattr(model, 'k_opp_patch_fc')
    assert not hasattr(model, 'k_combine')
    assert not hasattr(model, 'k_out')
    # New scalar param must exist
    assert hasattr(model, 'k_logit')
    assert isinstance(model.k_logit, nn.Parameter)
    assert model.k_logit.shape == ()  # scalar

def test_scalar_k_is_position_independent():
    """k must be identical for all inputs (global scalar)."""
    model = OracleNet()
    model.eval()
    with torch.no_grad():
        s = torch.zeros(1, 2)
        _, _, k1 = model(torch.randn(1, 17, 8, 8), s)
        _, _, k2 = model(torch.randn(1, 17, 8, 8), s)
    assert k1.item() == k2.item()

def test_v_fc_accepts_65_inputs():
    """Value head FC widened to 65 (64 conv + 1 q_result)."""
    model = OracleNet()
    assert model.v_fc.in_features == 65

def test_k_logit_zero_init_gives_texel_calibrated():
    """0.47 * softplus(0) = 0.47 * ln(2) ≈ 0.326."""
    model = OracleNet()
    k = 0.47 * F.softplus(model.k_logit)
    assert abs(k.item() - 0.326) < 0.01

def test_q_result_influences_v_logit():
    """After minimal training, changing scalars[0] (q_result) must change v_logit.

    At zero-init v_out is all zeros so v_logit=0 regardless of input.
    A single training step breaks the symmetry and lets q_result flow through.
    """
    model = OracleNet()
    # One training step to break zero-init symmetry
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dummy_board = torch.randn(1, 17, 8, 8)
    dummy_scalars = torch.tensor([[1.0, 1.0]])
    dummy_policy = torch.randn(1, 4672).softmax(dim=1)
    dummy_value = torch.tensor([[0.5]])
    pred_policy, pred_value, _ = model(dummy_board, dummy_scalars)
    loss = F.mse_loss(pred_value, dummy_value)
    loss.backward()
    optimizer.step()

    # Now test in eval mode
    model.eval()
    board = torch.randn(1, 17, 8, 8)
    with torch.no_grad():
        _, vlogit_a, _ = model(board, torch.tensor([[0.0, 1.0]]))
        _, vlogit_b, _ = model(board, torch.tensor([[5.0, 1.0]]))
    assert vlogit_a.item() != vlogit_b.item(), "q_result should influence v_logit after training"

def test_param_count_reduced():
    """~21,536 fewer params than old model."""
    model = OracleNet()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Old: ~2,000,622. New: ~1,979,086. Allow margin.
    assert count < 1_985_000, f"Too many params: {count}"
    assert count > 1_970_000, f"Too few params: {count}"


def sample_output_demo():
    print(f"\nSample Inference Output")
    print("-" * 30)
    
    model = OracleNet()
    model.eval()
    
    # Use starting position
    input_board = get_starting_position_tensor()
    input_material = torch.zeros(1, 2) # [material=0, qsearch_flag=0]
    
    # Visualize
    print_tensor_as_board(input_board)
    print()
    
    with torch.no_grad():
        policy, value, k = model(input_board, input_material)
        
    # Get top 3 moves
    # policy is log_softmax, convert to prob
    probs = torch.exp(policy).squeeze()
    topk_probs, topk_indices = torch.topk(probs, 3)
    
    moves_str = []
    for i in range(3):
        idx = topk_indices[i].item()
        prob = topk_probs[i].item()
        uci = index_to_uci(idx)
        moves_str.append(f"{uci} {prob:.4f}")
        
    val = value.item()
    print(f"Value: {val:.4f}")
    print(f"Top moves: {', '.join(moves_str)}")

if __name__ == "__main__":
    print("=== Verifying OracleNet Architecture ===")

    # 1. Test Shapes & Sizes
    test_model_shapes()

    # 2. K-head simplification tests
    test_no_k_head_layers()
    test_scalar_k_is_position_independent()
    test_v_fc_accepts_65_inputs()
    test_k_logit_zero_init_gives_texel_calibrated()
    test_q_result_influences_v_logit()
    test_param_count_reduced()

    # 3. Test Gradient Flow (Overfitting)
    test_overfitting()

    # 4. Sample Output
    sample_output_demo()
