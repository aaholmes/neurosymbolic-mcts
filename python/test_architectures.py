import torch
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

def test_k_feature_extraction():
    """Verify k head feature extraction on known starting position."""
    print(f"\nTesting k Feature Extraction...")
    model = OracleNet()
    model.eval()

    board = get_starting_position_tensor()  # [1, 17, 8, 8]
    errors = []

    # --- Test scalar features ---
    scalars = model._extract_k_scalars(board)  # [1, 12]
    assert scalars.shape == (1, 12), f"Scalars shape {scalars.shape} != (1, 12)"

    total_pawns = scalars[0, 0].item()
    stm_pieces = scalars[0, 1].item()
    opp_pieces = scalars[0, 2].item()
    stm_queen = scalars[0, 3].item()
    opp_queen = scalars[0, 4].item()
    contacts = scalars[0, 5].item()
    castling = scalars[0, 6].item()
    stm_king_rank = scalars[0, 7].item()

    if abs(total_pawns - 16.0) > 0.01:
        errors.append(f"total_pawns={total_pawns}, expected 16")
    if abs(stm_pieces - 7.0) > 0.01:
        errors.append(f"stm_pieces={stm_pieces}, expected 7 (2N+2B+2R+1Q)")
    if abs(opp_pieces - 7.0) > 0.01:
        errors.append(f"opp_pieces={opp_pieces}, expected 7")
    if abs(stm_queen - 1.0) > 0.01:
        errors.append(f"stm_queen={stm_queen}, expected 1")
    if abs(opp_queen - 1.0) > 0.01:
        errors.append(f"opp_queen={opp_queen}, expected 1")
    if abs(contacts - 0.0) > 0.01:
        errors.append(f"contacts={contacts}, expected 0 (pawns not adjacent)")
    if abs(castling - 4.0) > 0.01:
        errors.append(f"castling={castling}, expected 4")
    # STM king is at tensor row 7, col 4 -> flat pos = 7*8+4 = 60, rank = 60//8 = 7
    if abs(stm_king_rank - 7.0) > 0.01:
        errors.append(f"stm_king_rank={stm_king_rank}, expected 7")

    # --- Test king patch extraction ---
    stm_patch = model._extract_king_patch(board, king_plane=5)  # [1, 300]
    opp_patch = model._extract_king_patch(board, king_plane=11)  # [1, 300]

    assert stm_patch.shape == (1, 300), f"STM patch shape {stm_patch.shape} != (1, 300)"
    assert opp_patch.shape == (1, 300), f"Opp patch shape {opp_patch.shape} != (1, 300)"

    # STM king at row 7, col 4 (e1 in tensor coords)
    # 5x5 patch centered there: rows 5-9, cols 2-6 (padded board rows 7-11, cols 4-8)
    # Rows 7 (rank 1) has pieces, row 6 (rank 2) has pawns, rows 5,8,9 are empty/padding
    # Patch should contain non-zero values (pieces around king)
    stm_nonzero = (stm_patch.abs() > 0.01).sum().item()
    if stm_nonzero == 0:
        errors.append("STM king patch is all zeros — expected pieces around king")

    # Opponent king at row 0, col 4 (e8 in tensor coords)
    opp_nonzero = (opp_patch.abs() > 0.01).sum().item()
    if opp_nonzero == 0:
        errors.append("Opp king patch is all zeros — expected pieces around king")

    # --- Test k output on starting position ---
    with torch.no_grad():
        scalars_input = torch.zeros(1, 2)  # [material=0, qsearch_flag=0]
        _, _, k = model(board, scalars_input)

    if abs(k.item() - 0.5) > 0.05:
        errors.append(f"k={k.item():.4f}, expected ~0.5 with zero init")

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        print(f"❌ k Feature Extraction FAILED ({len(errors)} errors)")
    else:
        print(f"  Scalars: pawns={total_pawns:.0f}, stm_pcs={stm_pieces:.0f}, opp_pcs={opp_pieces:.0f}, "
              f"stm_Q={stm_queen:.0f}, opp_Q={opp_queen:.0f}, contacts={contacts:.0f}, "
              f"castling={castling:.0f}, king_rank={stm_king_rank:.0f}")
        print(f"  STM patch nonzero: {stm_nonzero}/300, Opp patch nonzero: {opp_nonzero}/300")
        print(f"  k = {k.item():.4f} (expected ~0.5)")
        print(f"✅ k Feature Extraction Correct")


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
    
    # 2. Test k Feature Extraction
    test_k_feature_extraction()

    # 3. Test Gradient Flow (Overfitting)
    test_overfitting()

    # 4. Sample Output
    sample_output_demo()
