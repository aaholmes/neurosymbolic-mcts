"""Board symmetry augmentation for training data.

Exploits horizontal flip symmetry (no castling) and D4 dihedral symmetry
(no castling, no pawns, no en passant) to augment training samples on-the-fly.

Policy encoding must match src/tensor.rs exactly:
  - Planes 0-55: Queen slides, dir(0-7)*7 + (dist-1)
    Dirs: N, NE, E, SE, S, SW, W, NW
  - Planes 56-63: Knight moves, 8 dirs
    (1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)
  - Planes 64-72: Underpromotions, piece_offset + dir_offset
    piece: N=0, B=3, R=6; dir: straight=0, capL=1, capR=2
  - Index = src_sq * 73 + plane
"""

import numpy as np

# ---------- Square permutation tables (64 entries) ----------

def _build_hflip_sq():
    """Horizontal flip: file -> 7-file, rank unchanged."""
    table = np.zeros(64, dtype=np.int32)
    for sq in range(64):
        rank, file = divmod(sq, 8)
        table[sq] = rank * 8 + (7 - file)
    return table

def _build_rot90_sq():
    """90 degrees CW rotation: (file, rank) -> (7-rank, file).
    new_sq = old_file * 8 + (7 - old_rank)."""
    table = np.zeros(64, dtype=np.int32)
    for sq in range(64):
        rank = sq // 8
        file = sq % 8
        table[sq] = file * 8 + (7 - rank)
    return table

HFLIP_SQ = _build_hflip_sq()
ROT90_SQ = _build_rot90_sq()

# ---------- Policy permutation tables (4672 entries) ----------

# Queen direction remapping under hflip (negate dx):
# N(0)->N(0), NE(1)->NW(7), E(2)->W(6), SE(3)->SW(5),
# S(4)->S(4), SW(5)->SE(3), W(6)->E(2), NW(7)->NE(1)
_HFLIP_QUEEN_DIR = [0, 7, 6, 5, 4, 3, 2, 1]

# Knight direction remapping under hflip (negate dx):
# (1,2)->(-1,2)=7, (2,1)->(-2,1)=6, (2,-1)->(-2,-1)=5, (1,-2)->(-1,-2)=4,
# (-1,-2)->(1,-2)=3, (-2,-1)->(2,-1)=2, (-2,1)->(2,1)=1, (-1,2)->(1,2)=0
_HFLIP_KNIGHT_DIR = [7, 6, 5, 4, 3, 2, 1, 0]

# Underpromotion direction remapping under hflip (negate dx):
# straight(0)->straight(0), capL(1)->capR(2), capR(2)->capL(1)
_HFLIP_UPROMO_DIR = [0, 2, 1]

# Under 90 CW rotation: (dx,dy) -> (dy, -dx), so direction index shifts +2 mod 8
_ROT90_QUEEN_DIR = [(d + 2) % 8 for d in range(8)]
_ROT90_KNIGHT_DIR = [(k + 2) % 8 for k in range(8)]
# Underpromotion: identity (D4 positions have no pawns, so these planes are unused)
_ROT90_UPROMO_DIR = [0, 1, 2]


def _build_policy_perm(sq_perm, queen_dir_map, knight_dir_map, upromo_dir_map):
    """Build a full 4672-element policy permutation from square + direction mappings."""
    perm = np.zeros(4672, dtype=np.int32)
    for src in range(64):
        new_src = sq_perm[src]
        for plane in range(73):
            old_idx = src * 73 + plane
            if plane < 56:
                # Queen slide: dir*7 + (dist-1)
                direction = plane // 7
                dist_minus_1 = plane % 7
                new_dir = queen_dir_map[direction]
                new_plane = new_dir * 7 + dist_minus_1
            elif plane < 64:
                # Knight: plane - 56 is direction index
                k_dir = plane - 56
                new_k_dir = knight_dir_map[k_dir]
                new_plane = 56 + new_k_dir
            else:
                # Underpromotion: plane - 64
                upromo_idx = plane - 64
                piece_offset = (upromo_idx // 3) * 3  # 0, 3, or 6
                dir_offset = upromo_idx % 3  # 0, 1, or 2
                new_dir_offset = upromo_dir_map[dir_offset]
                new_plane = 64 + piece_offset + new_dir_offset
            perm[old_idx] = new_src * 73 + new_plane
    return perm

HFLIP_POLICY = _build_policy_perm(HFLIP_SQ, _HFLIP_QUEEN_DIR, _HFLIP_KNIGHT_DIR, _HFLIP_UPROMO_DIR)
ROT90_POLICY = _build_policy_perm(ROT90_SQ, _ROT90_QUEEN_DIR, _ROT90_KNIGHT_DIR, _ROT90_UPROMO_DIR)

# ---------- D4 group: 8 transforms via composition ----------

def _compose_sq(a, b):
    """Compose two square permutations: result[sq] = b[a[sq]]."""
    return b[a]

def _compose_policy(a, b):
    """Compose two policy permutations: result[idx] = b[a[idx]]."""
    return b[a]

def _build_d4_group():
    """Build all 8 D4 transforms as (sq_perm, policy_perm) pairs.
    identity, rot90, rot180, rot270, hflip, hflip∘rot90, hflip∘rot180, hflip∘rot270
    """
    identity_sq = np.arange(64, dtype=np.int32)
    identity_pol = np.arange(4672, dtype=np.int32)

    rot180_sq = _compose_sq(ROT90_SQ, ROT90_SQ)
    rot270_sq = _compose_sq(rot180_sq, ROT90_SQ)
    rot180_pol = _compose_policy(ROT90_POLICY, ROT90_POLICY)
    rot270_pol = _compose_policy(rot180_pol, ROT90_POLICY)

    hflip_rot90_sq = _compose_sq(ROT90_SQ, HFLIP_SQ)
    hflip_rot180_sq = _compose_sq(rot180_sq, HFLIP_SQ)
    hflip_rot270_sq = _compose_sq(rot270_sq, HFLIP_SQ)
    hflip_rot90_pol = _compose_policy(ROT90_POLICY, HFLIP_POLICY)
    hflip_rot180_pol = _compose_policy(rot180_pol, HFLIP_POLICY)
    hflip_rot270_pol = _compose_policy(rot270_pol, HFLIP_POLICY)

    return [
        (identity_sq, identity_pol),
        (ROT90_SQ, ROT90_POLICY),
        (rot180_sq, rot180_pol),
        (rot270_sq, rot270_pol),
        (HFLIP_SQ, HFLIP_POLICY),
        (hflip_rot90_sq, hflip_rot90_pol),
        (hflip_rot180_sq, hflip_rot180_pol),
        (hflip_rot270_sq, hflip_rot270_pol),
    ]

D4_GROUP = _build_d4_group()

# The hflip-only group: identity + hflip
HFLIP_GROUP = [D4_GROUP[0], D4_GROUP[4]]

# ---------- Classification ----------

def classify_symmetry(board_tensor):
    """Classify the symmetry group of a board tensor (17x8x8 numpy array).

    Returns "d4", "hflip", or "none".

    Planes 13-16 are castling rights (all 1s if right exists).
    Planes 0 and 6 are STM and opponent pawns.
    Plane 12 is en passant.
    """
    # Check castling: planes 13-16
    has_castling = False
    for plane_idx in range(13, 17):
        if board_tensor[plane_idx].any():
            has_castling = True
            break

    if has_castling:
        return "none"

    # No castling — check for pawns and en passant for D4
    has_pawns = board_tensor[0].any() or board_tensor[6].any()
    has_ep = board_tensor[12].any()

    if not has_pawns and not has_ep:
        return "d4"

    return "hflip"


# ---------- Transform application ----------

def apply_transform(board, policy, sq_perm, pol_perm):
    """Apply a spatial transform to board tensor and policy vector.

    Args:
        board: numpy array of shape (17, 8, 8)
        policy: numpy array of shape (4672,)
        sq_perm: square permutation array of shape (64,)
        pol_perm: policy permutation array of shape (4672,)

    Returns:
        (new_board, new_policy)
    """
    # Transform board: for each plane, flatten, permute squares, reshape
    new_board = np.zeros_like(board)
    for plane_idx in range(board.shape[0]):
        flat = board[plane_idx].ravel()  # (64,)
        new_flat = np.zeros(64, dtype=flat.dtype)
        # sq_perm[old_sq] = new_sq, so new_flat[new_sq] = flat[old_sq]
        new_flat[sq_perm] = flat
        new_board[plane_idx] = new_flat.reshape(8, 8)

    # Transform policy: pol_perm[old_idx] = new_idx
    new_policy = np.zeros_like(policy)
    new_policy[pol_perm] = policy

    return new_board, new_policy


def augment_sample(board, material, value, policy, rng=None):
    """Augment a training sample using its symmetry group.

    DEPRECATED: use random_transform(). This returns a weight of 1/|group|,
    which underweights symmetric positions if applied to the loss.
    """
    if rng is None:
        rng = np.random.default_rng()

    sym = classify_symmetry(board)

    if sym == "d4":
        group = D4_GROUP
    elif sym == "hflip":
        group = HFLIP_GROUP
    else:
        return board, material, value, policy, 1.0

    idx = rng.integers(len(group))
    sq_perm, pol_perm = group[idx]

    if idx == 0:
        return board, material, value, policy, 1.0 / len(group)

    new_board, new_policy = apply_transform(board, policy, sq_perm, pol_perm)
    return new_board, material, value, new_policy, 1.0 / len(group)


def augment_all_transforms(board, material, value, policy):
    """Return ALL equivalent transforms of a training sample.

    For positions with castling rights: returns [original] (1 sample).
    For positions without castling (hflip): returns [original, hflip] (2 samples).
    For positions without castling/pawns/EP (D4): returns all 8 dihedral transforms.

    Training on every image counts a position once per image (8 times for a
    pawnless ending), so training draws one image with random_transform().

    Returns:
        List of (board, material, value, policy) tuples.
    """
    sym = classify_symmetry(board)

    if sym == "d4":
        group = D4_GROUP
    elif sym == "hflip":
        group = HFLIP_GROUP
    else:
        return [(board, material, value, policy)]

    results = []
    for i, (sq_perm, pol_perm) in enumerate(group):
        if i == 0:
            results.append((board, material, value, policy))
        else:
            new_board, new_policy = apply_transform(board, policy, sq_perm, pol_perm)
            results.append((new_board, material, value, new_policy))
    return results


def symmetry_group(board):
    """The transforms that map this position to an equivalent one."""
    sym = classify_symmetry(board)
    if sym == "d4":
        return D4_GROUP
    if sym == "hflip":
        return HFLIP_GROUP
    return D4_GROUP[:1]


def random_transform(board, material, value, policy, rng):
    """One image of a training sample, drawn uniformly from its symmetry group.

    Every position counts once in expectation whatever its symmetry, and
    repeated draws of one position see different images.
    """
    group = symmetry_group(board)
    i = int(rng.integers(len(group)))
    if i == 0:
        return board, material, value, policy
    sq_perm, pol_perm = group[i]
    new_board, new_policy = apply_transform(board, policy, sq_perm, pol_perm)
    return new_board, material, value, new_policy
