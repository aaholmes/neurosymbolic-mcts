"""Tests for board symmetry augmentation."""

import numpy as np
import pytest
from augmentation import (
    HFLIP_SQ, ROT90_SQ, HFLIP_POLICY, ROT90_POLICY,
    D4_GROUP, HFLIP_GROUP,
    classify_symmetry, apply_transform, augment_sample, augment_all_transforms,
    _compose_sq, _compose_policy,
)


class TestPermutationTables:
    def test_hflip_sq_invertible(self):
        """hflip composed with itself = identity."""
        result = HFLIP_SQ[HFLIP_SQ]
        np.testing.assert_array_equal(result, np.arange(64))

    def test_rot90_sq_order_4(self):
        """rot90 applied 4 times = identity."""
        perm = np.arange(64, dtype=np.int32)
        for _ in range(4):
            perm = ROT90_SQ[perm]
        np.testing.assert_array_equal(perm, np.arange(64))

    def test_hflip_policy_invertible(self):
        """hflip policy composed with itself = identity."""
        result = HFLIP_POLICY[HFLIP_POLICY]
        np.testing.assert_array_equal(result, np.arange(4672))

    def test_rot90_policy_order_4(self):
        """rot90 policy applied 4 times = identity."""
        perm = np.arange(4672, dtype=np.int32)
        for _ in range(4):
            perm = ROT90_POLICY[perm]
        np.testing.assert_array_equal(perm, np.arange(4672))

    def test_sq_permutations_are_bijections(self):
        """Each square permutation maps to unique targets."""
        assert len(set(HFLIP_SQ)) == 64
        assert len(set(ROT90_SQ)) == 64

    def test_policy_permutations_are_bijections(self):
        """Each policy permutation maps to unique targets."""
        assert len(set(HFLIP_POLICY)) == 4672
        assert len(set(ROT90_POLICY)) == 4672

    def test_hflip_sq_known_values(self):
        """a1(0)->h1(7), e2(12)->d2(11), h8(63)->a8(56)."""
        assert HFLIP_SQ[0] == 7    # a1 -> h1
        assert HFLIP_SQ[7] == 0    # h1 -> a1
        assert HFLIP_SQ[12] == 11  # e2 -> d2
        assert HFLIP_SQ[56] == 63  # a8 -> h8
        assert HFLIP_SQ[63] == 56  # h8 -> a8

    def test_rot90_sq_known_values(self):
        """a1(0)->a8(56), h1(7)->a1(0), h8(63)->h1(7), a8(56)->h8(63)."""
        # 90 CW: (file, rank) -> (7-rank, file)
        # a1: rank=0, file=0 -> new_sq = 0*8 + 7 = 7 -> h1
        # Wait, let me recompute: new_sq = file*8 + (7-rank) = 0*8+7 = 7
        assert ROT90_SQ[0] == 7    # a1 -> h1
        # h1: rank=0, file=7 -> new_sq = 7*8 + 7 = 63
        assert ROT90_SQ[7] == 63   # h1 -> h8
        # h8: rank=7, file=7 -> new_sq = 7*8 + 0 = 56
        assert ROT90_SQ[63] == 56  # h8 -> a8
        # a8: rank=7, file=0 -> new_sq = 0*8 + 0 = 0
        assert ROT90_SQ[56] == 0   # a8 -> a1


class TestPolicySumPreservation:
    def test_hflip_preserves_sum(self):
        """Hflip permutation preserves total policy mass."""
        rng = np.random.default_rng(42)
        policy = rng.random(4672).astype(np.float32)
        policy /= policy.sum()
        new_policy = np.zeros_like(policy)
        new_policy[HFLIP_POLICY] = policy
        np.testing.assert_almost_equal(new_policy.sum(), policy.sum(), decimal=5)

    def test_rot90_preserves_sum(self):
        """Rot90 permutation preserves total policy mass."""
        rng = np.random.default_rng(42)
        policy = rng.random(4672).astype(np.float32)
        policy /= policy.sum()
        new_policy = np.zeros_like(policy)
        new_policy[ROT90_POLICY] = policy
        np.testing.assert_almost_equal(new_policy.sum(), policy.sum(), decimal=5)


class TestKnownMoveMapping:
    def test_e2e4_hflip_to_d2d4(self):
        """e2-e4 (pawn push) hflipped -> d2-d4.

        e2=12, e4=28: dx=0, dy=2, N dir=0, dist=2, plane=1. idx=12*73+1=877
        d2=11, d4=27: same dir/dist, plane=1. idx=11*73+1=804
        """
        assert HFLIP_POLICY[877] == 804

    def test_ng1f3_hflip_to_nb1c3(self):
        """Ng1-f3 hflipped -> Nb1-c3.

        g1=6, f3=21: dx=-1, dy=2, knight dir (-1,2)=idx7, plane=63. idx=6*73+63=501
        b1=1, c3=18: dx=1, dy=2, knight dir (1,2)=idx0, plane=56. idx=1*73+56=129
        """
        assert HFLIP_POLICY[501] == 129

    def test_e2e4_rot90(self):
        """e2-e4 rot90 CW.

        e2=12 (rank=1, file=4) -> rot90 -> file*8+(7-rank) = 4*8+6 = 38
        N dir(0) -> E dir(2) under +2 mod 8. dist=2 unchanged. plane = 2*7+1 = 15.
        idx = 38*73+15 = 2789
        """
        old_idx = 12 * 73 + 1  # 877
        assert ROT90_POLICY[old_idx] == 38 * 73 + 15


class TestClassification:
    def _make_board(self, castling=False, pawns=False, ep=False):
        """Create a minimal board tensor for classification testing."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        # Place kings (plane 5=STM king, plane 11=opp king)
        board[5, 0, 4] = 1.0  # e1
        board[11, 7, 4] = 1.0  # e8
        if castling:
            board[13] = 1.0  # STM kingside castling
        if pawns:
            board[0, 1, 0] = 1.0  # STM pawn on a2
        if ep:
            board[12, 2, 3] = 1.0  # en passant on d3
        return board

    def test_castling_position_is_none(self):
        board = self._make_board(castling=True)
        assert classify_symmetry(board) == "none"

    def test_no_castling_with_pawns_is_hflip(self):
        board = self._make_board(pawns=True)
        assert classify_symmetry(board) == "hflip"

    def test_no_castling_with_ep_is_hflip(self):
        """En passant without pawns (edge case) is still hflip."""
        board = self._make_board(ep=True)
        assert classify_symmetry(board) == "hflip"

    def test_bare_kings_is_d4(self):
        board = self._make_board()
        assert classify_symmetry(board) == "d4"

    def test_no_castling_no_pawns_with_pieces_is_d4(self):
        board = self._make_board()
        board[1, 3, 3] = 1.0  # STM knight on d4
        assert classify_symmetry(board) == "d4"


class TestWeights:
    def test_none_weight(self):
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[13] = 1.0  # castling
        policy = np.zeros(4672, dtype=np.float32)
        _, _, _, _, w = augment_sample(board, 0.0, 0.5, policy)
        assert w == 1.0

    def test_hflip_weight(self):
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[0, 1, 0] = 1.0  # pawn
        policy = np.zeros(4672, dtype=np.float32)
        _, _, _, _, w = augment_sample(board, 0.0, 0.5, policy)
        assert w == 0.5

    def test_d4_weight(self):
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[5, 0, 4] = 1.0  # king only
        policy = np.zeros(4672, dtype=np.float32)
        _, _, _, _, w = augment_sample(board, 0.0, 0.5, policy)
        assert w == 0.125


class TestD4GroupClosure:
    def test_all_8_are_distinct(self):
        """All 8 D4 transforms produce distinct square permutations."""
        sq_perms = [tuple(sq) for sq, _ in D4_GROUP]
        assert len(set(sq_perms)) == 8

    def test_all_are_bijections(self):
        """Each D4 element is a valid permutation (bijection)."""
        for sq_perm, pol_perm in D4_GROUP:
            assert len(set(sq_perm)) == 64
            assert len(set(pol_perm)) == 4672

    def test_closure_under_composition(self):
        """Composing any two D4 elements yields another D4 element."""
        sq_set = {tuple(sq) for sq, _ in D4_GROUP}
        for sq_a, _ in D4_GROUP:
            for sq_b, _ in D4_GROUP:
                composed = tuple(_compose_sq(sq_a, sq_b))
                assert composed in sq_set, "Composition not in D4 group"


class TestBoardRoundtrip:
    def test_hflip_roundtrip(self):
        """Applying hflip twice returns the original board."""
        rng = np.random.default_rng(42)
        board = rng.random((17, 8, 8)).astype(np.float32)
        policy = rng.random(4672).astype(np.float32)

        b1, p1 = apply_transform(board, policy, HFLIP_SQ, HFLIP_POLICY)
        b2, p2 = apply_transform(b1, p1, HFLIP_SQ, HFLIP_POLICY)

        np.testing.assert_array_almost_equal(b2, board)
        np.testing.assert_array_almost_equal(p2, policy)

    def test_rot90_four_times_roundtrip(self):
        """Applying rot90 four times returns the original."""
        rng = np.random.default_rng(42)
        board = rng.random((17, 8, 8)).astype(np.float32)
        policy = rng.random(4672).astype(np.float32)

        b, p = board, policy
        for _ in range(4):
            b, p = apply_transform(b, p, ROT90_SQ, ROT90_POLICY)

        np.testing.assert_array_almost_equal(b, board)
        np.testing.assert_array_almost_equal(p, policy)

    def test_hflip_known_position(self):
        """Verify piece placement after hflip of a known position."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        # White king on e1 (tensor plane 5, row 7 if white STM, col 4)
        # Using tensor coordinates directly: plane 5, row 6, col 4 = f1 equivalent
        # Place a piece at a known tensor cell
        board[1, 3, 2] = 1.0  # some piece at tensor (3,2) = rank3, file2 = c4

        policy = np.zeros(4672, dtype=np.float32)
        new_board, _ = apply_transform(board, policy, HFLIP_SQ, HFLIP_POLICY)

        # After hflip, (3,2) -> file 7-2=5, so (3,5)
        assert new_board[1, 3, 5] == 1.0
        assert new_board[1, 3, 2] == 0.0

    def test_augment_preserves_material_and_value(self):
        """Material and value are unchanged by augmentation."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[0, 1, 0] = 1.0  # pawn -> hflip eligible
        policy = np.zeros(4672, dtype=np.float32)

        rng = np.random.default_rng(123)
        for _ in range(20):
            _, mat, val, _, _ = augment_sample(board, 3.14, -0.5, policy, rng)
            assert mat == 3.14
            assert val == -0.5


class TestAugmentSampleDistribution:
    def test_hflip_produces_both_transforms(self):
        """Over many calls, augment_sample should produce both identity and hflip."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[0, 1, 3] = 1.0  # pawn on d2 -> hflip eligible
        policy = np.zeros(4672, dtype=np.float32)
        policy[0] = 1.0  # some nonzero entry

        rng = np.random.default_rng(42)
        saw_identity = False
        saw_flipped = False
        for _ in range(100):
            new_b, _, _, new_p, _ = augment_sample(board, 0.0, 0.0, policy, rng)
            if new_b[0, 1, 3] == 1.0:
                saw_identity = True
            if new_b[0, 1, 4] == 1.0:  # d2 hflipped -> e2 (file 3 -> file 4)
                saw_flipped = True
        assert saw_identity and saw_flipped

    def test_d4_produces_multiple_transforms(self):
        """D4 augmentation should produce more than 2 distinct boards."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[1, 2, 3] = 1.0  # knight at tensor (2,3), no pawns, no castling

        policy = np.zeros(4672, dtype=np.float32)
        rng = np.random.default_rng(42)
        boards_seen = set()
        for _ in range(200):
            new_b, _, _, _, _ = augment_sample(board, 0.0, 0.0, policy, rng)
            boards_seen.add(tuple(new_b.ravel()))
        assert len(boards_seen) >= 4  # Should see many of the 8 transforms


class TestAugmentAllTransforms:
    def test_none_symmetry_returns_one(self):
        """Positions with castling return just the original."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[13] = 1.0  # castling
        policy = np.zeros(4672, dtype=np.float32)
        results = augment_all_transforms(board, 0.0, 0.5, policy)
        assert len(results) == 1
        np.testing.assert_array_equal(results[0][0], board)

    def test_hflip_returns_two(self):
        """Positions with hflip symmetry return 2 transforms."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[0, 1, 0] = 1.0  # pawn, no castling
        policy = np.zeros(4672, dtype=np.float32)
        results = augment_all_transforms(board, 0.0, 0.5, policy)
        assert len(results) == 2
        # First is identity
        np.testing.assert_array_equal(results[0][0], board)
        # Second is different
        assert not np.array_equal(results[0][0], results[1][0])

    def test_d4_returns_eight(self):
        """Positions with D4 symmetry return 8 transforms."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[1, 2, 3] = 1.0  # knight, no pawns, no castling
        policy = np.zeros(4672, dtype=np.float32)
        results = augment_all_transforms(board, 0.0, 0.5, policy)
        assert len(results) == 8
        # All distinct
        boards_seen = {tuple(b.ravel()) for b, _, _, _ in results}
        assert len(boards_seen) == 8

    def test_preserves_material_and_value(self):
        """Material and value are unchanged across all transforms."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[0, 1, 0] = 1.0  # pawn -> hflip
        policy = np.zeros(4672, dtype=np.float32)
        results = augment_all_transforms(board, 3.14, -0.5, policy)
        for _, mat, val, _ in results:
            assert mat == 3.14
            assert val == -0.5

    def test_all_policies_are_valid_permutations(self):
        """Each transformed policy sums to the same value as the original."""
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[1, 2, 3] = 1.0  # D4
        policy = np.random.default_rng(42).random(4672).astype(np.float32)
        results = augment_all_transforms(board, 0.0, 0.0, policy)
        original_sum = policy.sum()
        for _, _, _, p in results:
            np.testing.assert_almost_equal(p.sum(), original_sum)


class TestRandomTransform:
    """random_transform returns one equivalent sample, uniform over the group,
    so every position counts once in expectation whatever its symmetry."""

    def _lone_king(self):
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[5, 0, 1] = 1.0  # STM king on b1: its 8 images are distinct squares
        return board

    def test_returns_one_sample(self):
        from augmentation import random_transform
        policy = np.zeros(4672, dtype=np.float32)
        out = random_transform(self._lone_king(), 1.5, -0.25, policy, np.random.default_rng(0))
        assert len(out) == 4
        assert out[0].shape == (17, 8, 8)
        assert out[1] == 1.5 and out[2] == -0.25

    def test_covers_the_whole_d4_group_uniformly(self):
        from augmentation import random_transform
        rng = np.random.default_rng(0)
        policy = np.zeros(4672, dtype=np.float32)
        counts = {}
        n = 4000
        for _ in range(n):
            b, _, _, _ = random_transform(self._lone_king(), 0.0, 0.0, policy, rng)
            sq = int(np.flatnonzero(b[5].ravel())[0])
            counts[sq] = counts.get(sq, 0) + 1
        assert len(counts) == 8
        # Each image should get n/8 = 500; 5 sd of a binomial is about 105.
        assert all(abs(c - n / 8) < 105 for c in counts.values()), counts

    def test_hflip_positions_get_both_images(self):
        from augmentation import random_transform
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[0, 1, 0] = 1.0  # pawn on a2
        rng = np.random.default_rng(0)
        policy = np.zeros(4672, dtype=np.float32)
        files = {int(np.flatnonzero(random_transform(board, 0.0, 0.0, policy, rng)[0][0].ravel())[0])
                 for _ in range(200)}
        assert files == {8, 15}

    def test_castling_positions_are_unchanged(self):
        from augmentation import random_transform
        board = np.zeros((17, 8, 8), dtype=np.float32)
        board[13] = 1.0
        board[5, 0, 4] = 1.0
        policy = np.random.default_rng(1).random(4672).astype(np.float32)
        b, _, _, p = random_transform(board, 0.0, 0.0, policy, np.random.default_rng(0))
        np.testing.assert_array_equal(b, board)
        np.testing.assert_array_equal(p, policy)
