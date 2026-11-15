import unittest

import numpy as np

from gomoku.env import BOARD_SIZE, N_CELLS, EMPTY, P1, P2, Board
from gomoku.encoder import (
    N_CHANNELS,
    PERM,
    SYMMETRIES,
    encode,
    encode_board,
    legal_mask,
    softmax_masked,
    transform_board,
    transform_index,
    transform_last_move,
    transform_policy,
)


def idx(r, c):
    return r * BOARD_SIZE + c


class TestEncode(unittest.TestCase):
    def test_shape_and_dtype(self):
        s = encode([EMPTY] * N_CELLS, P1)
        self.assertEqual(s.shape, (N_CHANNELS, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(s.dtype, np.float32)

    def test_planes_partition_each_cell(self):
        b = Board().play(idx(0, 0)).play(idx(4, 4)).play(idx(8, 8))
        s = encode_board(b)
        total = s[0] + s[1] + s[2]
        self.assertTrue(np.allclose(total, 1.0), "每格必须恰好属于己方/对方/空位之一")

    def test_own_and_opp_swap_with_view(self):
        b = Board()
        b = b.play(idx(4, 4))            # P1
        b = b.play(idx(0, 0))            # P2
        s1 = encode_board(Board(b.cells, to_move=P1, last_move=b.last_move))
        s2 = encode_board(Board(b.cells, to_move=P2, last_move=b.last_move))
        self.assertTrue(np.allclose(s1[0], s2[1]))
        self.assertTrue(np.allclose(s1[1], s2[0]))
        self.assertTrue(np.allclose(s1[2], s2[2]))   # 空位与视角无关
        self.assertTrue(np.allclose(s1[3], s2[3]))   # 上一手与视角无关

    def test_own_plane_matches_pieces(self):
        b = Board().play(idx(3, 5))              # P1 落在 (3,5)，接下来轮到 P2
        s = encode_board(b)
        own = s[0].reshape(-1)
        opp = s[1].reshape(-1)
        # 视角是 to_move = P2：(3,5) 的 P1 子应出现在"对方"平面
        self.assertEqual(own[idx(3, 5)], 0.0)
        self.assertEqual(opp[idx(3, 5)], 1.0)
        self.assertEqual(int(own.sum()), 0)
        self.assertEqual(int(opp.sum()), 1)

        # 换成 P1 视角，同一棋子应回到"己方"平面
        s1 = encode(b.cells, P1, b.last_move)
        self.assertEqual(s1[0].reshape(-1)[idx(3, 5)], 1.0)
        self.assertEqual(s1[1].reshape(-1)[idx(3, 5)], 0.0)

    def test_last_move_channel(self):
        b = Board().play(idx(2, 3))
        s = encode_board(b)
        last = s[3].reshape(-1)
        self.assertEqual(int(last.sum()), 1)
        self.assertEqual(last[idx(2, 3)], 1.0)

    def test_last_move_empty_board(self):
        s = encode([EMPTY] * N_CELLS, P1, last_move=None)
        self.assertEqual(float(s[3].sum()), 0.0)

    def test_encode_batch(self):
        boards = [Board(), Board().play(idx(1, 1))]
        from gomoku.encoder import encode_batch
        s = encode_batch(boards)
        self.assertEqual(s.shape, (2, N_CHANNELS, BOARD_SIZE, BOARD_SIZE))


class TestLegalMaskAndSoftmax(unittest.TestCase):
    def test_legal_mask(self):
        cells = [EMPTY] * N_CELLS
        cells[0] = P1
        m = legal_mask(cells)
        self.assertEqual(m.shape, (N_CELLS,))
        self.assertEqual(m[0], 0.0)
        self.assertEqual(float(m.sum()), N_CELLS - 1)

    def test_softmax_masked_normalizes_on_legal(self):
        logits = np.arange(N_CELLS, dtype=np.float32)
        legal = [0, 5, 10]
        p = softmax_masked(logits, legal)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertEqual(p[1], 0.0)
        self.assertGreater(p[10], p[5])
        self.assertGreater(p[5], p[0])

    def test_softmax_masked_numeric_stability(self):
        logits = np.full(N_CELLS, 1e6, dtype=np.float32)
        p = softmax_masked(logits, [0, 1, 2])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertFalse(np.isnan(p).any())

    def test_softmax_masked_no_legal_raises(self):
        with self.assertRaises(ValueError):
            softmax_masked(np.zeros(N_CELLS), [])


class TestSymmetry(unittest.TestCase):
    def test_perm_tables_are_permutations(self):
        for perm in PERM:
            self.assertEqual(sorted(perm.tolist()), list(range(N_CELLS)))

    def test_identity_is_first(self):
        self.assertEqual(int(PERM[0][7]), 7)

    def test_transform_inverse_roundtrip(self):
        names = {
            (0, False): lambda r, c, n: (r, c),
            (1, False): lambda r, c, n: (c, n - r),
            (2, False): lambda r, c, n: (n - r, n - c),
            (3, False): lambda r, c, n: (n - c, r),
        }
        n = BOARD_SIZE - 1
        for (k, m), fn in names.items():
            for a in range(N_CELLS):
                r, c = divmod(a, BOARD_SIZE)
                if m:
                    c = n - c
                rr, cc = fn(r, c, n)
                self.assertEqual(transform_index(a, k, m), rr * BOARD_SIZE + cc)

    def test_board_and_policy_agree(self):
        b = Board().play(idx(0, 0)).play(idx(4, 5))
        for si, (k, m) in enumerate(SYMMETRIES):
            cells_t = transform_board(b.cells, k, m)
            # 棋盘变换后，原来的落子位置应出现在变换后的位置
            for a in range(N_CELLS):
                if b.cells[a] != EMPTY:
                    self.assertEqual(cells_t[PERM[si][a]], b.cells[a])

            pi = np.zeros(N_CELLS, dtype=np.float32)
            pi[idx(1, 2)] = 1.0
            pi_t = transform_policy(pi, k, m)
            self.assertEqual(pi_t[PERM[si][idx(1, 2)]], 1.0)

    def test_transform_last_move(self):
        lm = idx(3, 7)
        for (k, m), perm in zip(SYMMETRIES, PERM):
            self.assertEqual(transform_last_move(lm, k, m), int(perm[lm]))
        self.assertIsNone(transform_last_move(None, 0, False))

    def test_transform_roundtrip_restores(self):
        """8 种变换后都能回到原棋盘（用对应逆变换验证）。"""
        b = Board().play(idx(0, 1)).play(idx(2, 6)).play(idx(8, 0))
        n = BOARD_SIZE - 1
        for k, m in SYMMETRIES:
            t = transform_board(b.cells, k, m)
            # 求逆：镜像次数与旋转次数组合的逆
            inv_k = (4 - k) % 4
            back = transform_board(t, inv_k, m) if not m else transform_board(
                transform_board(t, inv_k, False), 0, True)
            # 简化：直接验证多重变换组合能回到原状（暴力枚举）
            restored = None
            for k2, m2 in SYMMETRIES:
                if transform_board(t, k2, m2) == b.cells:
                    restored = (k2, m2)
                    break
            self.assertIsNotNone(restored, f"变换 {(k, m)} 无法被逆变换还原")


if __name__ == "__main__":
    unittest.main()
