import unittest

from gomoku.env import (
    EMPTY,
    P1,
    P2,
    Board,
    action_region,
    region_actions,
    region_mask,
)


def idx(size, r, c):
    return r * size + c


class TestActionRegion(unittest.TestCase):
    def test_empty_board_returns_center_block(self):
        cells = [EMPTY] * 225
        r_lo, r_hi, c_lo, c_hi = action_region(cells, margin=2)
        self.assertEqual((r_lo, r_hi, c_lo, c_hi), (5, 9, 5, 9))
        self.assertEqual(len(region_actions(cells, 2)), 25)

    def test_region_is_bounding_box_plus_margin(self):
        size = 15
        cells = [EMPTY] * (size * size)
        cells[idx(size, 4, 3)] = P1
        cells[idx(size, 6, 8)] = P2
        # 外接矩形 rows 4..6, cols 3..8，扩 2 圈 -> rows 2..8, cols 1..10
        self.assertEqual(action_region(cells, 2), (2, 8, 1, 10))
        self.assertEqual(action_region(cells, 0), (4, 6, 3, 8))

    def test_region_clipped_to_board(self):
        size = 15
        cells = [EMPTY] * (size * size)
        cells[idx(size, 0, 0)] = P1
        self.assertEqual(action_region(cells, 5), (0, 5, 0, 5))
        cells2 = [EMPTY] * (size * size)
        cells2[idx(size, 14, 14)] = P2
        self.assertEqual(action_region(cells2, 5), (9, 14, 9, 14))

    def test_margin_never_exceeds_board(self):
        size = 15
        cells = [EMPTY] * (size * size)
        cells[idx(size, 7, 7)] = P1
        r_lo, r_hi, c_lo, c_hi = action_region(cells, 100)
        self.assertEqual((r_lo, r_hi, c_lo, c_hi), (0, 14, 0, 14))

    def test_region_actions_exclude_occupied(self):
        size = 15
        cells = [EMPTY] * (size * size)
        cells[idx(size, 7, 7)] = P1
        acts = region_actions(cells, 1)
        self.assertNotIn(idx(size, 7, 7), acts)
        self.assertEqual(len(acts), 9 - 1)  # 3x3 减 1 个已占

    def test_region_mask(self):
        size = 15
        cells = [EMPTY] * (size * size)
        cells[idx(size, 7, 7)] = P1
        mask = region_mask(cells, 1)
        self.assertEqual(len(mask), size * size)
        self.assertEqual(sum(mask), 9)
        self.assertTrue(mask[idx(size, 7, 7)])

    def test_margin_zero_bounding_box_only(self):
        size = 15
        b = Board(size=size)
        b = b.play(idx(size, 3, 3)).play(idx(size, 5, 5))
        r_lo, r_hi, c_lo, c_hi = action_region(b.cells, 0)
        self.assertEqual((r_lo, r_hi, c_lo, c_hi), (3, 5, 3, 5))


class TestMCTSMargin(unittest.TestCase):
    def test_search_stays_inside_region(self):
        from gomoku.evaluators import HeuristicEvaluator
        from gomoku.mcts import MCTS

        size = 15
        b = Board(size=size)
        for a in [(7, 7), (6, 6), (8, 8)]:
            b = b.play(idx(size, *a))
        m = MCTS(HeuristicEvaluator(), simulations=60, margin=1)
        res = m.search(b.cells, b.to_move, b.last_move)
        r_lo, r_hi, c_lo, c_hi = action_region(b.cells, 1)
        r, c = divmod(res.action, size)
        self.assertTrue(r_lo <= r <= r_hi and c_lo <= c <= c_hi, f"落子 {(r, c)} 越界")

    def test_root_legal_matches_region(self):
        from gomoku.evaluators import HeuristicEvaluator
        from gomoku.mcts import MCTS

        size = 15
        b = Board(size=size).play(idx(size, 7, 7))
        m = MCTS(HeuristicEvaluator(), simulations=10, margin=2)
        res = m.search(b.cells, b.to_move, b.last_move)
        self.assertEqual(sorted(res.root.legal), sorted(region_actions(b.cells, 2)))


class TestSharpenPi(unittest.TestCase):
    def test_sharpen_removes_tail(self):
        import numpy as np

        from gomoku.selfplay import sharpen_pi

        pi = np.zeros(225, dtype=np.float32)
        # 500 次模拟：3 个大头 + 200 个各 1 次
        pi[0], pi[1], pi[2] = 300 / 500, 100 / 500, 90 / 500
        for a in range(3, 203):
            pi[a] = 1 / 500
        pi /= pi.sum()
        out = sharpen_pi(pi, min_visits=5, total_visits=500)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        self.assertGreater(out[0], pi[0])
        self.assertEqual(out[203], 0.0)
        self.assertEqual(int((out > 0).sum()), 3)

    def test_sharpen_keeps_pi_when_nothing_survives(self):
        import numpy as np

        from gomoku.selfplay import sharpen_pi

        pi = np.zeros(225, dtype=np.float32)
        pi[0:10] = 0.1
        out = sharpen_pi(pi, min_visits=100, total_visits=10)
        self.assertTrue(np.allclose(out, pi))


class TestSelfPlayOptions(unittest.TestCase):
    def test_opening_moves_and_pi_sharpen_smoke(self):
        try:
            from gomoku.network import build_network
            from gomoku.selfplay import SelfPlayConfig, play_selfplay
        except ImportError:  # pragma: no cover
            self.skipTest("需要 torch")

        net = build_network(channels=8, blocks=1, size=15).eval()
        cfg = SelfPlayConfig(
            simulations=3,
            size=15,
            opening_moves=3,
            opening_max=5,
            pi_min_visits=1,
            search_margin=2,
        )
        records, winner, steps = play_selfplay(net, cfg=cfg, opponent="self", seed=1)
        self.assertGreater(steps, 0)
        if records:
            for rec in records:
                self.assertAlmostEqual(float(rec.pi.sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
