import unittest

from gomoku.env import N_CELLS, P1, P2, Board
from gomoku.selfplay import HeuristicPlayer, make_opponent

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def idx(size, r, c):
    return r * size + c


class TestHeuristicPlayer(unittest.TestCase):
    def test_returns_legal_move(self):
        import random

        b = Board(size=15).play(idx(15, 7, 7))
        p = HeuristicPlayer(random.Random(0))
        a = p.select(b)
        self.assertIn(a, b.legal_actions())

    def test_takes_winning_move(self):
        import random

        size = 15
        b = Board(size=size)
        # P1 四连，接下来轮到 P2；用镜像再测一次轮到 P1 能赢
        cells = list(b.cells)
        for c in range(4):
            cells[idx(size, 3, c)] = P1
        b = Board(cells=cells, to_move=P1, last_move=idx(size, 3, 3))
        p = HeuristicPlayer(random.Random(0))
        self.assertEqual(p.select(b), idx(size, 3, 4))

    def test_blocks_opponent_four(self):
        import random

        size = 15
        cells = [0] * (size * size)
        for c in range(4):
            cells[idx(size, 5, c)] = P2
        b = Board(cells=cells, to_move=P1, last_move=idx(size, 5, 3))
        p = HeuristicPlayer(random.Random(0), noise=0.0)
        self.assertEqual(p.select(b), idx(size, 5, 4))

    def test_opening_near_center(self):
        """空盘开局应下在中心附近，而不是边缘。"""
        import random

        b = Board(size=15)
        p = HeuristicPlayer(random.Random(0), noise=0.0)
        a = p.select(b)
        r, c = divmod(a, 15)
        self.assertLess(abs(r - 7), 3)
        self.assertLess(abs(c - 7), 3)


class TestOpponentFactory(unittest.TestCase):
    def test_self_is_none(self):
        import random

        self.assertIsNone(make_opponent("self", random.Random(0)))

    def test_heuristic_type(self):
        import random

        p = make_opponent("heuristic", random.Random(0))
        self.assertIsInstance(p, HeuristicPlayer)

    def test_unknown_raises(self):
        import random

        with self.assertRaises(ValueError):
            make_opponent("nope", random.Random(0))


class TestOpponentSchedule(unittest.TestCase):
    def test_warmup_then_self(self):
        import random

        from gomoku.train import TrainConfig, _opponent_for_iteration

        cfg = TrainConfig(heuristic_warmup_iters=50)
        rng = random.Random(0)
        self.assertEqual(_opponent_for_iteration(cfg, 1, rng), "heuristic")
        self.assertEqual(_opponent_for_iteration(cfg, 50, rng), "heuristic")
        self.assertEqual(_opponent_for_iteration(cfg, 51, rng), "self")

    def test_zero_warmup_is_pure_selfplay(self):
        import random

        from gomoku.train import TrainConfig, _opponent_for_iteration

        cfg = TrainConfig(heuristic_warmup_iters=0)
        rng = random.Random(0)
        self.assertEqual(_opponent_for_iteration(cfg, 1, rng), "self")


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestEvaluateVsHeuristic(unittest.TestCase):
    def test_evaluate_returns_heuristic_key(self):
        from gomoku.network import build_network
        from gomoku.train import evaluate

        net = build_network(channels=8, blocks=1, size=15).eval()
        scores = evaluate(
            net, "cpu", games=2, sims=2, seed=0, vs_heuristic_sims=2
        )
        self.assertIn("heuristic", scores)
        self.assertGreaterEqual(scores["heuristic"], 0.0)
        self.assertLessEqual(scores["heuristic"], 1.0)


if __name__ == "__main__":
    unittest.main()
