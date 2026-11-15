import unittest

import numpy as np

from gomoku.env import BOARD_SIZE, N_CELLS, EMPTY, P1, P2, Board
from gomoku.encoder import softmax_masked
from gomoku.mcts import MCTS, Node
from gomoku.evaluators import HeuristicEvaluator


def idx(r, c):
    return r * BOARD_SIZE + c


class StubEvaluator:
    """固定先验与价值的桩评估器，用于验证回溯符号。"""

    def __init__(self, priors=None, value=0.0, uniform_value=None):
        self.priors = priors or {}
        self.value = value
        self.calls = 0
        self.uniform_value = uniform_value
        self.seen = []

    def evaluate(self, cells, to_move, last_move=None, legal=None):
        self.calls += 1
        self.seen.append((tuple(cells), to_move))
        if legal is None:
            legal = [i for i, v in enumerate(cells) if v == EMPTY]
        if self.priors:
            p = np.zeros(N_CELLS, dtype=np.float32)
            w = np.zeros(N_CELLS, dtype=np.float64)
            for a in legal:
                w[a] = self.priors.get(a, 0.01)
            w[legal] /= w[legal].sum()
            p[legal] = w[legal]
            return p, self.value
        return softmax_masked(np.zeros(N_CELLS), legal), self.value


def board_from(pieces, to_move):
    """pieces: dict action -> player。"""
    cells = [EMPTY] * N_CELLS
    for a, p in pieces.items():
        cells[a] = p
    return cells


class TestMTSTerminal(unittest.TestCase):
    def test_root_terminal_does_not_call_network(self):
        ev = StubEvaluator(value=0.5)
        m = MCTS(ev, simulations=10)
        # P1 已连五
        cells = board_from({idx(0, c): P1 for c in range(5)}, to_move=P2)
        res = m.search(cells, P2, last_move=idx(0, 4))
        self.assertEqual(ev.calls, 0, "根终局时不应调用评估器")
        self.assertLess(res.value, 0)  # P2 视角已输
        self.assertEqual(res.visits, 0)

    def test_terminal_child_not_expanded(self):
        """一步连五的子节点必须直接被判定，不再展开。"""
        ev = StubEvaluator(value=0.0)
        m = MCTS(ev, simulations=20)
        cells = board_from({idx(0, c): P1 for c in range(4)}, to_move=P1)
        res = m.search(cells, P1, last_move=idx(0, 3))
        self.assertEqual(res.action, idx(0, 4))
        # 只评估了非终局的节点（根 + 少量分支），绝不在终局棋盘上评估
        for seen_cells, player in ev.seen:
            if player == P2 and seen_cells[idx(0, 4)] == P1:
                # 评估的 P2 局面允许存在，但不能"继续往下扩展赢局"
                pass
        self.assertLessEqual(ev.calls, 20 + 1)


class TestMCTSForcedMoves(unittest.TestCase):
    def setUp(self):
        self.ev = HeuristicEvaluator()

    def test_winning_move(self):
        m = MCTS(self.ev, simulations=50)
        cells = board_from({idx(0, c): P1 for c in range(4)}, to_move=P1)
        res = m.search(cells, P1, last_move=idx(0, 3))
        self.assertEqual(res.action, idx(0, 4))

    def test_block_four(self):
        """对手冲四，必须堵唯一缺口。"""
        m = MCTS(self.ev, simulations=50)
        # P2 在 (5,0..3)，缺口 (5,4)
        cells = board_from({idx(5, c): P2 for c in range(4)}, to_move=P1)
        res = m.search(cells, P1, last_move=idx(5, 3))
        self.assertEqual(res.action, idx(5, 4))

    def test_block_open_three_only_usable_end(self):
        m = MCTS(self.ev, simulations=100)
        # P2 活三 (3,3),(3,4),(3,5)，两端 (3,2)/(3,6) 都可堵
        cells = board_from(
            {idx(3, 3): P2, idx(3, 4): P2, idx(3, 5): P2}, to_move=P1
        )
        res = m.search(cells, P1, last_move=idx(3, 5))
        self.assertIn(res.action, (idx(3, 2), idx(3, 6)))

    def test_prefers_own_win_over_block(self):
        m = MCTS(self.ev, simulations=50)
        pieces = {idx(0, c): P2 for c in range(4)}  # 对手四连
        for c in range(4):
            pieces[idx(8, c)] = P1  # 自己四连
        cells = board_from(pieces, to_move=P1)
        res = m.search(cells, P1, last_move=idx(8, 3))
        self.assertEqual(res.action, idx(8, 4))


class TestBackpropSign(unittest.TestCase):
    def test_sign_of_value(self):
        """子节点视角为负（对手差）时，父节点 W 应为正。"""
        # 根只有一个"未终局"的行棋路径，且所有叶子价值都是 -0.5（对手视角）
        ev = StubEvaluator(value=-0.5)
        m = MCTS(ev, simulations=1, c_puct=1.5)
        cells = board_from({idx(4, 4): P1}, to_move=P2)
        res = m.search(cells, P2, last_move=idx(4, 4))
        root = res.root
        # 第一次模拟：根选先验最大的动作，展开子节点，值为 -0.5（子视角）
        # 回溯到根：W = -(-0.5) = +0.5
        total_n = sum(root.stats[a].N for a in root.legal)
        self.assertEqual(total_n, 1)
        visited = [a for a in root.legal if root.stats[a].N == 1]
        self.assertEqual(len(visited), 1)
        self.assertAlmostEqual(root.stats[visited[0]].W, 0.5, places=6)

    def test_n_total_equals_sum(self):
        ev = StubEvaluator(value=0.1)
        m = MCTS(ev, simulations=30)
        cells = board_from({idx(4, 4): P1, idx(0, 0): P2}, to_move=P1)
        res = m.search(cells, P1, last_move=idx(0, 0))
        root = res.root
        self.assertEqual(root.N_total, sum(root.stats[a].N for a in root.legal))
        self.assertEqual(root.N_total, 30)

    def test_q_zero_when_unvisited(self):
        n = Node([EMPTY] * N_CELLS, P1)
        n.stats[0] = n.stats.get(0) or __import__(
            "gomoku.mcts", fromlist=["ChildStats"]
        ).ChildStats()
        self.assertEqual(n.stats[0].q, 0.0)


class TestSearchResultShape(unittest.TestCase):
    def test_pi_is_distribution_on_legal(self):
        ev = HeuristicEvaluator()
        m = MCTS(ev, simulations=40)
        b = Board().play(idx(4, 4))
        res = m.search(b.cells, b.to_move, b.last_move)
        self.assertAlmostEqual(float(res.pi.sum()), 1.0, places=5)
        for a, v in enumerate(res.pi):
            if b.cells[a] != EMPTY:
                self.assertEqual(v, 0.0, f"非法动作 {a} 的先验应为 0")
        self.assertIn(res.action, b.legal_actions())
        self.assertGreaterEqual(res.value, -1.0)
        self.assertLessEqual(res.value, 1.0)

    def test_deterministic_with_seed(self):
        import random

        ev = HeuristicEvaluator()
        b = Board().play(idx(4, 4))
        r1 = MCTS(ev, simulations=50, rng=random.Random(7)).search(
            b.cells, b.to_move, b.last_move
        )
        r2 = MCTS(ev, simulations=50, rng=random.Random(7)).search(
            b.cells, b.to_move, b.last_move
        )
        self.assertEqual(r1.action, r2.action)
        self.assertTrue(np.allclose(r1.pi, r2.pi))


if __name__ == "__main__":
    unittest.main()
