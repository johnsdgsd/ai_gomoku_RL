import unittest

import numpy as np

from gomoku.env import BOARD_SIZE, EMPTY, N_CELLS, P1, P2
from gomoku.evaluators import (
    HeuristicEvaluator,
    candidate_moves,
    find_win_moves,
    potential,
    shape_score,
)


def idx(r, c):
    return r * BOARD_SIZE + c


def make(pieces):
    cells = [EMPTY] * N_CELLS
    for a, p in pieces.items():
        cells[a] = p
    return cells


class TestShapeScore(unittest.TestCase):
    def test_five_scores_highest(self):
        five = make({idx(0, c): P1 for c in range(5)})
        self.assertGreater(shape_score(five, P1), shape_score(five, P2))

    def test_open_four_beats_four(self):
        open_four = make({idx(2, c): P1 for c in range(1, 5)})
        blocked_four = make({idx(2, 0): P2, **{idx(2, c): P1 for c in range(1, 5)}})
        self.assertGreater(shape_score(open_four, P1), shape_score(blocked_four, P1))

    def test_open_three_beats_sleep_three(self):
        open_three = make({idx(4, c): P1 for c in range(3, 6)})
        sleep_three = make({idx(4, 2): P2, **{idx(4, c): P1 for c in range(3, 6)}})
        self.assertGreater(shape_score(open_three, P1), shape_score(sleep_three, P1))

    def test_open_three_beats_open_two(self):
        open_three = make({idx(4, c): P1 for c in range(3, 6)})
        open_two = make({idx(4, c): P1 for c in range(3, 5)})
        self.assertGreater(shape_score(open_three, P1), shape_score(open_two, P1))

    def test_no_pattern_on_empty_board(self):
        self.assertEqual(shape_score([EMPTY] * N_CELLS, P1), 0)

    def test_potential_sign(self):
        cells = make({idx(0, c): P1 for c in range(3)})  # 仅 P1 有三连
        self.assertGreater(potential(cells, P1), 0)
        self.assertLess(potential(cells, P2), 0)

    def test_opponent_stone_is_wall(self):
        """对手子与墙等价：被封住的四不应算活四。"""
        blocked = make({idx(2, 0): P2, **{idx(2, c): P1 for c in range(1, 5)}})
        walled = make({idx(2, c): P1 for c in range(0, 4)})  # 左端是棋盘边界
        self.assertEqual(shape_score(blocked, P1), shape_score(walled, P1))


class TestThreatDetection(unittest.TestCase):
    def test_find_win_moves(self):
        cells = make({idx(0, c): P1 for c in range(4)})
        self.assertEqual(find_win_moves(cells, P1), [idx(0, 4)])

    def test_broken_four_win_points(self):
        # X X _ X X -> 缺口可成五（两端也可能成五，这里只验证缺口被找到）
        cells = make(
            {idx(5, 1): P1, idx(5, 2): P1, idx(5, 4): P1, idx(5, 5): P1}
        )
        wins = find_win_moves(cells, P1)
        self.assertIn(idx(5, 3), wins)

    def test_no_win_moves_for_three(self):
        cells = make({idx(0, c): P1 for c in range(3)})
        self.assertEqual(find_win_moves(cells, P1), [])

    def test_candidates_near_stones_only(self):
        cells = make({idx(4, 4): P1})
        cands = candidate_moves(cells, radius=1)
        self.assertIn(idx(3, 3), cands)
        self.assertNotIn(idx(4, 4), cands)
        self.assertNotIn(idx(0, 0), cands)
        self.assertTrue(all(cells[a] == EMPTY for a in cands))

    def test_candidates_empty_board_center(self):
        cands = candidate_moves([EMPTY] * N_CELLS)
        self.assertEqual(cands, [idx(4, 4)])


class TestHeuristicEvaluator(unittest.TestCase):
    def test_prior_is_distribution_and_masks_illegal(self):
        cells = make({idx(4, 4): P1})
        ev = HeuristicEvaluator()
        probs, value = ev.evaluate(cells, P2, last_move=idx(4, 4))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(probs[idx(4, 4)], 0.0)
        self.assertTrue(-1.0 <= value <= 1.0)

    def test_winning_move_gets_top_prior(self):
        cells = make({idx(0, c): P1 for c in range(4)})
        ev = HeuristicEvaluator()
        probs, _ = ev.evaluate(cells, P1, last_move=idx(0, 3))
        self.assertEqual(int(np.argmax(probs)), idx(0, 4))

    def test_blocking_move_outranks_random_move(self):
        cells = make({idx(5, c): P2 for c in range(4)})
        ev = HeuristicEvaluator()
        probs, _ = ev.evaluate(cells, P1, last_move=idx(5, 3))
        self.assertGreater(probs[idx(5, 4)], probs[idx(0, 0)])

    def test_value_perspective_flips(self):
        cells = make({idx(0, c): P1 for c in range(3)})
        ev = HeuristicEvaluator()
        _, v_p1 = ev.evaluate(cells, P1)
        _, v_p2 = ev.evaluate(cells, P2)
        self.assertGreater(v_p1, 0)
        self.assertLess(v_p2, 0)


if __name__ == "__main__":
    unittest.main()
