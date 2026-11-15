import unittest

from gomoku.env import (
    BOARD_SIZE,
    N_CELLS,
    EMPTY,
    P1,
    P2,
    Board,
    check_win_at,
    find_winner,
    is_full,
    legal_actions,
    other,
)


def idx(r, c):
    return r * BOARD_SIZE + c


class TestOther(unittest.TestCase):
    def test_other(self):
        self.assertEqual(other(P1), P2)
        self.assertEqual(other(P2), P1)
        with self.assertRaises(ValueError):
            other(EMPTY)


class TestWinDetection(unittest.TestCase):
    def test_horizontal(self):
        cells = [EMPTY] * N_CELLS
        for c in range(4):
            cells[idx(3, c)] = P1
        cells[idx(3, 4)] = P1
        self.assertTrue(check_win_at(cells, idx(3, 4), P1))
        self.assertEqual(find_winner(cells, idx(3, 4)), P1)

    def test_vertical(self):
        cells = [EMPTY] * N_CELLS
        for r in range(4):
            cells[idx(r, 2)] = P2
        cells[idx(4, 2)] = P2
        self.assertTrue(check_win_at(cells, idx(4, 2), P2))

    def test_diagonal(self):
        cells = [EMPTY] * N_CELLS
        for i in range(4):
            cells[idx(i, i)] = P1
        cells[idx(4, 4)] = P1
        self.assertTrue(check_win_at(cells, idx(4, 4), P1))

    def test_anti_diagonal(self):
        cells = [EMPTY] * N_CELLS
        for i in range(4):
            cells[idx(i, 8 - i)] = P2
        cells[idx(4, 4)] = P2
        self.assertTrue(check_win_at(cells, idx(4, 4), P2))

    def test_no_wrap_across_edges(self):
        """行末到行首不应被当成连珠。"""
        cells = [EMPTY] * N_CELLS
        # 第 1 行末尾两个 + 第 2 行开头两个（一维相邻，但二维不连续）
        cells[idx(1, 7)] = P1
        cells[idx(1, 8)] = P1
        cells[idx(2, 0)] = P1
        cells[idx(2, 1)] = P1
        cells[idx(2, 2)] = P1
        self.assertFalse(check_win_at(cells, idx(2, 2), P1))

    def test_four_is_not_win(self):
        cells = [EMPTY] * N_CELLS
        for c in range(4):
            cells[idx(0, c)] = P1
        self.assertFalse(check_win_at(cells, idx(0, 3), P1))

    def test_five_at_border(self):
        cells = [EMPTY] * N_CELLS
        for c in range(4, 9):
            cells[idx(8, c)] = P2
        self.assertTrue(check_win_at(cells, idx(8, 8), P2))

    def test_winner_not_emitted_for_empty_last(self):
        cells = [EMPTY] * N_CELLS
        for c in range(5):
            cells[idx(0, c)] = P1
        self.assertIsNone(find_winner(cells, idx(1, 1)))


class TestBoardPlay(unittest.TestCase):
    def test_play_returns_new_board(self):
        b = Board()
        nb = b.play(idx(4, 4))
        self.assertEqual(b.cells[idx(4, 4)], EMPTY)  # 原棋盘不变
        self.assertEqual(nb.cells[idx(4, 4)], P1)
        self.assertEqual(nb.to_move, P2)
        self.assertEqual(nb.last_move, idx(4, 4))

    def test_turn_alternates(self):
        b = Board()
        b = b.play(idx(0, 0))
        b = b.play(idx(0, 1))
        self.assertEqual(b.to_move, P1)
        self.assertEqual(b.cells[idx(0, 1)], P2)

    def test_illegal_move_occupied(self):
        b = Board().play(idx(4, 4))
        with self.assertRaises(ValueError):
            b.play(idx(4, 4))

    def test_illegal_move_out_of_range(self):
        b = Board()
        with self.assertRaises(ValueError):
            b.play(N_CELLS)
        with self.assertRaises(ValueError):
            b.play(-1)

    def test_play_after_terminal(self):
        b = Board()
        for c in range(5):
            b = b.play(idx(0, c))       # P1
            if c < 4:
                b = b.play(idx(5, c))   # P2
        self.assertEqual(b.winner, P1)
        self.assertTrue(b.is_terminal())
        with self.assertRaises(ValueError):
            b.play(idx(8, 8))

    def test_undo(self):
        b = Board()
        b = b.play(idx(4, 4))               # P1 走，轮到 P2
        b2 = b.play(idx(4, 5))              # P2 走，轮到 P1
        b3 = b2.undo()                      # 撤销 P2 那一手
        self.assertEqual(b3.cells[idx(4, 5)], EMPTY)
        self.assertEqual(b3.cells[idx(4, 4)], P1)
        self.assertEqual(b3.to_move, P2)    # 撤销后应由被撤销的一方重新走
        self.assertIsNone(b3.winner)

    def test_undo_empty_raises(self):
        with self.assertRaises(ValueError):
            Board().undo()

    def test_legal_actions(self):
        b = Board().play(idx(0, 0)).play(idx(0, 1))
        moves = b.legal_actions()
        self.assertEqual(len(moves), N_CELLS - 2)
        self.assertNotIn(idx(0, 0), moves)
        self.assertNotIn(idx(0, 1), moves)
        self.assertEqual(legal_actions(b.cells), moves)

    def test_is_full_draw(self):
        cells = [P1 if i % 2 == 0 else P2 for i in range(N_CELLS)]
        self.assertTrue(is_full(cells))
        # 行内交替不会产生五连，但为求稳妥只断言"满盘"与"无合法动作"
        self.assertEqual(legal_actions(cells), [])

    def test_render_contains_symbols(self):
        b = Board().play(idx(4, 4))
        text = b.render()
        self.assertIn("x", text.lower())
        self.assertEqual(text.count("\n") >= BOARD_SIZE, True)


if __name__ == "__main__":
    unittest.main()
