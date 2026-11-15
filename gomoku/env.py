"""棋盘环境：纯函数落子、终局判定、赢家、合法动作、可视化。

棋盘尺寸由 Board.size 决定（默认 BOARD_SIZE=9，可在 configs/default.yaml 的
`board.size` 配置，例如 15）。所有与尺寸相关的形状都从 cells 长度推导，
不要在其他模块里写死 9 或 81。
"""

from __future__ import annotations

from math import isqrt

BOARD_SIZE = 9  # 默认边长；实际使用请以 Board.size / side_of(cells) 为准
WIN_LEN = 5  # 连珠长度（当前规则固定 5 连获胜）
N_CELLS = BOARD_SIZE * BOARD_SIZE

EMPTY, P1, P2 = 0, 1, 2

_SYMBOL = {EMPTY: ".", P1: "X", P2: "O"}
_PLAYER_NAME = {P1: "P1(X)", P2: "P2(O)"}

# 四个方向：横、竖、主对角、副对角
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def other(player):
    """返回对手。"""
    if player == P1:
        return P2
    if player == P2:
        return P1
    raise ValueError(f"非法玩家编号: {player!r}")


def player_name(player):
    return _PLAYER_NAME.get(player, str(player))


def side_of(cells):
    """从一维棋盘长度推导边长。"""
    n = len(cells)
    s = isqrt(n)
    if s * s != n:
        raise ValueError(f"棋盘长度 {n} 不是完全平方数，无法确定边长")
    return s


def n_cells_for(size):
    return int(size) * int(size)


def _in_bounds(r, c, size):
    return 0 <= r < size and 0 <= c < size


def check_win_at(cells, idx, player):
    """判断在 idx 落 player 后是否形成 >= WIN_LEN 连珠。"""
    size = side_of(cells)
    r, c = divmod(idx, size)
    for dr, dc in _DIRECTIONS:
        count = 1
        nr, nc = r + dr, c + dc
        while _in_bounds(nr, nc, size) and cells[nr * size + nc] == player:
            count += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while _in_bounds(nr, nc, size) and cells[nr * size + nc] == player:
            count += 1
            nr -= dr
            nc -= dc
        if count >= WIN_LEN:
            return True
    return False


def find_winner(cells, last_move=None):
    """若最后一手直接连五，返回赢家；否则 None。"""
    if last_move is None:
        for idx, v in enumerate(cells):
            if v != EMPTY and check_win_at(cells, idx, v):
                return v
        return None
    v = cells[last_move]
    if v == EMPTY:
        return None
    return v if check_win_at(cells, last_move, v) else None


def legal_actions(cells):
    return [i for i, v in enumerate(cells) if v == EMPTY]


def is_full(cells):
    return all(v != EMPTY for v in cells)


class Board:
    """不可变语义的棋盘：play 返回新 Board，自身保持不变。"""

    __slots__ = ("cells", "to_move", "last_move", "winner", "history", "size")

    def __init__(
        self,
        cells=None,
        to_move=P1,
        last_move=None,
        winner=None,
        history=(),
        size=BOARD_SIZE,
    ):
        if cells is not None:
            self.cells = list(cells)
            self.size = side_of(self.cells)
        else:
            self.size = int(size)
            if self.size < 3:
                raise ValueError(f"棋盘尺寸过小: {size}")
            self.cells = [EMPTY] * n_cells_for(self.size)
        self.to_move = to_move
        self.last_move = last_move
        self.winner = winner
        self.history = tuple(history)

    # ---------- 查询 ----------
    def __len__(self):
        return len(self.cells)

    def at(self, r, c):
        return self.cells[r * self.size + c]

    def legal_actions(self):
        return legal_actions(self.cells)

    def is_full(self):
        return is_full(self.cells)

    def is_terminal(self):
        return self.winner is not None or self.is_full()

    def moves_played(self):
        return sum(1 for v in self.cells if v != EMPTY)

    def copy(self):
        return Board(
            self.cells, self.to_move, self.last_move, self.winner, self.history, self.size
        )

    # ---------- 变更 ----------
    def play(self, a):
        """落子，返回新 Board。非法动作抛 ValueError。"""
        if not isinstance(a, int) or not (0 <= a < len(self.cells)):
            raise ValueError(f"非法动作索引: {a!r}（棋盘 {self.size}x{self.size}）")
        if self.cells[a] != EMPTY:
            raise ValueError(f"位置 {a} 已被占用")
        if self.is_terminal():
            raise ValueError("对局已结束，无法继续落子")

        p = self.to_move
        cells = list(self.cells)
        cells[a] = p
        winner = p if check_win_at(cells, a, p) else None
        return Board(cells, other(p), a, winner, self.history + (a,), self.size)

    def undo(self):
        """撤销最后一手，返回新 Board。"""
        if not self.history:
            raise ValueError("没有可撤销的落子")
        cells = list(self.cells)
        a = self.history[-1]
        cells[a] = EMPTY
        history = self.history[:-1]
        last_move = history[-1] if history else None
        winner = find_winner(cells) if history else None
        return Board(cells, other(self.to_move), last_move, winner, history, self.size)

    # ---------- 显示 ----------
    def render(self):
        size = self.size
        lines = ["    " + " ".join(f"{c:2d}" for c in range(size))]
        lines.append("   " + "-" * (size * 3 + 1))
        for r in range(size):
            row = []
            for c in range(size):
                idx = r * size + c
                sym = _SYMBOL[self.cells[idx]]
                if idx == self.last_move:
                    sym = sym.lower() if sym != "." else "*"
                row.append(f"{sym:>2}")
            lines.append(f" {r:2d}| " + " ".join(row))
        lines.append("")
        lines.append("上一个落子用小写/星号标记; 输入格式: 行 列（如 4 4）")
        return "\n".join(lines)

    def __str__(self):
        return self.render()

    def __repr__(self):
        return (
            f"Board({self.size}x{self.size}, to_move={self.to_move}, "
            f"moves={self.moves_played()}, winner={self.winner})"
        )


def board_size_of(cells):
    """兼容旧名：等价于 side_of。"""
    return side_of(cells)
