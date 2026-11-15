"""局面评估器：为 MCTS 提供 (动作先验, 局面价值)。

统一接口：
    evaluate(cells, to_move, last_move, legal) -> (probs[N], value)
    - probs: 与棋盘等长，只在合法动作上非零且和为 1
    - value: [-1, 1]，以 to_move 视角

尺寸从 cells 长度推导，支持 9x9 / 13x13 / 15x15 等任意 N×N。

实现：
    NetEvaluator        神经网络（AlphaZero 式）
    HeuristicEvaluator  棋型启发式，无模型、可解释、用于冷启动、对手池和回归基线
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .env import (
    BOARD_SIZE,
    EMPTY,
    WIN_LEN,
    check_win_at,
    other,
    side_of,
)
from .encoder import softmax_masked


# ============================================================
# 线表：横 / 竖 / 两斜，长度 >= WIN_LEN（按尺寸缓存）
# ============================================================
_lines_cache: dict[int, tuple] = {}


def _build_lines(size):
    B, W = size, WIN_LEN
    lines = []
    for r in range(B):
        lines.append([r * B + c for c in range(B)])
    for c in range(B):
        lines.append([r * B + c for r in range(B)])
    for start_c in range(B):
        line, r, c = [], 0, start_c
        while r < B and c < B:
            line.append(r * B + c)
            r += 1
            c += 1
        if len(line) >= W:
            lines.append(line)
    for start_r in range(1, B):
        line, r, c = [], start_r, 0
        while r < B and c < B:
            line.append(r * B + c)
            r += 1
            c += 1
        if len(line) >= W:
            lines.append(line)
    for start_c in range(B):
        line, r, c = [], 0, start_c
        while r < B and c >= 0:
            line.append(r * B + c)
            r += 1
            c -= 1
        if len(line) >= W:
            lines.append(line)
    for start_r in range(1, B):
        line, r, c = [], start_r, B - 1
        while r < B and c >= 0:
            line.append(r * B + c)
            r += 1
            c -= 1
        if len(line) >= W:
            lines.append(line)
    return lines


def lines_for(size):
    """返回 (lines, cell_lines)，按尺寸缓存。"""
    size = int(size)
    if size not in _lines_cache:
        lines = tuple(tuple(line) for line in _build_lines(size))
        cell_lines = [[] for _ in range(size * size)]
        for li, line in enumerate(lines):
            for a in line:
                cell_lines[a].append(li)
        _lines_cache[size] = (lines, tuple(tuple(x) for x in cell_lines))
    return _lines_cache[size]


# 兼容默认尺寸
ALL_LINES, CELL_LINES = lines_for(BOARD_SIZE)


# ============================================================
# 棋型评分（固定 5 连的棋型表）
# ============================================================
SCORE_FIVE = 10_000_000
SCORE_OPEN_FOUR = 100_000
SCORE_FOUR = 10_000
SCORE_OPEN_THREE = 1_000
SCORE_SLEEP_THREE = 100
SCORE_OPEN_TWO = 50
SCORE_SLEEP_TWO = 10

_PATTERNS = (
    ("11111", SCORE_FIVE),
    ("011110", SCORE_OPEN_FOUR),
    ("211110", SCORE_FOUR),
    ("011112", SCORE_FOUR),
    ("10111", SCORE_FOUR),
    ("11011", SCORE_FOUR),
    ("11101", SCORE_FOUR),
    ("011100", SCORE_OPEN_THREE),
    ("001110", SCORE_OPEN_THREE),
    ("010110", SCORE_OPEN_THREE),
    ("011010", SCORE_OPEN_THREE),
    ("211100", SCORE_SLEEP_THREE),
    ("001112", SCORE_SLEEP_THREE),
    ("010112", SCORE_SLEEP_THREE),
    ("211010", SCORE_SLEEP_THREE),
    ("011012", SCORE_SLEEP_THREE),
    ("210110", SCORE_SLEEP_THREE),
    ("10011", SCORE_SLEEP_THREE),
    ("11001", SCORE_SLEEP_THREE),
    ("10101", SCORE_SLEEP_THREE),
    ("001100", SCORE_OPEN_TWO),
    ("010100", SCORE_OPEN_TWO),
    ("001010", SCORE_OPEN_TWO),
    ("011000", SCORE_OPEN_TWO),
    ("000110", SCORE_OPEN_TWO),
    ("000112", SCORE_SLEEP_TWO),
    ("211000", SCORE_SLEEP_TWO),
    ("10001", SCORE_SLEEP_TWO),
)


@lru_cache(maxsize=500_000)
def _score_pattern_string(s: str) -> int:
    """对单条线的字符串评分：1=己方、0=空、2=对方或墙。

    命中棋型后用**等长**掩码覆盖，避免重叠重复计数与坐标错位。
    """
    total = 0
    for pat, score in _PATTERNS:
        cnt = s.count(pat)
        if cnt:
            total += cnt * score
            s = s.replace(pat, "2" * len(pat))
    return total


def _line_string(cells, line, player):
    opp = other(player)
    parts = ["2"]
    for idx in line:
        v = cells[idx]
        if v == EMPTY:
            parts.append("0")
        elif v == player:
            parts.append("1")
        else:
            parts.append("2")
    parts.append("2")
    return "".join(parts)


def shape_score(cells, player):
    """player 在整盘上的棋型总分（尺寸自适应）。"""
    lines = lines_for(side_of(cells))[0]
    return sum(_score_pattern_string(_line_string(cells, line, player)) for line in lines)


def potential(cells, me):
    """势函数 Φ = 己方棋型分 - 对手棋型分。"""
    return shape_score(cells, me) - shape_score(cells, other(me))


def find_win_moves(cells, player, candidates=None):
    """player 下一步能直接连五的点。

    candidates 可传入候选点集合：一步成五的点必然紧邻已有棋子，不会漏判。
    """
    moves = candidates if candidates is not None else range(len(cells))
    wins = []
    for a in moves:
        if cells[a] != EMPTY:
            continue
        cells[a] = player
        if check_win_at(cells, a, player):
            wins.append(a)
        cells[a] = EMPTY
    return wins


def candidate_moves(cells, radius=2):
    """棋子附近 radius 半径内的空位；空盘返回中心（尺寸自适应）。"""
    size = side_of(cells)
    occupied = [i for i, v in enumerate(cells) if v != EMPTY]
    if not occupied:
        center = size // 2
        return [center * size + center]
    out, seen = [], set()
    for idx in occupied:
        r0, c0 = divmod(idx, size)
        for r in range(max(0, r0 - radius), min(size, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(size, c0 + radius + 1)):
                a = r * size + c
                if a not in seen and cells[a] == EMPTY:
                    seen.add(a)
                    out.append(a)
    if not out:
        out = [i for i, v in enumerate(cells) if v == EMPTY]
    return out


# ============================================================
# 启发式评估器
# ============================================================
class HeuristicEvaluator:
    """棋型启发式：先验 = 落子势差 softmax；价值 = 势差 tanh。"""

    # 势差 -> 价值 的压缩尺度（四连 10000、活三 1000 量级）
    VALUE_SCALE = 3000.0
    # 势差 -> 先验 logit 的温度
    PRIOR_TEMP = 500.0
    # 非候选点的 logit 相对下界（保证仍有极小先验，不被完全排除）
    FLOOR_GAP = 8.0

    def __init__(self, radius=2):
        self.radius = radius

    def _delta(self, cells, a, player, cell_lines):
        """在 a 落 player 的子后，player 视角的势差（只看经过 a 的线）。"""
        opp = other(player)
        lines, _ = lines_for(side_of(cells))
        delta = 0
        for li in cell_lines[a]:
            line = lines[li]
            before_me = _score_pattern_string(_line_string(cells, line, player))
            before_opp = _score_pattern_string(_line_string(cells, line, opp))
            cells[a] = player
            after_me = _score_pattern_string(_line_string(cells, line, player))
            after_opp = _score_pattern_string(_line_string(cells, line, opp))
            cells[a] = EMPTY
            delta += (after_me - before_me) - (after_opp - before_opp)
        return delta

    def evaluate(self, cells, to_move, last_move=None, legal=None):
        if legal is None:
            legal = [i for i, v in enumerate(cells) if v == EMPTY]
        if not legal:
            raise ValueError("没有合法动作")

        n = len(cells)
        size = side_of(cells)
        lines, cell_lines = lines_for(size)
        opp = other(to_move)
        cand = set(candidate_moves(cells, self.radius))
        logits = np.full(n, -1e9, dtype=np.float64)
        cand_list = [a for a in legal if a in cand]

        # 每条相关线的"落子前"棋型分：同一局面只算一次
        lines_needed = set()
        for a in cand_list:
            lines_needed.update(cell_lines[a])
        before = {
            p: {
                li: _score_pattern_string(_line_string(cells, lines[li], p))
                for li in lines_needed
            }
            for p in (to_move, opp)
        }

        deltas = {}
        for a in cand_list:
            delta = 0
            for li in cell_lines[a]:
                line = lines[li]
                cells[a] = to_move
                after_me = _score_pattern_string(_line_string(cells, line, to_move))
                after_opp = _score_pattern_string(_line_string(cells, line, opp))
                cells[a] = EMPTY
                delta += (after_me - before[to_move][li]) - (
                    after_opp - before[opp][li]
                )
            deltas[a] = delta
        if not deltas:
            for a in legal:
                deltas[a] = 0.0

        hi = max(deltas.values())
        top_logit = hi / self.PRIOR_TEMP
        floor = top_logit - self.FLOOR_GAP
        for a in legal:
            logits[a] = deltas[a] / self.PRIOR_TEMP if a in deltas else floor
        # 必胜点置顶：必须高于"由它自己 delta 得到的最高分"
        for a in find_win_moves(cells, to_move, cand_list):
            logits[a] = top_logit + 1.0

        phi = potential(cells, to_move)
        value = float(np.tanh(phi / self.VALUE_SCALE))
        return softmax_masked(logits, legal), value


# ============================================================
# 神经网络评估器
# ============================================================
class NetEvaluator:
    """神经网络评估器（AlphaZero 式）。"""

    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def evaluate(self, cells, to_move, last_move=None, legal=None):
        import torch

        from .encoder import encode

        if legal is None:
            legal = [i for i, v in enumerate(cells) if v == EMPTY]
        if not legal:
            raise ValueError("没有合法动作")

        s = encode(cells, to_move, last_move)
        x = torch.from_numpy(s).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits.squeeze(0).float().cpu().numpy()
        v = float(value.squeeze().item())
        return softmax_masked(logits, legal), v
