"""状态编码与对称变换。

输入编码 encode(cells, to_move) -> float32[4, size, size]：
  平面 0: 己方（to_move）子
  平面 1: 对方子
  平面 2: 空位
  平面 3: 上一手落点（空盘/无上一手时全 0）

尺寸从 cells / pi 的长度推导，不写死 9。
8 种对称变换（旋转 0/90/180/270 × 是否镜像）按尺寸缓存置换表，
棋盘、动作、策略分布、上一手共用同一张表。
"""

from __future__ import annotations

import numpy as np

from .env import BOARD_SIZE, EMPTY, side_of

N_CHANNELS = 4

# 8 种对称：(rot_k, mirror)，rot_k 为逆时针旋转 90 度的次数
SYMMETRIES = tuple((k, mirror) for mirror in (False, True) for k in range(4))

_perm_cache: dict[int, tuple] = {}


def _sym_index(rot_k, mirror):
    return SYMMETRIES.index((rot_k % 4, bool(mirror)))


def transform_index(a, rot_k, mirror, size=BOARD_SIZE):
    """把动作索引 a 经过"先镜像再旋转"变换为新索引。"""
    r, c = divmod(a, size)
    n = size - 1
    if mirror:
        c = n - c
    for _ in range(rot_k % 4):
        r, c = c, n - r
    return r * size + c


def perm_tables(size):
    """返回该尺寸下 8 张置换表：PERM[si][a] = a'。"""
    size = int(size)
    if size not in _perm_cache:
        n = size * size
        _perm_cache[size] = tuple(
            np.array([transform_index(a, k, m, size) for a in range(n)], dtype=np.int64)
            for (k, m) in SYMMETRIES
        )
    return _perm_cache[size]


def transform_board(cells, rot_k, mirror):
    """返回变换后的棋盘（list），尺寸从 cells 推导。"""
    size = side_of(cells)
    perm = perm_tables(size)[_sym_index(rot_k, mirror)]
    out = [EMPTY] * len(cells)
    for a in range(len(cells)):
        out[perm[a]] = cells[a]
    return out


def transform_last_move(last_move, rot_k, mirror, size=BOARD_SIZE):
    if last_move is None:
        return None
    return int(perm_tables(size)[_sym_index(rot_k, mirror)][last_move])


def transform_policy(pi, rot_k, mirror):
    """把策略分布按同一置换表搬运；尺寸从 len(pi) 推导。"""
    pi = np.asarray(pi, dtype=np.float32)
    size = side_of(pi.reshape(-1))
    perm = perm_tables(size)[_sym_index(rot_k, mirror)]
    out = np.zeros_like(pi)
    out[perm] = pi.reshape(-1)
    return out


def encode(cells, to_move, last_move=None):
    """把棋盘编码为 (4, size, size) 的 float32 张量。"""
    size = side_of(cells)
    n = size * size
    own = np.zeros(n, dtype=np.float32)
    opp = np.zeros(n, dtype=np.float32)
    empty = np.zeros(n, dtype=np.float32)
    last = np.zeros(n, dtype=np.float32)

    from .env import other

    enemy = other(to_move)
    for i, v in enumerate(cells):
        if v == EMPTY:
            empty[i] = 1.0
        elif v == to_move:
            own[i] = 1.0
        elif v == enemy:
            opp[i] = 1.0

    if last_move is not None:
        last[last_move] = 1.0

    return np.stack([own, opp, empty, last], axis=0).reshape(N_CHANNELS, size, size)


def encode_board(board):
    return encode(board.cells, board.to_move, board.last_move)


def encode_batch(boards):
    """boards: list[Board] -> (B, 4, size, size)"""
    return np.stack([encode_board(b) for b in boards], axis=0)


def legal_mask(cells):
    """与 cells 等长的 float32 掩码：1 表示空位可落子。"""
    return np.array([1.0 if v == EMPTY else 0.0 for v in cells], dtype=np.float32)


def softmax_masked(logits, legal):
    """只在合法动作上做 softmax；非法位置为 0，合法位置和为 1。"""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    legal = [int(a) for a in legal]
    if not legal:
        raise ValueError("没有合法动作，无法归一化")
    if len(legal) > logits.shape[0]:
        raise ValueError(f"合法动作数 {len(legal)} 超过 logits 长度 {logits.shape[0]}")

    out = np.zeros(logits.shape[0], dtype=np.float32)
    sel = logits[legal]
    m = float(sel.max())
    e = np.exp(sel - m)
    s = float(e.sum())
    if s <= 0 or not np.isfinite(s):
        out[legal] = 1.0 / len(legal)
        return out
    out[legal] = (e / s).astype(np.float32)
    return out


# 兼容旧接口：默认尺寸的置换表
PERM = perm_tables(BOARD_SIZE)
