"""回放池：只存**真实对局路径**（MCTS 内部的仿真数据不进回放池）。

一条记录：
    state    : uint8[4, S, S]   局面编码（S = 棋盘边长）
    action   : int              实际走出的动作
    pi       : float32[S*S]     MCTS 访问分布（策略头标签，软分布）
    vbar     : float            搜索估计的仿真价值（价值头可选标签）
    to_move  : int              该记录的视角主人
    z        : float            真实终局信号（一局结束后回填，±1/0）

尺寸由记录本身决定，9x9 与 15x15 的记录不会混用（同一训练运行只产生一种尺寸）。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .encoder import SYMMETRIES, perm_tables, transform_policy
from .env import side_of


@dataclass
class Record:
    state: np.ndarray  # (4, S, S) uint8
    action: int
    pi: np.ndarray  # (S*S,) float32
    vbar: float
    to_move: int
    z: float = 0.0


def _transform_planes(planes: np.ndarray, si: int) -> np.ndarray:
    size = int(planes.shape[-1])
    perm = perm_tables(size)[si]
    out = np.empty_like(planes)
    for ch in range(planes.shape[0]):
        out[ch].reshape(-1)[perm] = planes[ch].reshape(-1)
    return out


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.buf: deque = deque(maxlen=int(capacity))
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buf)

    # ---------- 写入 ----------
    def add_record(self, rec: Record):
        self.buf.append(rec)

    def add(self, state: np.ndarray, action: int, pi: np.ndarray, vbar: float,
            to_move: int, z: float = 0.0):
        self.buf.append(
            Record(
                state=np.asarray(state, dtype=np.uint8),
                action=int(action),
                pi=np.asarray(pi, dtype=np.float32),
                vbar=float(vbar),
                to_move=int(to_move),
                z=float(z),
            )
        )

    def add_game(self, records: Sequence[Record], winner: Optional[int]):
        """一局结束后回填真实终局信号 z 并写入。

        z 按每条记录自己的 to_move 给：赢 +1、输 -1、平 0。
        """
        for rec in records:
            if winner is None:
                z = 0.0
            elif winner == rec.to_move:
                z = 1.0
            else:
                z = -1.0
            rec.z = z
            self.add(rec.state, rec.action, rec.pi, rec.vbar, rec.to_move, z)

    # ---------- 读取 ----------
    def sample(self, batch_size: int, augment: bool = True, lambda_vbar: float = 0.2):
        """采样一个 batch。

        返回 dict：states float32(B,4,S,S)、actions int64、pis float32(B,S*S)、
        targets float32(B)（价值标签）。
        """
        n = len(self.buf)
        if n == 0:
            raise ValueError("回放池为空")
        idxs = [self.rng.randrange(n) for _ in range(int(batch_size))]
        batch = [self.buf[i] for i in idxs]

        size = int(batch[0].state.shape[-1])
        n_cells = size * size
        states = np.empty((len(batch), 4, size, size), dtype=np.float32)
        actions = np.empty(len(batch), dtype=np.int64)
        pis = np.empty((len(batch), n_cells), dtype=np.float32)
        targets = np.empty(len(batch), dtype=np.float32)

        for i, rec in enumerate(batch):
            state = rec.state
            action = rec.action
            pi = rec.pi
            if augment:
                si = self.rng.randrange(len(SYMMETRIES))
                if si != 0:
                    state = _transform_planes(state, si)
                    action = int(perm_tables(size)[si][action])
                    pi = transform_policy(pi, *SYMMETRIES[si])
            states[i] = state.astype(np.float32)
            actions[i] = action
            pis[i] = pi
            targets[i] = (1.0 - lambda_vbar) * rec.z + lambda_vbar * rec.vbar

        return {
            "states": states,
            "actions": actions,
            "pis": pis,
            "targets": targets,
        }
