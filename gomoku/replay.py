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

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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

    # ---------- 持久化 ----------
    def save(self, path) -> int:
        """把整个回放池原子写入 .npz，返回写入的条数。

        先写 `xxx.npz.tmp` 再 `os.replace`，中途被打断不会损坏已有文件。
        所有数组都是定长的，压缩后体积约为裸 float32 的 1/10（局面是 0/1 平面）。
        """
        n = len(self.buf)
        if n == 0:
            raise ValueError("回放池为空，无法保存")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = int(self.buf[0].state.shape[-1])

        states = np.empty((n, 4, size, size), dtype=np.uint8)
        actions = np.empty(n, dtype=np.int16)
        pis = np.empty((n, size * size), dtype=np.float32)
        vbars = np.empty(n, dtype=np.float32)
        to_moves = np.empty(n, dtype=np.int8)
        zs = np.empty(n, dtype=np.float32)
        for i, rec in enumerate(self.buf):
            states[i] = rec.state
            actions[i] = rec.action
            pis[i] = rec.pi
            vbars[i] = rec.vbar
            to_moves[i] = rec.to_move
            zs[i] = rec.z

        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                capacity=np.int64(self.buf.maxlen or n),
                size=np.int64(size),
                states=states,
                actions=actions,
                pis=pis,
                vbars=vbars,
                to_moves=to_moves,
                zs=zs,
            )
        os.replace(tmp, path)
        return n

    @classmethod
    def load(cls, path, capacity: Optional[int] = None, seed: int = 0) -> "ReplayBuffer":
        """从 .npz 恢复回放池。

        `capacity` 显式给出时以它为准（用于续训时把池子改小），否则沿用保存时的容量。
        z 已经回填过，恢复后可以直接采样训练。
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            states = np.asarray(data["states"])
            actions = np.asarray(data["actions"])
            pis = np.asarray(data["pis"])
            vbars = np.asarray(data["vbars"])
            to_moves = np.asarray(data["to_moves"])
            zs = np.asarray(data["zs"]) if "zs" in files else np.zeros(len(actions), np.float32)
            stored_cap = int(data["capacity"]) if "capacity" in files else len(actions)
        size = int(states.shape[-1])
        if pis.shape[1] != size * size:
            raise ValueError(
                f"回放池尺寸不一致：state 是 {size}x{size}，pi 长度却是 {pis.shape[1]}"
            )

        cap = int(capacity) if capacity is not None else stored_cap
        if cap < len(actions):
            # 池子被改小时只保留最新的记录
            keep = slice(len(actions) - cap, len(actions))
            states, actions = states[keep], actions[keep]
            pis, vbars = pis[keep], vbars[keep]
            to_moves, zs = to_moves[keep], zs[keep]

        buf = cls(capacity=max(1, cap), seed=seed)
        for i in range(len(actions)):
            buf.buf.append(
                Record(
                    state=states[i].copy(),
                    action=int(actions[i]),
                    pi=pis[i].copy(),
                    vbar=float(vbars[i]),
                    to_move=int(to_moves[i]),
                    z=float(zs[i]),
                )
            )
        return buf
