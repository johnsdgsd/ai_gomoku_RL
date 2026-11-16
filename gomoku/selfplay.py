"""自对弈：真实棋盘对局 → 记录 (s, a, π, V̄, to_move) → 一局结束回填 z。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .encoder import encode
from .env import BOARD_SIZE, EMPTY, Board, P1, P2, other, region_actions
from .evaluators import HeuristicEvaluator, NetEvaluator, find_win_moves
from .mcts import MCTS
from .players import Player, RandomPlayer, sample_by_visits
from .replay import Record


@dataclass
class SelfPlayConfig:
    simulations: int = 150
    c_puct: float = 1.5
    root_noise: bool = True
    dir_alpha: float = 0.3
    dir_eps: float = 0.05
    temp_moves: int = 15
    temp: float = 1.0
    size: int = BOARD_SIZE
    max_moves: Optional[int] = None  # 默认 size*size
    # 训练对手（启发式）每步的搜索次数；0 表示不走搜索、直接用启发式贪心
    opponent_sims: int = 32
    # 可落子范围："棋子外接正方形再外扩 margin 圈"；<0 表示全棋盘
    search_margin: int = -1
    # 训练开始时先用启发式在空盘随机走 K 手，跳过网络的弱开局（0 表示不启用）
    opening_moves: int = 0
    opening_max: int = 0
    # 训练标签 π 的最小访问次数，低于此值的动作清零后重新归一化（0 表示不截断）
    pi_min_visits: int = 0


class HeuristicPlayer(Player):
    """启发式评估器 + MCTS 的陪练对手。

    - sims > 0：用 HeuristicEvaluator 做 PUCT 搜索（比纯贪心强，但慢）
    - sims = 0：直接用启发式先验贪心（每步约 0.3–1 ms）
    两种情况都先判断"能赢就赢 / 对手能赢就堵"。
    """

    name = "heuristic"

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        radius: int = 2,
        noise: float = 0.05,
        sims: int = 32,
        margin: int = -1,
    ):
        self.rng = rng or random.Random()
        self.ev = HeuristicEvaluator(radius=radius)
        self.noise = float(noise)  # greedy 档位：以该概率在前几名里随机
        self.sims = int(sims)
        self.margin = int(margin)
        self.mcts = (
            MCTS(self.ev, simulations=self.sims, margin=self.margin, rng=self.rng)
            if self.sims > 0
            else None
        )

    def select(self, board) -> int:
        cells = board.cells
        legal = board.legal_actions()
        if not legal:
            raise ValueError("没有合法动作")
        # 1) 能赢就赢 / 2) 对手能赢就堵
        wins = find_win_moves(cells, board.to_move)
        if wins:
            return wins[0]
        blocks = find_win_moves(cells, other(board.to_move))
        if blocks:
            return blocks[0]

        # 3) 走搜索
        if self.mcts is not None:
            res = self.mcts.search(cells, board.to_move, board.last_move)
            if res.action >= 0:
                return res.action

        # 4) 回退：只在可落子区域内按启发式先验贪心
        cands = region_actions(cells, self.margin) if self.margin >= 0 else legal
        cands = [a for a in cands if cells[a] == EMPTY]
        if not cands:
            cands = legal
        probs, _ = self.ev.evaluate(cells, board.to_move, board.last_move, cands)
        order = sorted(cands, key=lambda a: -float(probs[a]))
        if self.noise > 0 and self.rng.random() < self.noise and len(order) > 1:
            top = order[: min(5, len(order))]
            return self.rng.choice(top)
        return order[0]


def make_opponent(
    name: str, rng: random.Random, sims: int = 32, margin: int = -1
) -> Optional[Player]:
    """构造训练对手；返回 None 表示"自己"（纯自对弈）。"""
    if name in ("self", "none"):
        return None
    if name == "random":
        return RandomPlayer(rng)
    if name == "heuristic":
        return HeuristicPlayer(rng, sims=sims, margin=margin)
    if name == "greedy":
        # "greedy" 档位：加大随机、不搜索，作为更弱的陪练
        return HeuristicPlayer(rng, noise=0.3, sims=0, margin=margin)
    raise ValueError(f"未知对手: {name}（可选 self / heuristic / greedy / random）")


def _opponent(name: str, rng: random.Random, sims: int = 32, margin: int = -1) -> Optional[Player]:
    return make_opponent(name, rng, sims=sims, margin=margin)


def play_selfplay(
    net,
    device: str = "cpu",
    cfg: Optional[SelfPlayConfig] = None,
    opponent: str = "self",
    seed: int = 0,
    agent_first: bool = True,
) -> tuple[List[Record], Optional[int], int]:
    """跑一局。返回 (记录列表, 赢家 P1/P2/None, 步数)。

    opponent="self" 时双方都由网络+MCTS 控制（两份记录）；
    否则网络只控制 `agent_first` 指定的一方，对手控制另一方（只记录网络自己的着法）。
    记录只包含**真实棋盘**上的着法，不包含 MCTS 内部推演。
    """
    cfg = cfg or SelfPlayConfig()
    rng = random.Random(seed)
    evaluator = NetEvaluator(net, device=device)
    mcts = MCTS(
        evaluator,
        simulations=cfg.simulations,
        c_puct=cfg.c_puct,
        root_noise=cfg.root_noise,
        dir_alpha=cfg.dir_alpha,
        dir_eps=cfg.dir_eps,
        margin=cfg.search_margin,
        rng=rng,
    )
    opp = _opponent(opponent, rng, sims=cfg.opponent_sims, margin=cfg.search_margin)
    agent_color = P1 if agent_first else P2

    board = Board(size=cfg.size)
    max_moves = cfg.max_moves if cfg.max_moves is not None else cfg.size * cfg.size
    records: List[Record] = []
    step = 0

    # 可选：用启发式先走开局，跳过网络的薄弱开局阶段（不记录这些着法）
    k = int(cfg.opening_moves or 0)
    if k > 0 and cfg.opening_max > 0:
        k = rng.randint(k, max(k, int(cfg.opening_max)))
        opener = HeuristicPlayer(rng, sims=0, margin=cfg.search_margin)
        for _ in range(k):
            if board.is_terminal():
                break
            board = board.play(opener.select(board))
            step += 1

    while not board.is_terminal() and step < max_moves:
        controlled = (opp is None) or (board.to_move == agent_color)
        if controlled:
            res = mcts.search(board.cells, board.to_move, board.last_move)
            if res.action < 0:
                break
            temperature = cfg.temp if step < cfg.temp_moves else 0.0
            a = (
                sample_by_visits(res, temperature, rng)
                if temperature > 0
                else res.action
            )
            pi = res.pi.copy()
            if cfg.pi_min_visits > 0:
                pi = sharpen_pi(pi, cfg.pi_min_visits)
            records.append(
                Record(
                    state=encode(board.cells, board.to_move, board.last_move),
                    action=a,
                    pi=pi,
                    vbar=res.value,
                    to_move=board.to_move,
                )
            )
        else:
            a = opp.select(board)
        board = board.play(a)
        step += 1

    return records, board.winner, step


def sharpen_pi(pi, min_visits, total_visits=None):
    """把访问次数低于 min_visits 的动作清零后重新归一化（让标签变尖）。

    pi 是"访问次数 / 总访问数"的分布。给定 total_visits 可精确换算访问数；
    否则用 pi * total 近似。任何情况下若截断后为空，则原样返回。
    """
    pi = np.asarray(pi, dtype=np.float32)
    total = float(total_visits) if total_visits else float(pi.sum())
    if total <= 0:
        return pi
    visits = pi * total
    keep = visits >= float(min_visits)
    if not keep.any():
        return pi
    out = np.where(keep, pi, 0.0).astype(np.float32)
    s = float(out.sum())
    if s <= 0:
        return pi
    return out / s
