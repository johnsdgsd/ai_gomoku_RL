"""自对弈：真实棋盘对局 → 记录 (s, a, π, V̄, to_move) → 一局结束回填 z。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .encoder import encode
from .env import BOARD_SIZE, Board, P1, P2, other
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
    dir_eps: float = 0.25
    temp_moves: int = 15
    temp: float = 1.0
    size: int = BOARD_SIZE
    max_moves: Optional[int] = None  # 默认 size*size
    # 训练对手（启发式）每步的搜索次数；0 表示不走搜索、直接用启发式贪心
    opponent_sims: int = 32


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
    ):
        self.rng = rng or random.Random()
        self.ev = HeuristicEvaluator(radius=radius)
        self.noise = float(noise)  # greedy 档位：以该概率在前几名里随机
        self.sims = int(sims)
        self.mcts = (
            MCTS(self.ev, simulations=self.sims, rng=self.rng) if self.sims > 0 else None
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

        # 4) 回退：按启发式先验贪心
        probs, _ = self.ev.evaluate(cells, board.to_move, board.last_move, legal)
        order = sorted(legal, key=lambda a: -float(probs[a]))
        if self.noise > 0 and self.rng.random() < self.noise and len(order) > 1:
            top = order[: min(5, len(order))]
            return self.rng.choice(top)
        return order[0]


def make_opponent(
    name: str, rng: random.Random, sims: int = 32
) -> Optional[Player]:
    """构造训练对手；返回 None 表示"自己"（纯自对弈）。"""
    if name in ("self", "none"):
        return None
    if name == "random":
        return RandomPlayer(rng)
    if name == "heuristic":
        return HeuristicPlayer(rng, sims=sims)
    if name == "greedy":
        # "greedy" 档位：加大随机、不搜索，作为更弱的陪练
        return HeuristicPlayer(rng, noise=0.3, sims=0)
    raise ValueError(f"未知对手: {name}（可选 self / heuristic / greedy / random）")


def _opponent(name: str, rng: random.Random, sims: int = 32) -> Optional[Player]:
    return make_opponent(name, rng, sims=sims)


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
        rng=rng,
    )
    opp = _opponent(opponent, rng, sims=cfg.opponent_sims)
    agent_color = P1 if agent_first else P2

    board = Board(size=cfg.size)
    max_moves = cfg.max_moves if cfg.max_moves is not None else cfg.size * cfg.size
    records: List[Record] = []
    step = 0
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
            records.append(
                Record(
                    state=encode(board.cells, board.to_move, board.last_move),
                    action=a,
                    pi=res.pi.copy(),
                    vbar=res.value,
                    to_move=board.to_move,
                )
            )
        else:
            a = opp.select(board)
        board = board.play(a)
        step += 1

    return records, board.winner, step
