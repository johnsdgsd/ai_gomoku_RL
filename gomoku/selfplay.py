"""自对弈：真实棋盘对局 → 记录 (s, a, π, V̄, to_move) → 一局结束回填 z。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .encoder import encode
from .env import BOARD_SIZE, Board, P1, P2, other
from .evaluators import NetEvaluator
from .mcts import MCTS
from .players import Player, RandomPlayer, GreedyPlayer, sample_by_visits
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


def _opponent(name: str, rng: random.Random) -> Optional[Player]:
    if name in ("self", "none"):
        return None
    if name == "random":
        return RandomPlayer(rng)
    if name == "greedy":
        return GreedyPlayer(rng)
    if name == "heuristic":
        return GreedyPlayer(rng)  # 轻量对手，避免训练时对手本身太慢
    raise ValueError(f"未知对手: {name}")


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
    opp = _opponent(opponent, rng)
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
