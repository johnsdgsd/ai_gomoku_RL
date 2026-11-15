"""对弈参与者（玩家/对手），供 cli、arena、selfplay 复用。"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .env import EMPTY, N_CELLS, legal_actions, other
from .evaluators import HeuristicEvaluator, NetEvaluator, candidate_moves, find_win_moves
from .mcts import MCTS, SearchResult


def sample_by_visits(result: SearchResult, temperature: float, rng: random.Random) -> int:
    """按访问次数采样动作；temperature<=0 时取访问最多。"""
    pi = np.asarray(result.pi, dtype=np.float64)
    legal = np.flatnonzero(pi > 0)
    if legal.size == 0:
        return result.action
    if temperature <= 1e-6:
        best = max(legal, key=lambda a: pi[a])
        return int(best)
    w = pi[legal] ** (1.0 / temperature)
    w = w / w.sum()
    idx = rng.choices(list(legal), weights=w.tolist(), k=1)[0]
    return int(idx)


class Player:
    name = "player"

    def select(self, board) -> int:  # pragma: no cover - 接口
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


class RandomPlayer(Player):
    name = "random"

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def select(self, board) -> int:
        return self.rng.choice(board.legal_actions())


class GreedyPlayer(Player):
    """一步贪心：能赢就赢，其次堵对手五连，最后在棋子附近随机。"""

    name = "greedy"

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def select(self, board) -> int:
        cells = board.cells
        me = board.to_move
        wins = find_win_moves(cells, me)
        if wins:
            return wins[0]
        opp = other(me)
        blocks = find_win_moves(cells, opp)
        if blocks:
            return blocks[0]
        cands = candidate_moves(cells, radius=2)
        cands = [a for a in cands if cells[a] == EMPTY]
        return self.rng.choice(cands) if cands else self.rng.choice(legal_actions(cells))


class MCTSPlayer(Player):
    """PUCT 搜索玩家。evaluator 可换（启发式 / 网络）。"""

    def __init__(
        self,
        evaluator,
        simulations: int = 1000,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        root_noise: bool = False,
        rng: Optional[random.Random] = None,
    ):
        self.name = "mcts"
        self.evaluator = evaluator
        self.mcts = MCTS(
            evaluator,
            simulations=simulations,
            c_puct=c_puct,
            root_noise=root_noise,
            rng=rng,
        )
        self.temperature = temperature
        self.rng = rng or random.Random()
        self.last_result: Optional[SearchResult] = None

    def select(self, board) -> int:
        res = self.mcts.search(board.cells, board.to_move, board.last_move)
        self.last_result = res
        if res.action < 0:
            raise RuntimeError("终局局面不应请求落子")
        if self.temperature > 0:
            return sample_by_visits(res, self.temperature, self.rng)
        return res.action


def heuristic_player(simulations: int = 800, seed: int = 0, **kwargs) -> MCTSPlayer:
    rng = random.Random(seed)
    return MCTSPlayer(HeuristicEvaluator(), simulations=simulations, rng=rng, **kwargs)


def net_player(net, device="cpu", simulations: int = 1000, seed: int = 0, **kwargs) -> MCTSPlayer:
    rng = random.Random(seed)
    return MCTSPlayer(
        NetEvaluator(net, device=device), simulations=simulations, rng=rng, **kwargs
    )


def make_player(
    spec: str,
    cfg: Optional[dict] = None,
    net=None,
    device: str = "cpu",
    seed: int = 0,
    size: Optional[int] = None,
):
    """按名称构造玩家：random / greedy / heuristic / net:path"""
    cfg = cfg or {}
    mcts_cfg = cfg.get("mcts", {})
    sims = int(mcts_cfg.get("sims_play", 800))
    c_puct = float(mcts_cfg.get("c_puct", 1.5))

    spec = (spec or "heuristic").strip()
    if spec == "random":
        return RandomPlayer(random.Random(seed))
    if spec == "greedy":
        return GreedyPlayer(random.Random(seed))
    if spec in ("heuristic", "mcts-heuristic"):
        return heuristic_player(sims, seed=seed, c_puct=c_puct)
    if spec.startswith("net"):
        if net is None:
            from .network import load_network

            path = spec.split(":", 1)[1] if ":" in spec else "models/best.pt"
            net = load_network(path, device=device, size=size)
        return net_player(net, device=device, simulations=sims, seed=seed, c_puct=c_puct)
    raise ValueError(f"未知玩家类型: {spec!r}（可选 random/greedy/heuristic/net:path）")
