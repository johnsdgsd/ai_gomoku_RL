"""对拉评测：两个玩家交手，交换先后手，统计胜率。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .env import BOARD_SIZE, Board, P1, P2


@dataclass
class ArenaResult:
    games: int
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """胜=1 平=0.5 的得分率。"""
        if not self.games:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def __str__(self):
        return (
            f"{self.games} 局: 胜 {self.wins} / 负 {self.losses} / 平 {self.draws} "
            f"(胜率 {self.win_rate:.1%}, 得分率 {self.score:.1%})"
        )


def play_game(first, second, size: int = BOARD_SIZE, max_moves: Optional[int] = None) -> int:
    """first 执 P1、second 执 P2。返回 1(first 胜) / -1(second 胜) / 0(平)。"""
    size = int(size)
    limit = size * size if max_moves is None else int(max_moves)
    board = Board(size=size)
    while not board.is_terminal() and board.moves_played() < limit:
        player = first if board.to_move == P1 else second
        a = player.select(board)
        board = board.play(a)
    if board.winner == P1:
        return 1
    if board.winner == P2:
        return -1
    return 0


def arena(
    a,
    b,
    games: int = 100,
    size: int = BOARD_SIZE,
    max_moves: Optional[int] = None,
    progress: Optional[Callable] = None,
) -> ArenaResult:
    """a 与 b 对拉，先后手交替。"""
    wins = losses = draws = 0
    for i in range(games):
        if i % 2 == 0:
            r = play_game(a, b, size, max_moves)
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
        else:
            r = play_game(b, a, size, max_moves)
            if r < 0:
                wins += 1
            elif r > 0:
                losses += 1
            else:
                draws += 1
        if progress is not None:
            progress(i + 1, games, wins, losses, draws)
    return ArenaResult(games, wins, losses, draws)
