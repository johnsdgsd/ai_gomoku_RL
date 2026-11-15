"""PUCT MCTS（AlphaZero 风格搜索）。尺寸由 cells 长度决定（9/13/15...）。

定稿语义（不可违反的不变量）：
  1. 每个节点都判终局：根在搜索入口判，其余在创建子节点时判；搜索中只读标记。
  2. 终局节点不展开、不调网络、不当父节点继续向下。
  3. N_total == Σ N[a]，回溯时同一轮同步加一。
  4. W[a] 恒为父节点视角；回溯每层先 v = -v 再累加。
  5. q(a) = W[a]/N[a]，N[a]=0 时为 0（不是 -inf）。
  6. 先验只在合法动作上归一化，每节点只算一次。
  7. 每节点最多一次评估（展开时），终局节点零次。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .env import (
    EMPTY,
    check_win_at,
    is_full,
    legal_actions,
    other,
    side_of,
)
from .evaluators import find_win_moves


@dataclass
class ChildStats:
    """动作统计（父节点视角）。"""

    N: int = 0
    W: float = 0.0

    @property
    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class Node:
    """MCTS 节点。"""

    __slots__ = (
        "cells",
        "to_move",
        "last_move",
        "size",
        "parent",
        "move",
        "terminal",
        "terminal_value",
        "expanded",
        "priors",
        "value",
        "children",
        "stats",
        "legal",
        "N_total",
    )

    def __init__(self, cells, to_move, last_move=None, parent=None, move=None):
        self.cells = cells
        self.to_move = to_move
        self.last_move = last_move
        self.size = side_of(cells)
        self.parent = parent
        self.move = move
        self.children: Dict[int, Optional["Node"]] = {}
        self.stats: Dict[int, ChildStats] = {}
        self.priors: Dict[int, float] = {}
        self.value = 0.0
        self.expanded = False
        self.N_total = 0
        self.legal: List[int] = []

        # 终局判定：创建时一次，之后只读
        if last_move is not None and cells[last_move] != EMPTY:
            winner = (
                cells[last_move]
                if check_win_at(cells, last_move, cells[last_move])
                else None
            )
        else:
            winner = None
        full = is_full(cells)
        self.terminal = winner is not None or full
        if self.terminal:
            # 本节点 to_move 视角：有人连五则本方输，满盘平局为 0
            self.terminal_value = 0.0 if winner is None else -1.0
        else:
            self.terminal_value = 0.0
            self.legal = legal_actions(cells)


@dataclass
class SearchResult:
    action: int
    pi: np.ndarray  # 与棋盘等长，和 = 1 的软分布
    value: float  # V̄ = ΣW/ΣN，根 to_move 视角
    best_q: float  # q(argmax N)
    visits: int
    root: Node = field(repr=False)
    chosen_by_rule: bool = False

    @property
    def root_value(self) -> float:
        return self.value


class MCTS:
    def __init__(
        self,
        evaluator,
        simulations: int = 200,
        c_puct: float = 1.5,
        root_noise: bool = False,
        dir_alpha: float = 0.3,
        dir_eps: float = 0.25,
        rng: Optional[random.Random] = None,
    ):
        self.evaluator = evaluator
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.root_noise = bool(root_noise)
        self.dir_alpha = float(dir_alpha)
        self.dir_eps = float(dir_eps)
        self.rng = rng or random.Random()

    # ---------------- 展开 ----------------
    def _expand(self, node: Node) -> float:
        """展开非终局节点：一次评估，登记全部动作的空壳子节点。返回 node.value。

        子节点是**惰性**的：先为所有合法动作建统计项与先验，真正的 Node 对象
        在第一次被 PUCT 选中时才创建。这样每次模拟最多创建一个节点。
        """
        probs, value = self.evaluator.evaluate(
            node.cells, node.to_move, node.last_move, node.legal
        )
        node.value = float(value)
        node.priors = {a: float(probs[a]) for a in node.legal}
        for a in node.legal:
            node.children[a] = None
            node.stats[a] = ChildStats()
        node.expanded = True
        node.N_total = 0
        return node.value

    def _materialize(self, parent: Node, a: int) -> Node:
        """创建（并缓存）父节点在动作 a 上的子节点。"""
        child = parent.children.get(a)
        if child is None:
            child_cells = list(parent.cells)
            child_cells[a] = parent.to_move
            child = Node(child_cells, other(parent.to_move), a, parent=parent, move=a)
            parent.children[a] = child
        return child

    def _add_root_noise(self, node: Node):
        if not self.root_noise or not node.legal:
            return
        rng = np.random.default_rng(self.rng.randrange(2**32))
        noise = rng.dirichlet([self.dir_alpha] * len(node.legal))
        for i, a in enumerate(node.legal):
            node.priors[a] = (1.0 - self.dir_eps) * node.priors[a] + self.dir_eps * float(
                noise[i]
            )

    # ---------------- 选择 ----------------
    def _select_child(self, node: Node) -> int:
        legal = node.legal
        # 特判：从未被访问过 -> 按先验最大
        if node.N_total == 0:
            best_p = max(node.priors[a] for a in legal)
            cand = [a for a in legal if node.priors[a] >= best_p - 1e-12]
            return cand[self.rng.randrange(len(cand))]

        sqrt_n = math.sqrt(node.N_total)
        best_a = legal[0]
        best_s = -math.inf
        ties: List[int] = []
        for a in legal:
            st = node.stats[a]
            explore = self.c_puct * node.priors[a] * sqrt_n / (1.0 + st.N)
            s = st.q + explore
            if s > best_s + 1e-9:
                best_s, best_a, ties = s, a, [a]
            elif abs(s - best_s) <= 1e-9:
                ties.append(a)
        if ties:
            return ties[self.rng.randrange(len(ties))]
        return best_a

    # ---------------- 主搜索 ----------------
    def search(
        self,
        cells,
        to_move,
        last_move=None,
        reuse_root: Optional[Node] = None,
    ) -> SearchResult:
        n_cells = len(cells)
        # 0. 根节点：优先复用旧树中对应的子节点
        if reuse_root is not None:
            root = reuse_root
        else:
            root = Node(cells, to_move, last_move)

        if root.terminal:
            return SearchResult(
                action=-1,
                pi=np.zeros(n_cells, dtype=np.float32),
                value=root.terminal_value,
                best_q=root.terminal_value,
                visits=0,
                root=root,
            )

        # 1. 展开根
        if not root.expanded:
            self._expand(root)
            self._add_root_noise(root)

        # 2. 模拟
        for _ in range(self.simulations):
            node = root
            path: List[Tuple[Node, int]] = []

            while node.expanded and not node.terminal:
                a = self._select_child(node)
                path.append((node, a))
                node = self._materialize(node, a)

            if node.terminal:
                v = node.terminal_value
            elif not node.expanded:
                v = self._expand(node)
            else:  # 防御分支
                v = node.value

            while path:
                parent, a = path.pop()
                v = -v
                st = parent.stats[a]
                st.W += v
                st.N += 1
                parent.N_total += 1

        # 3. 收口
        legal = root.legal
        total = sum(root.stats[a].N for a in legal)
        if total == 0:  # 防御：模拟数为 0
            best = self._argmax_prior(root)
            pi = np.zeros(n_cells, dtype=np.float32)
            pi[best] = 1.0
            return SearchResult(best, pi, root.value, 0.0, 0, root)

        pi = np.zeros(n_cells, dtype=np.float32)
        for a in legal:
            pi[a] = root.stats[a].N / total

        best = self._argmax_visits(root)
        best_q = root.stats[best].q

        # 硬规则保险（O(legal)）
        chosen_by_rule = False
        wins = find_win_moves(root.cells, root.to_move)
        if wins:
            best = min(wins)
            chosen_by_rule = True
        else:
            opp_wins = find_win_moves(root.cells, other(root.to_move))
            if len(opp_wins) == 1 and opp_wins[0] in root.children:
                best = opp_wins[0]
                chosen_by_rule = True

        v_bar = sum(root.stats[a].W for a in legal) / total
        return SearchResult(
            action=best,
            pi=pi,
            value=float(v_bar),
            best_q=float(best_q),
            visits=int(total),
            root=root,
            chosen_by_rule=chosen_by_rule,
        )

    # ---------------- 收口辅助 ----------------
    def _argmax_visits(self, node: Node) -> int:
        best_n = -1
        ties: List[int] = []
        for a in node.legal:
            n = node.stats[a].N
            if n > best_n:
                best_n, ties = n, [a]
            elif n == best_n:
                ties.append(a)
        return ties[self.rng.randrange(len(ties))]

    def _argmax_prior(self, node: Node) -> int:
        best_p = -1.0
        ties: List[int] = []
        for a in node.legal:
            p = node.priors[a]
            if p > best_p:
                best_p, ties = p, [a]
            elif abs(p - best_p) <= 1e-12:
                ties.append(a)
        return ties[self.rng.randrange(len(ties))]

    # ---------------- 调试 ----------------
    def describe(self, result: SearchResult, top_k: int = 5) -> str:
        root = result.root
        if root.terminal:
            return "[MCTS] 根局面已终局，无需搜索"
        rows = sorted(root.legal, key=lambda a: -root.stats[a].N)[:top_k]
        lines = [
            f"[MCTS] 模拟 {result.visits} 次 | 预估价值 V̄={result.value:+.3f} | "
            f"最佳 q={result.best_q:+.3f}"
        ]
        for a in rows:
            r, c = divmod(a, root.size)
            st = root.stats[a]
            lines.append(
                f"   ({r},{c})  N={st.N:4d}  q={st.q:+.3f}  P={root.priors[a]:.3f}"
            )
        return "\n".join(lines)
