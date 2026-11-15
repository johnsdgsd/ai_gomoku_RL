import os
import unittest

import numpy as np

from gomoku.env import (
    BOARD_SIZE,
    EMPTY,
    P1,
    P2,
    Board,
    check_win_at,
    side_of,
)
from gomoku.encoder import encode, perm_tables

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def idx(size, r, c):
    return r * size + c


class TestBoardSizes(unittest.TestCase):
    def test_side_of(self):
        for size in (9, 13, 15, 19):
            self.assertEqual(side_of([EMPTY] * (size * size)), size)
        with self.assertRaises(ValueError):
            side_of([EMPTY] * 10)

    def test_board_size_15_basic(self):
        b = Board(size=15)
        self.assertEqual(b.size, 15)
        self.assertEqual(len(b.cells), 225)
        self.assertEqual(len(b.legal_actions()), 225)
        b2 = b.play(idx(15, 7, 7))
        self.assertEqual(b2.last_move, idx(15, 7, 7))
        self.assertEqual(b2.to_move, P2)
        board_lines = [ln for ln in b2.render().splitlines() if "|" in ln]
        self.assertEqual(len(board_lines), 15)
        self.assertIn("14", b2.render())

    def test_win_detection_15_horizontal_vertical_diagonal(self):
        for positions in (
            [idx(15, 7, c) for c in range(5)],
            [idx(15, r, 7) for r in range(5)],
            [idx(15, i, i) for i in range(5)],
            [idx(15, i, 14 - i) for i in range(5)],
        ):
            cells = [EMPTY] * 225
            for a in positions:
                cells[a] = P1
            self.assertTrue(check_win_at(cells, positions[-1], P1))

    def test_no_win_across_edge_15(self):
        cells = [EMPTY] * 225
        for c in range(13, 15):
            cells[idx(15, 5, c)] = P1
        for c in range(3):
            cells[idx(15, 6, c)] = P1
        self.assertFalse(check_win_at(cells, idx(15, 6, 2), P1))

    def test_four_is_not_win_15(self):
        cells = [EMPTY] * 225
        for c in range(4):
            cells[idx(15, 0, c)] = P1
        self.assertFalse(check_win_at(cells, idx(15, 0, 3), P1))


class TestEncodeSizes(unittest.TestCase):
    def test_encode_shapes(self):
        for size in (9, 15):
            cells = [EMPTY] * (size * size)
            cells[0] = P1
            s = encode(cells, P2, last_move=0)
            self.assertEqual(s.shape, (4, size, size))
            self.assertAlmostEqual(float(s[0].sum() + s[1].sum() + s[2].sum()), size * size)
            self.assertEqual(float(s[3].sum()), 1.0)

    def test_perm_tables_sizes(self):
        for size in (9, 15):
            perms = perm_tables(size)
            self.assertEqual(len(perms), 8)
            for perm in perms:
                self.assertEqual(sorted(perm.tolist()), list(range(size * size)))

    def test_roundtrip_15(self):
        from gomoku.encoder import SYMMETRIES, transform_board

        b = Board(size=15)
        b = b.play(idx(15, 0, 1)).play(idx(15, 7, 9)).play(idx(15, 14, 0))
        for k, m in SYMMETRIES:
            t = transform_board(b.cells, k, m)
            restored = False
            for k2, m2 in SYMMETRIES:
                if transform_board(t, k2, m2) == b.cells:
                    restored = True
                    break
            self.assertTrue(restored, f"15x15 变换 {(k, m)} 无法还原")


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestNetworkSizes(unittest.TestCase):
    def test_network_15_output_shape(self):
        from gomoku.network import build_network

        net = build_network(channels=16, blocks=1, size=15)
        x = torch.randn(2, 4, 15, 15)
        logits, value = net(x)
        self.assertEqual(tuple(logits.shape), (2, 225))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.all(value.abs() <= 1.0))

    def test_network_size_mismatch_on_load(self):
        from gomoku.network import Network

        d = os.path.join(os.path.dirname(__file__), "_artifacts")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "size15.pt")
        try:
            net = Network(channels=8, blocks=1, size=15)
            net.save(path)
            loaded = Network.load(path)  # 不指定尺寸 -> 用文件里的 15
            self.assertEqual(loaded.size, 15)
            with self.assertRaises(ValueError):
                Network.load(path, size=9)
        finally:
            if os.path.exists(path):
                os.remove(path)


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestMCTS15(unittest.TestCase):
    def test_win_move_15(self):
        from gomoku.evaluators import HeuristicEvaluator
        from gomoku.mcts import MCTS

        size = 15
        cells = [EMPTY] * (size * size)
        for c in range(4):
            cells[idx(size, 3, c)] = P1
        res = MCTS(HeuristicEvaluator(), simulations=40).search(
            cells, P1, last_move=idx(size, 3, 3)
        )
        self.assertEqual(res.action, idx(size, 3, 4))
        self.assertEqual(res.pi.shape, (225,))

    def test_block_four_15(self):
        from gomoku.evaluators import HeuristicEvaluator
        from gomoku.mcts import MCTS

        size = 15
        cells = [EMPTY] * (size * size)
        for r in range(4):
            cells[idx(size, r, 10)] = P2
        res = MCTS(HeuristicEvaluator(), simulations=40).search(
            cells, P1, last_move=idx(size, 3, 10)
        )
        self.assertEqual(res.action, idx(size, 4, 10))


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestPipeline15(unittest.TestCase):
    def test_replay_sample_15(self):
        from gomoku.replay import Record, ReplayBuffer

        size = 15
        state = np.zeros((4, size, size), dtype=np.uint8)
        state[2] = 1
        a = idx(size, 7, 7)
        state[2].reshape(-1)[a] = 0
        state[0].reshape(-1)[a] = 1
        state[3].reshape(-1)[a] = 1
        pi = np.zeros(size * size, dtype=np.float32)
        pi[a] = 1.0
        buf = ReplayBuffer(capacity=10)
        buf.add_game([Record(state, a, pi, 0.0, P1, 0.0)], winner=P1)
        batch = buf.sample(1, augment=True)
        self.assertEqual(batch["states"].shape, (1, 4, size, size))
        self.assertEqual(batch["pis"].shape, (1, size * size))
        aa = int(batch["actions"][0])
        self.assertEqual(batch["states"][0][0].reshape(-1)[aa], 1.0)
        self.assertEqual(batch["pis"][0][aa], 1.0)

    def test_arena_and_selfplay_15(self):
        from gomoku.env import Board
        from gomoku.network import build_network
        from gomoku.players import MCTSPlayer
        from gomoku.evaluators import NetEvaluator
        from gomoku.selfplay import SelfPlayConfig, play_selfplay
        from gomoku.arena import arena

        net = build_network(channels=8, blocks=1, size=15).eval()
        cfg = SelfPlayConfig(simulations=2, size=15)
        records, winner, steps = play_selfplay(
            net, cfg=cfg, opponent="self", seed=1
        )
        self.assertGreater(steps, 0)
        if records:
            self.assertEqual(records[0].state.shape, (4, 15, 15))

        p1 = MCTSPlayer(NetEvaluator(net), simulations=2)
        p2 = MCTSPlayer(NetEvaluator(net), simulations=2)
        res = arena(p1, p2, games=2, size=15)
        self.assertEqual(res.games, 2)


class TestConfigBoardSize(unittest.TestCase):
    def test_train_config_reads_board_size(self):
        from gomoku.train import TrainConfig

        cfg = {"board": {"size": 15}, "net": {"channels": 8, "blocks": 1}}
        tcfg = TrainConfig.from_dict(cfg)
        self.assertEqual(tcfg.size, 15)

    def test_default_config_board_size(self):
        from gomoku.device import load_config

        cfg = load_config()
        self.assertEqual(cfg["board"]["size"], 15)
        self.assertGreaterEqual(cfg["board"]["size"], 5)


if __name__ == "__main__":
    unittest.main()
