import os
import unittest

import numpy as np

from gomoku.env import N_CELLS, P1, P2

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

ART = os.path.join(os.path.dirname(__file__), "_artifacts")


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestSelfPlaySmoke(unittest.TestCase):
    def setUp(self):
        from gomoku.network import build_network

        self.net = build_network(channels=8, blocks=1).eval()

    def test_selfplay_produces_records_with_z(self):
        from gomoku.replay import ReplayBuffer
        from gomoku.selfplay import SelfPlayConfig, play_selfplay

        cfg = SelfPlayConfig(simulations=4, root_noise=True, temp_moves=2)
        records, winner, steps = play_selfplay(
            self.net, device="cpu", cfg=cfg, opponent="self", seed=1
        )
        self.assertGreater(steps, 0)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(r.state.shape == (4, 9, 9) for r in records))

        buf = ReplayBuffer(capacity=100)
        buf.add_game(records, winner)
        for rec in buf.buf:
            self.assertIn(rec.z, (-1.0, 0.0, 1.0))
            self.assertAlmostEqual(float(rec.pi.sum()), 1.0, places=4)

    def test_selfplay_vs_opponent_records_only_agent(self):
        from gomoku.selfplay import SelfPlayConfig, play_selfplay

        cfg = SelfPlayConfig(simulations=3)
        records, winner, steps = play_selfplay(
            self.net, device="cpu", cfg=cfg, opponent="random", seed=2, agent_first=False
        )
        self.assertGreater(steps, 0)
        # 网络执 P2，所以所有记录都应是 P2 的着法（除非对手先结束）
        self.assertTrue(all(r.to_move == P2 for r in records))


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestTrainSmoke(unittest.TestCase):
    def test_train_runs_and_saves(self):
        from gomoku.network import Network
        from gomoku.train import TrainConfig, train

        os.makedirs(ART, exist_ok=True)
        cfg = TrainConfig(
            iterations=1,
            games_per_iter=2,
            batch_size=8,
            updates_per_iter=2,
            eval_every=2,
            eval_games=2,
            sims_train=4,
            sims_eval=2,
            channels=8,
            blocks=1,
            models_dir=os.path.join(ART, "models"),
            logs_dir=os.path.join(ART, "logs"),
            opponents={"self": 1.0},
            seed=3,
        )
        net = train(cfg, device="cpu")
        self.assertTrue(os.path.isfile(os.path.join(ART, "models", "latest.pt")))
        self.assertTrue(os.path.isfile(os.path.join(ART, "models", "final.pt")))
        self.assertTrue(os.path.isfile(os.path.join(ART, "logs", "train_log.csv")))

        # 续训：加载后迭代号应继续增长
        cfg2 = TrainConfig(
            iterations=1,
            games_per_iter=1,
            batch_size=4,
            updates_per_iter=1,
            eval_every=1000,
            eval_games=1,
            sims_train=3,
            sims_eval=2,
            channels=8,
            blocks=1,
            models_dir=os.path.join(ART, "models"),
            logs_dir=os.path.join(ART, "logs"),
            opponents={"self": 1.0},
            seed=4,
            resume=os.path.join(ART, "models", "latest.pt"),
        )
        train(cfg2, device="cpu")
        loaded = Network.load(os.path.join(ART, "models", "latest.pt"))
        self.assertGreaterEqual(loaded.meta["extra"]["iteration"], 2)


if __name__ == "__main__":
    unittest.main()
