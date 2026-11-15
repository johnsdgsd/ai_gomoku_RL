import os
import unittest

import numpy as np

from gomoku.encoder import PERM, SYMMETRIES
from gomoku.env import BOARD_SIZE, EMPTY, N_CELLS, P1, P2, check_win_at
from gomoku.replay import ReplayBuffer, Record


def make_record(action, to_move=P1, a_win=False):
    state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    state[2] = 1
    state[2].reshape(-1)[action] = 0
    state[2].reshape(-1)[0] = 0
    state[0].reshape(-1)[action] = 1
    state[3].reshape(-1)[action] = 1
    pi = np.zeros(N_CELLS, dtype=np.float32)
    pi[action] = 1.0
    return Record(state=state, action=action, pi=pi, vbar=0.4, to_move=to_move)


class TestReplayBuffer(unittest.TestCase):
    def test_add_game_fills_z_by_to_move(self):
        buf = ReplayBuffer(capacity=100)
        r1 = make_record(10, to_move=P1)
        r2 = make_record(11, to_move=P2)
        buf.add_game([r1, r2], winner=P1)

        self.assertEqual(len(buf), 2)
        self.assertAlmostEqual(buf.buf[0].z, 1.0)   # P1 赢 → 记录者赢
        self.assertAlmostEqual(buf.buf[1].z, -1.0)  # P2 视角输

    def test_draw_z_is_zero(self):
        buf = ReplayBuffer(capacity=10)
        buf.add_game([make_record(3, to_move=P2)], winner=None)
        self.assertEqual(buf.buf[0].z, 0.0)

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        recs = [make_record(a) for a in range(5)]
        buf.add_game(recs, winner=P1)
        batch = buf.sample(4, augment=False)
        self.assertEqual(batch["states"].shape, (4, 4, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(batch["actions"].shape, (4,))
        self.assertEqual(batch["pis"].shape, (4, N_CELLS))
        self.assertEqual(batch["targets"].shape, (4,))
        self.assertEqual(batch["states"].dtype, np.float32)

    def test_target_mixes_z_and_vbar(self):
        buf = ReplayBuffer(capacity=10)
        rec = make_record(7)
        rec.vbar = 0.0
        buf.add_game([rec], winner=P1)  # z = 1.0
        batch = buf.sample(1, augment=False, lambda_vbar=0.25)
        self.assertAlmostEqual(float(batch["targets"][0]), 0.75, places=5)

    def test_augmentation_keeps_action_and_pi_aligned(self):
        action = 3 * BOARD_SIZE + 5
        state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        state[2] = 1
        state[2].reshape(-1)[action] = 0
        state[0].reshape(-1)[action] = 1
        state[3].reshape(-1)[action] = 1
        pi = np.zeros(N_CELLS, dtype=np.float32)
        pi[action] = 1.0
        buf = ReplayBuffer(capacity=10)
        buf.add_record(Record(state, action, pi, 0.0, P1, 0.0))

        checked = 0
        for _ in range(40):
            batch = buf.sample(1, augment=True)
            st = batch["states"][0]
            a = int(batch["actions"][0])
            flat_own = st[0].reshape(-1)
            flat_last = st[3].reshape(-1)
            flat_pi = batch["pis"][0]
            self.assertEqual(flat_own[a], 1.0, "动作索引必须与己方平面一致")
            self.assertEqual(flat_last[a], 1.0, "动作索引必须与上一手平面一致")
            self.assertEqual(flat_pi[a], 1.0, "策略分布必须与动作同时变换")
            self.assertAlmostEqual(float(flat_pi.sum()), 1.0, places=5)
            checked += 1
        self.assertGreater(checked, 0)

    def test_capacity_is_bounded(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(10):
            buf.add_game([make_record(i)], winner=P1)
        self.assertEqual(len(buf), 3)

    def test_empty_sample_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=3).sample(2, augment=False)


if __name__ == "__main__":
    unittest.main()
