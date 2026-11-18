"""回放池持久化：save / load 往返、容量收缩、续训自动恢复。"""

import contextlib
import io
import os
import shutil
import unittest

import numpy as np

from gomoku.replay import Record, ReplayBuffer

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

ART = os.path.join(os.path.dirname(__file__), "_artifacts")
SIZE = 9


def _rec(i: int, size: int = SIZE) -> Record:
    state = np.zeros((4, size, size), dtype=np.uint8)
    flat = state.reshape(4, -1)
    flat[0][i % (size * size)] = 1
    flat[2][i % (size * size)] = 1
    pi = np.zeros(size * size, dtype=np.float32)
    pi[i % (size * size)] = 1.0
    return Record(
        state=state,
        action=i % (size * size),
        pi=pi,
        vbar=0.5 * i,
        to_move=1 + (i % 2),
        z=1.0 if i % 2 else -1.0,
    )


class TestBufferRoundTrip(unittest.TestCase):
    def setUp(self):
        os.makedirs(ART, exist_ok=True)
        self.path = os.path.join(ART, "buffer_roundtrip.npz")

    def tearDown(self):
        for p in (self.path, self.path + ".tmp"):
            if os.path.exists(p):
                os.unlink(p)

    def test_save_load_preserves_records(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(17):
            buf.add_record(_rec(i))
        self.assertEqual(buf.save(self.path), 17)
        self.assertTrue(os.path.isfile(self.path))

        back = ReplayBuffer.load(self.path)
        self.assertEqual(len(back), 17)
        self.assertEqual(back.buf[0].action, 0)
        self.assertEqual(back.buf[16].action, 16)
        np.testing.assert_allclose(back.buf[5].pi, buf.buf[5].pi)
        self.assertAlmostEqual(back.buf[4].vbar, 2.0)
        self.assertEqual(back.buf[3].z, 1.0)
        self.assertEqual(back.buf[2].z, -1.0)
        self.assertEqual(back.buf[0].state.dtype, np.uint8)
        np.testing.assert_array_equal(back.buf[6].state, buf.buf[6].state)

    def test_capacity_smaller_keeps_newest(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.add_record(_rec(i))
        buf.save(self.path)
        back = ReplayBuffer.load(self.path, capacity=5)
        self.assertEqual(len(back), 5)
        # 保留的是最新的 5 条（15..19）
        self.assertEqual([r.action for r in back.buf], [15, 16, 17, 18, 19])

    def test_sample_works_after_load(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(32):
            buf.add_record(_rec(i))
        buf.save(self.path)
        back = ReplayBuffer.load(self.path)
        batch = back.sample(8, augment=True, lambda_vbar=0.2)
        self.assertEqual(batch["states"].shape, (8, 4, SIZE, SIZE))
        self.assertEqual(batch["states"].dtype, np.float32)
        self.assertEqual(batch["pis"].shape, (8, SIZE * SIZE))
        self.assertEqual(batch["targets"].shape, (8,))

    def test_empty_buffer_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=10).save(self.path)

    def test_no_tmp_file_left(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.add_record(_rec(i))
        buf.save(self.path)
        self.assertFalse(os.path.exists(self.path + ".tmp"))


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestBufferResumeInTraining(unittest.TestCase):
    def _cfg(self, d, **kw):
        from gomoku.train import TrainConfig

        base = dict(
            iterations=2,
            games_per_iter=1,
            batch_size=8,
            updates_per_iter=1,
            eval_every=10000,
            eval_games=1,
            sims_train=3,
            sims_eval=2,
            channels=8,
            blocks=1,
            size=SIZE,
            models_dir=d,
            logs_dir=d,
            seed=5,
            save_every_iters=1,
            keep_checkpoints=0,
            heuristic_warmup_iters=0,
        )
        base.update(kw)
        return TrainConfig(**base)

    def test_resume_restores_buffer(self):
        from gomoku.train import train

        d = os.path.join(ART, "buffer_resume")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        try:
            train(self._cfg(d), device="cpu")
            bpath = os.path.join(d, "buffer.npz")
            self.assertTrue(os.path.isfile(bpath))
            n1 = len(ReplayBuffer.load(bpath))
            self.assertGreater(n1, 0)

            latest = os.path.join(d, "latest.pt")
            self.assertTrue(os.path.isfile(latest))

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                train(self._cfg(d, iterations=1, resume=latest), device="cpu")
            self.assertIn("恢复回放池", out.getvalue())
            n2 = len(ReplayBuffer.load(bpath))
            self.assertGreater(n2, n1)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_fresh_buffer_flag_skips_restore(self):
        from gomoku.train import train

        d = os.path.join(ART, "buffer_fresh")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        try:
            train(self._cfg(d), device="cpu")
            latest = os.path.join(d, "latest.pt")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                train(
                    self._cfg(d, iterations=1, resume=latest, resume_buffer=False),
                    device="cpu",
                )
            self.assertNotIn("恢复回放池", out.getvalue())
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
