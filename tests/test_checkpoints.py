import os
import unittest

from gomoku.train import TrainConfig, _prune_checkpoints

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

ART = os.path.join(os.path.dirname(__file__), "_artifacts")


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestPeriodicCheckpoint(unittest.TestCase):
    def _cfg(self, tmp_dir):
        return TrainConfig(
            iterations=4,
            games_per_iter=1,
            batch_size=8,
            updates_per_iter=1,
            eval_every=10000,
            eval_games=1,
            sims_train=3,
            sims_eval=2,
            channels=8,
            blocks=1,
            models_dir=tmp_dir,
            logs_dir=tmp_dir,
            opponents={"self": 1.0},
            seed=11,
            save_every_iters=2,
            keep_checkpoints=0,
        )

    def test_checkpoints_written_at_interval(self):
        import shutil
        from pathlib import Path

        from gomoku.train import train

        d = os.path.join(ART, "ckpt_run")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        try:
            train(self._cfg(d), device="cpu")
            files = sorted(p.name for p in Path(d).glob("ckpt_*.pt"))
            # iterations=4, save_every_iters=2 -> ckpt_000002.pt, ckpt_000004.pt
            self.assertEqual(files, ["ckpt_000002.pt", "ckpt_000004.pt"])
            self.assertTrue((Path(d) / "latest.pt").is_file())
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_prune_keeps_newest(self):
        from pathlib import Path

        d = os.path.join(ART, "prune_run")
        os.makedirs(d, exist_ok=True)
        paths = []
        for i in range(1, 7):
            p = Path(d) / f"ckpt_{i:06d}.pt"
            p.write_bytes(b"x")
            os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))  # 保持时间顺序
            paths.append(p)
        try:
            _prune_checkpoints(Path(d), keep=3)
            left = sorted(p.name for p in Path(d).glob("ckpt_*.pt"))
            self.assertEqual(
                left, ["ckpt_000004.pt", "ckpt_000005.pt", "ckpt_000006.pt"]
            )
        finally:
            for p in paths:
                if p.exists():
                    p.unlink()


class TestConfigKeys(unittest.TestCase):
    def test_defaults_contain_save_keys(self):
        from gomoku.device import DEFAULTS

        self.assertIn("save_every_iters", DEFAULTS["train"])
        self.assertIn("keep_checkpoints", DEFAULTS["train"])

    def test_from_dict_reads_save_keys(self):
        cfg = {"train": {"save_every_iters": 7, "keep_checkpoints": 3}}
        tcfg = TrainConfig.from_dict(cfg)
        self.assertEqual(tcfg.save_every_iters, 7)
        self.assertEqual(tcfg.keep_checkpoints, 3)


if __name__ == "__main__":
    unittest.main()
