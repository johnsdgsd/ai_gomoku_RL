import os
import unittest
from unittest import mock

from gomoku.device import (
    DEFAULTS,
    device_spec,
    doctor_report,
    load_config,
    resolve_device,
)

try:
    import torch

    HAS_TORCH = True
    CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    CUDA = False


class TestDevicePrecedence(unittest.TestCase):
    def test_cli_wins(self):
        with mock.patch.dict(os.environ, {"GOMOKU_DEVICE": "cpu"}):
            self.assertEqual(device_spec("auto", "cuda:0"), "cuda:0")

    def test_env_beats_config(self):
        with mock.patch.dict(os.environ, {"GOMOKU_DEVICE": "mps"}):
            self.assertEqual(device_spec("auto", None), "mps")

    def test_config_used_without_env(self):
        env = {k: v for k, v in os.environ.items() if k != "GOMOKU_DEVICE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(device_spec("cpu", None), "cpu")

    def test_fallback_auto(self):
        env = {k: v for k, v in os.environ.items() if k != "GOMOKU_DEVICE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(device_spec(None, None), "auto")


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestResolveDevice(unittest.TestCase):
    def test_cpu(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_is_available_backend(self):
        self.assertIn(resolve_device("auto"), ("cpu", "cuda", "mps"))

    @unittest.skipIf(CUDA, "本机 CUDA 可用，跳过不可用分支")
    def test_cuda_unavailable_raises_helpful_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_device("cuda")
        msg = str(ctx.exception)
        self.assertIn("CUDA", msg)
        self.assertIn("--device cpu", msg)

    def test_invalid_spec_raises(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")


class TestDoctor(unittest.TestCase):
    def test_report_contains_key_lines(self):
        cfg = load_config()
        text = doctor_report(cfg, config_source="configs/default.yaml")
        self.assertIn("python", text)
        self.assertIn("torch", text)
        self.assertIn("configs/default.yaml", text)
        self.assertIn(f"棋盘         : {DEFAULTS['board']['size']}", text)

    def test_defaults_have_expected_keys(self):
        for key in ("device", "seed", "num_threads", "board", "paths", "net", "mcts", "train"):
            self.assertIn(key, DEFAULTS)


if __name__ == "__main__":
    unittest.main()
