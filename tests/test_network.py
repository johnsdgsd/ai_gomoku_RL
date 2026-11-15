import os
import tempfile
import unittest

import numpy as np

from gomoku.env import BOARD_SIZE, N_CELLS
from gomoku.network import Network, build_network, load_network, FORMAT_VERSION

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "需要 torch")
class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.net = build_network(channels=16, blocks=2)

    def test_output_shapes(self):
        x = torch.randn(3, 4, BOARD_SIZE, BOARD_SIZE)
        logits, value = self.net(x)
        self.assertEqual(tuple(logits.shape), (3, N_CELLS))
        self.assertEqual(tuple(value.shape), (3,))

    def test_value_in_range(self):
        x = torch.randn(8, 4, BOARD_SIZE, BOARD_SIZE) * 3
        _, value = self.net(x)
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_finite_outputs(self):
        x = torch.zeros(1, 4, BOARD_SIZE, BOARD_SIZE)
        logits, value = self.net(x)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_gradients_flow(self):
        x = torch.randn(4, 4, BOARD_SIZE, BOARD_SIZE)
        logits, value = self.net(x)
        loss = logits.pow(2).mean() + value.pow(2).mean()
        loss.backward()
        for name, p in self.net.model.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} 没有梯度")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name} 梯度非有限")

    def test_param_count_in_range(self):
        n = self.net.count_parameters()
        self.assertGreaterEqual(n, 10_000)
        self.assertLessEqual(n, 500_000, f"参数量 {n} 超出 CPU 友好范围")

    def test_forward_numpy_single_and_batch(self):
        planes = np.random.rand(4, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        logits, v = self.net.forward_numpy(planes)
        self.assertEqual(logits.shape, (N_CELLS,))
        self.assertIsInstance(v, float)

        batch = np.random.rand(5, 4, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        logits, v = self.net.forward_numpy(batch)
        self.assertEqual(logits.shape, (5, N_CELLS))
        self.assertEqual(v.shape, (5,))

    def test_save_load_roundtrip(self):
        x = torch.randn(1, 4, BOARD_SIZE, BOARD_SIZE)
        with torch.no_grad():
            logits, value = self.net(x)

        d = os.path.join(os.path.dirname(__file__), "_artifacts")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "test_net.pt")
        try:
            self.net.save(path, extra={"iteration": 3})
            loaded = load_network(path)
            with torch.no_grad():
                logits2, value2 = loaded(x)
            self.assertTrue(torch.allclose(logits, logits2, atol=1e-6))
            self.assertTrue(torch.allclose(value, value2, atol=1e-6))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_missing_file_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_network("models/__not_exists__.pt")
        self.assertIn("train", str(ctx.exception).lower())

    def test_format_version(self):
        self.assertEqual(FORMAT_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
