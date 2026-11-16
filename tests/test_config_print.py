import unittest

from gomoku.train import TrainConfig, _format_config


class TestFormatConfig(unittest.TestCase):
    def test_contains_key_fields(self):
        cfg = TrainConfig(
            size=15,
            channels=32,
            blocks=3,
            sims_train=200,
            dir_eps=0.05,
            search_margin=2,
            heuristic_warmup_iters=50,
            opponent_sims=32,
            eval_vs_heuristic_sims=300,
            updates_per_iter=100,
            lr=0.002,
            opening_moves=0,
            pi_min_visits=0,
        )
        text = _format_config(cfg, "cuda", start_iteration=7)
        for token in [
            "board.size=15",
            "channels=32",
            "blocks=3",
            "sims_train=200",
            "dir_eps=0.05",
            "search_margin=2",
            "heuristic_warmup_iters=50",
            "opponent_sims=32",
            "eval_vs_heuristic_sims=300",
            "updates_per_iter=100",
            "device=cuda",
            "起始迭代 7",
            "先手比例=50%",
        ]:
            self.assertIn(token, text, f"配置输出缺少 {token}")

    def test_full_board_margin(self):
        cfg = TrainConfig(search_margin=-1)
        text = _format_config(cfg, "cpu")
        self.assertIn("全棋盘", text)

    def test_no_attribute_errors_for_defaults(self):
        text = _format_config(TrainConfig(), "cpu")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 200)


if __name__ == "__main__":
    unittest.main()
