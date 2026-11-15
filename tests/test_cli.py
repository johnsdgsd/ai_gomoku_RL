import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from gomoku.env import P1, P2
from gomoku.cli import _choose_color, _parse_color


class TestParseColor(unittest.TestCase):
    def test_numeric(self):
        self.assertEqual(_parse_color("1"), P1)
        self.assertEqual(_parse_color("2"), P2)

    def test_names(self):
        self.assertEqual(_parse_color("p1"), P1)
        self.assertEqual(_parse_color("P2"), P2)
        self.assertEqual(_parse_color("先手"), P1)
        self.assertEqual(_parse_color("后手"), P2)
        self.assertEqual(_parse_color("x"), P1)
        self.assertEqual(_parse_color("o"), P2)

    def test_default_and_invalid(self):
        self.assertEqual(_parse_color(""), P1)
        self.assertIsNone(_parse_color("abc"))


class TestChooseColor(unittest.TestCase):
    def test_flag_overrides_prompt(self):
        self.assertEqual(_choose_color(SimpleNamespace(human="p1")), P1)
        self.assertEqual(_choose_color(SimpleNamespace(human="p2")), P2)

    def test_piped_mode_defaults_to_first(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            self.assertEqual(_choose_color(SimpleNamespace(human=None)), P1)

    def test_interactive_prompt_accepts_second(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True):
            with mock.patch("builtins.input", return_value="2"):
                self.assertEqual(_choose_color(SimpleNamespace(human=None)), P2)

    def test_interactive_prompt_reprompts_on_invalid(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True):
            with mock.patch("builtins.input", side_effect=["abc", "后手"]):
                self.assertEqual(_choose_color(SimpleNamespace(human=None)), P2)

    def test_interactive_prompt_empty_is_first(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True):
            with mock.patch("builtins.input", return_value=""):
                self.assertEqual(_choose_color(SimpleNamespace(human=None)), P1)


if __name__ == "__main__":
    unittest.main()
