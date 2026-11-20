"""Web 界面：起一个本地服务，用 HTTP 打一遍主要接口。

不依赖 torch：用启发式对手 + 小棋盘 + 少量模拟。
"""

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from gomoku.web import Game, make_handler


class WebCase(unittest.TestCase):
    def setUp(self):
        self.game = Game(size=9, human="p1", sims=30, spec="heuristic", cfg={})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.game))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    # ---------- 工具 ----------
    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read().decode("utf-8")
        conn.close()
        return r.status, body, r.getheader("Content-Type") or ""

    def post(self, path, payload=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        body = json.dumps(payload or {})
        conn.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
        r = conn.getresponse()
        raw = r.read().decode("utf-8")
        conn.close()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"_raw": raw}
        return r.status, data

    # ---------- 用例 ----------
    def test_index_is_html(self):
        status, body, ctype = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("五子棋", body)
        self.assertIn("/api/state", body)

    def test_state_shape(self):
        status, body, _ = self.get("/api/state")
        self.assertEqual(status, 200)
        st = json.loads(body)
        self.assertEqual(st["size"], 9)
        self.assertEqual(len(st["cells"]), 81)
        self.assertEqual(st["to_move"], 1)
        self.assertEqual(st["history"], [])
        self.assertFalse(st["terminal"])
        self.assertEqual(st["human"], "p1")

    def test_move_triggers_ai_reply(self):
        status, st = self.post("/api/move", {"row": 4, "col": 4})
        self.assertEqual(status, 200)
        self.assertEqual(len(st["history"]), 2)  # 人一步 + 机器一步
        self.assertEqual(st["history"][0]["by"], 1)
        self.assertEqual(st["history"][1]["by"], 2)
        self.assertEqual(st["history"][0]["action"], 4 * 9 + 4)
        self.assertIsNotNone(st["history"][1]["value_black"])

    def test_move_rejects_occupied(self):
        self.post("/api/move", {"row": 4, "col": 4})
        status, data = self.post("/api/move", {"row": 4, "col": 4})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_move_rejects_out_of_board(self):
        status, data = self.post("/api/move", {"row": 99, "col": 0})
        self.assertEqual(status, 400)

    def test_move_rejects_bad_coords(self):
        status, _ = self.post("/api/move", {"row": "x", "col": None})
        self.assertEqual(status, 400)

    def test_undo_returns_to_human_turn(self):
        self.post("/api/move", {"row": 4, "col": 4})
        status, st = self.post("/api/undo", {})
        self.assertEqual(status, 200)
        self.assertEqual(st["history"], [])
        self.assertEqual(st["to_move"], 1)

    def test_undo_on_empty_game(self):
        status, data = self.post("/api/undo", {})
        self.assertEqual(status, 400)

    def test_new_game_resets(self):
        self.post("/api/move", {"row": 4, "col": 4})
        status, st = self.post("/api/new", {"human": "p1", "sims": 40})
        self.assertEqual(status, 200)
        self.assertEqual(st["history"], [])
        self.assertEqual(st["sims"], 40)

    def test_hint_does_not_play(self):
        status, data = self.post("/api/hint", {})
        self.assertEqual(status, 200)
        self.assertIn("hint", data)
        self.assertIn("action", data["hint"])
        st = json.loads(self.get("/api/state")[1])
        self.assertEqual(st["history"], [])

    def test_export_contains_moves(self):
        self.post("/api/move", {"row": 4, "col": 4})
        status, body, ctype = self.get("/api/export")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", ctype)
        self.assertIn("共 2 手", body)
        self.assertIn("坐标串", body)

    def test_404(self):
        status, _, _ = self.get("/nope")
        self.assertEqual(status, 404)


class AiVsAiCase(unittest.TestCase):
    def setUp(self):
        self.game = Game(size=9, human="none", sims=30, spec="heuristic", cfg={})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.game))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def post(self, path, payload=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request(
            "POST",
            path,
            body=json.dumps(payload or {}),
            headers={"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        data = json.loads(r.read().decode("utf-8"))
        conn.close()
        return r.status, data

    def test_step_plays_one_move(self):
        status, st = self.post("/api/step", {})
        self.assertEqual(status, 200)
        self.assertEqual(len(st["history"]), 1)

    def test_human_move_rejected_in_ai_mode(self):
        status, data = self.post("/api/move", {"row": 4, "col": 4})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
