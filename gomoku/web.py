"""浏览器界面：下棋 + 可视化 MCTS 的思考过程。

用法：
    python -m gomoku web --model models/latest.pt --sims 1000

设计约束：
  - 完全在训练流程之外，只读取一个模型文件（或走启发式），不写任何训练产物。
  - 零第三方依赖：只用标准库 http.server，前端是一个内嵌的静态页面。
  - 所有 AI 计算都在请求线程里同步完成（1000 次模拟约 1~2 秒），
    因此不需要后台线程，也就不存在并发改棋盘的竞态；Game 内部仍加锁防御。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .env import EMPTY, P1, P2, Board, other
from .players import make_player

MAX_CANDIDATES = 8


class Game:
    """一局棋的状态机：棋盘、着法历史、每手的搜索详情。"""

    def __init__(
        self,
        size: int = 15,
        human: str = "p1",
        sims: int = 1000,
        spec: str = "heuristic",
        cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.size = int(size)
        self.human = human  # "p1" / "p2" / "none"（AI 互下）
        self.sims = int(sims)
        self.spec = spec
        self.cfg = cfg or {}
        self.device = device
        self.seed = seed
        self.lock = threading.RLock()  # 可重入：new_game() 内部还会调 state()
        self.board = Board(size=self.size)
        self.history: List[Dict[str, Any]] = []
        self._player = self._build_player(self.seed)

    # ---------------- 内部 ----------------
    def _build_player(self, seed: int):
        p = make_player(
            self.spec, cfg=self.cfg, seed=seed, device=self.device, size=self.size
        )
        p.mcts.simulations = self.sims
        return p

    def human_color(self) -> Optional[int]:
        if self.human in (None, "none", ""):
            return None
        return P1 if self.human == "p1" else P2

    def _replay(self):
        b = Board(size=self.size)
        for h in self.history:
            b = b.play(h["action"])
        self.board = b

    def _snapshot(self, action: int, to_move: int, res) -> Dict[str, Any]:
        cands: List[Dict[str, Any]] = []
        if res is not None and getattr(res, "root", None) is not None:
            stats = res.root.stats or {}
            priors = res.root.priors or {}
            for a, st in stats.items():
                if st.N <= 0:
                    continue
                cands.append(
                    {
                        "a": int(a),
                        "n": int(st.N),
                        "q": round(float(st.q), 4),
                        "pi": round(float(priors.get(a, 0.0)), 4),
                    }
                )
            cands.sort(key=lambda d: -d["n"])
            cands = cands[:MAX_CANDIDATES]
        value = float(res.value) if res is not None else 0.0
        return {
            "action": int(action),
            "by": int(to_move),
            "value": round(value, 4),
            "value_black": round(value if to_move == P1 else -value, 4),
            "best_q": round(float(res.best_q), 4) if res is not None else 0.0,
            "visits": int(res.visits) if res is not None else 0,
            "candidates": cands,
        }

    # ---------------- 对外操作 ----------------
    def state(self) -> Dict[str, Any]:
        with self.lock:
            b = self.board
            return {
                "size": self.size,
                "cells": [int(x) for x in b.cells],
                "to_move": int(b.to_move),
                "winner": None if b.winner is None else int(b.winner),
                "terminal": bool(b.is_terminal()),
                "human": self.human,
                "sims": self.sims,
                "spec": self.spec,
                "history": list(self.history),
                "last": self.history[-1] if self.history else None,
            }

    def new_game(self, human: Optional[str] = None, sims: Optional[int] = None):
        with self.lock:
            if human:
                self.human = human
            if sims:
                self.sims = max(20, int(sims))
                self._player.mcts.simulations = self.sims
            self.history = []
            self.board = Board(size=self.size)
            return self.state()

    def ai_step(self) -> Dict[str, Any]:
        """让 AI 走一步；棋局已结束或不该它走时什么也不做。"""
        with self.lock:
            b = self.board
            if b.is_terminal():
                return {"error": "对局已结束"}
            hc = self.human_color()
            if hc is not None and b.to_move == hc:
                return {"error": "现在轮到你走"}
            a = self._player.select(b)
            res = getattr(self._player, "last_result", None)
            self.history.append(self._snapshot(a, b.to_move, res))
            self.board = b.play(a)
            return {}

    def human_move(self, row, col) -> Dict[str, Any]:
        with self.lock:
            b = self.board
            if b.is_terminal():
                return {"error": "对局已结束"}
            hc = self.human_color()
            if hc is None:
                return {"error": "当前是 AI 互下模式，请用「AI 走一步」"}
            if b.to_move != hc:
                return {"error": "还没轮到你"}
            try:
                r, c = int(row), int(col)
            except (TypeError, ValueError):
                return {"error": "坐标不合法"}
            if not (0 <= r < self.size and 0 <= c < self.size):
                return {"error": "坐标超出棋盘"}
            a = r * self.size + c
            if b.cells[a] != EMPTY:
                return {"error": "该位置已有棋子"}
            self.history.append(
                {
                    "action": a,
                    "by": int(hc),
                    "value": None,
                    "value_black": None,
                    "best_q": None,
                    "visits": 0,
                    "candidates": [],
                }
            )
            self.board = b.play(a)
            # 人走完立刻让 AI 应手（同步，约 1~2 秒）
            if not self.board.is_terminal():
                return self.ai_step_locked()
            return {}

    def ai_step_locked(self) -> Dict[str, Any]:
        """调用前必须已持有 self.lock。"""
        b = self.board
        a = self._player.select(b)
        res = getattr(self._player, "last_result", None)
        self.history.append(self._snapshot(a, b.to_move, res))
        self.board = b.play(a)
        return {}

    def hint(self) -> Dict[str, Any]:
        """只算不落子，用来给人类提示 / 预览它的候选。"""
        with self.lock:
            b = self.board
            if b.is_terminal():
                return {"error": "对局已结束"}
            a = self._player.select(b)
            res = getattr(self._player, "last_result", None)
            snap = self._snapshot(a, b.to_move, res)
            return {"hint": snap}

    def undo(self) -> Dict[str, Any]:
        with self.lock:
            if not self.history:
                return {"error": "没有可悔的棋"}
            hc = self.human_color()
            self.history.pop()
            self._replay()
            # 悔到轮到人类为止（人机对局通常要退两步）
            if hc is not None and not self.board.is_terminal():
                if self.board.to_move != hc and self.history:
                    self.history.pop()
                    self._replay()
            return {}

    def export(self) -> str:
        with self.lock:
            lines = [
                f"# gomoku  size={self.size}  spec={self.spec}  sims={self.sims}",
                f"# 共 {len(self.history)} 手",
            ]
            for i, h in enumerate(self.history, 1):
                r, c = divmod(h["action"], self.size)
                mark = "X" if h["by"] == P1 else "O"
                extra = ""
                if h.get("value_black") is not None:
                    extra = f"   V(黑)={h['value_black']:+.3f} 访问={h['visits']}"
                lines.append(f"{i:3d}. {mark} ({r},{c}){extra}")
            w = self.board.winner
            if w is None:
                lines.append(f"# 结果：{'进行中' if not self.board.is_terminal() else '平局'}")
            else:
                lines.append(f"# 结果：{'黑(X)胜' if w == P1 else '白(O)胜'}")
            lines.append("# 坐标串：" + " ".join(f"({h['action'] // self.size},{h['action'] % self.size})" for h in self.history))
            return "\n".join(lines) + "\n"


# ============================================================
# 前端（单页，无依赖）
# ============================================================
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>五子棋 AI</title>
<style>
  :root{
    --bg:#12141a; --panel:#1b1e27; --line:#2b2f3c; --fg:#e8eaf0; --dim:#8b93a7;
    --accent:#6ea8fe; --x:#e8eaf0; --o:#4a5265; --warn:#f0a35e;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 "PingFang SC","Microsoft YaHei",system-ui,sans-serif}
  .wrap{display:flex;gap:18px;padding:18px;flex-wrap:wrap;justify-content:center}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
  .board-wrap{flex:0 0 auto}
  canvas{display:block;border-radius:8px;cursor:pointer}
  .bar{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
  button{background:#252a36;color:var(--fg);border:1px solid var(--line);
         border-radius:8px;padding:7px 13px;cursor:pointer;font-size:13px}
  button:hover{background:#2e3444}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.primary{background:var(--accent);color:#0d1017;border-color:var(--accent);font-weight:600}
  .right{flex:0 0 340px;display:flex;flex-direction:column;gap:14px}
  h3{margin:0 0 10px;font-size:13px;font-weight:600;color:var(--dim);letter-spacing:.4px}
  .vnum{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
  .sub{color:var(--dim);font-size:12px}
  .cand{display:grid;grid-template-columns:56px 1fr 62px;gap:8px;align-items:center;margin-bottom:7px}
  .cbar{height:8px;background:#252a36;border-radius:4px;overflow:hidden}
  .cbar>i{display:block;height:100%;background:var(--accent);border-radius:4px}
  .mono{font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums}
  .status{margin-top:10px;color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
  .row{display:flex;gap:8px;align-items:center;margin-bottom:8px}
  select,input{background:#252a36;color:var(--fg);border:1px solid var(--line);
               border-radius:8px;padding:6px 8px;font-size:13px}
  input[type=number]{width:88px}
</style>
</head>
<body>
<div class="wrap">
  <div>
    <div class="panel board-wrap">
      <canvas id="board" width="620" height="620"></canvas>
      <div class="bar">
        <button class="primary" id="btn-new">新局</button>
        <button id="btn-undo">悔棋</button>
        <button id="btn-hint">提示</button>
        <button id="btn-step">AI 走一步</button>
        <button id="btn-export">导出棋谱</button>
      </div>
      <div class="status" id="status">加载中…</div>
    </div>
  </div>

  <div class="right">
    <div class="panel">
      <h3>局面评估（黑方视角）</h3>
      <div class="vnum mono" id="v-now">—</div>
      <div class="sub" id="v-txt">等 AI 走一步后显示</div>
      <canvas id="chart" width="312" height="110" style="margin-top:10px"></canvas>
    </div>

    <div class="panel">
      <h3>MCTS 候选（访问次数）</h3>
      <div id="cands" class="sub">—</div>
    </div>

    <div class="panel">
      <h3>设置</h3>
      <div class="row">
        <span class="sub">你执</span>
        <select id="sel-human">
          <option value="p1">黑（先手）</option>
          <option value="p2">白（后手）</option>
          <option value="none">AI 互下</option>
        </select>
        <span class="sub">模拟</span>
        <input type="number" id="inp-sims" value="1000" min="20" step="100">
      </div>
      <div class="sub" id="cfg-line">—</div>
    </div>
  </div>
</div>

<script>
let st = null, busy = false, hintCells = [];

const $ = id => document.getElementById(id);
const boardCv = $("board"), chartCv = $("chart");
const bx = boardCv.getContext("2d");
const cx = chartCv.getContext("2d");

function fitCanvas(cv, cssW, cssH){
  const dpr = window.devicePixelRatio || 1;
  cv.width = cssW * dpr; cv.height = cssH * dpr;
  cv.style.width = cssW + "px"; cv.style.height = cssH + "px";
  cv.getContext("2d").setTransform(dpr,0,0,dpr,0,0);
  return [cssW, cssH];
}

let W = 620, PAD = 26, CELL = 0;
function layout(){
  W = Math.min(620, Math.max(320, window.innerWidth - 60));
  [W] = fitCanvas(boardCv, W, W);
  PAD = 26; CELL = (W - PAD*2) / 14;
  fitCanvas(chartCv, 312, 110);
}

function xy(idx){ const S = st.size, r = Math.floor(idx/S), c = idx%S;
  return [PAD + c*CELL, PAD + r*CELL]; }

function draw(){
  if(!st) return;
  const S = st.size;
  CELL = (W - PAD*2) / (S-1);
  bx.fillStyle = "#1b1e27"; bx.fillRect(0,0,W,W);
  bx.strokeStyle = "#3a4052"; bx.lineWidth = 1;
  for(let i=0;i<S;i++){
    const p = PAD + i*CELL;
    bx.beginPath(); bx.moveTo(PAD,p); bx.lineTo(PAD+(S-1)*CELL,p); bx.stroke();
    bx.beginPath(); bx.moveTo(p,PAD); bx.lineTo(p,PAD+(S-1)*CELL); bx.stroke();
  }
  // 坐标
  bx.fillStyle = "#5d6579"; bx.font = "10px ui-monospace,monospace";
  bx.textAlign = "center"; bx.textBaseline = "middle";
  for(let i=0;i<S;i++){
    bx.fillText(i, PAD + i*CELL, PAD - 13);
    bx.fillText(i, PAD - 14, PAD + i*CELL);
  }
  // 热力图：候选点的访问占比
  const cands = (st.last && st.last.candidates) || [];
  if(cands.length){
    const tot = cands.reduce((s,c)=>s+c.n,0) || 1;
    for(const c of cands){
      const r = Math.floor(c.a/S), cc = c.a%S;
      const x = PAD + cc*CELL, y = PAD + r*CELL;
      const ratio = c.n/tot;
      bx.fillStyle = "rgba(110,168,254," + (0.10 + 0.55*Math.min(1,ratio*1.6)) + ")";
      bx.beginPath();
      bx.arc(x, y, CELL*0.44, 0, Math.PI*2);
      bx.fill();
    }
  }
  for(const h of hintCells){
    const r = Math.floor(h/S), cc = h%S;
    bx.strokeStyle = "#f0a35e"; bx.lineWidth = 2;
    bx.beginPath(); bx.moveTo(PAD+cc*CELL-6, PAD+r*CELL-6); bx.lineTo(PAD+cc*CELL+6, PAD+r*CELL+6);
    bx.moveTo(PAD+cc*CELL+6, PAD+r*CELL-6); bx.lineTo(PAD+cc*CELL-6, PAD+r*CELL+6); bx.stroke();
  }
  // 棋子
  for(let i=0;i<st.cells.length;i++){
    const v = st.cells[i]; if(!v) continue;
    const r = Math.floor(i/S), c = i%S;
    const x = PAD + c*CELL, y = PAD + r*CELL;
    bx.beginPath(); bx.arc(x, y, CELL*0.40, 0, Math.PI*2);
    bx.fillStyle = v===1 ? "#e9ecf3" : "#454d61";
    bx.fill();
    bx.strokeStyle = v===1 ? "#9aa4bb" : "#2a3040"; bx.lineWidth = 1; bx.stroke();
  }
  // 最后一手标记
  if(st.history && st.history.length){
    const a = st.history[st.history.length-1].action;
    const [x,y] = xy(a);
    bx.strokeStyle = "#6ea8fe"; bx.lineWidth = 2.5;
    bx.beginPath(); bx.arc(x, y, CELL*0.46, 0, Math.PI*2); bx.stroke();
  }
}

function drawChart(){
  fitCanvas(chartCv, 312, 110);
  cx.clearRect(0,0,312,110);
  const pts = (st.history||[]).filter(h => h.value_black !== null);
  cx.strokeStyle = "#2b2f3c"; cx.lineWidth = 1;
  cx.beginPath(); cx.moveTo(0,55); cx.lineTo(312,55); cx.stroke();
  if(pts.length < 2) return;
  const n = pts.length;
  cx.beginPath();
  pts.forEach((p,i)=>{
    const x = n>1 ? (i/(n-1))*306 + 3 : 3;
    const y = 55 - p.value_black*45;
    i ? cx.lineTo(x,y) : cx.moveTo(x,y);
  });
  cx.strokeStyle = "#6ea8fe"; cx.lineWidth = 2; cx.stroke();
  const lastP = pts[n-1];
  cx.fillStyle = "#6ea8fe";
  cx.beginPath(); cx.arc((n>1?306:3)+3, 55 - lastP.value_black*45, 3.5, 0, Math.PI*2); cx.fill();
}

function renderCands(){
  const el = $("cands");
  const cands = (st.last && st.last.candidates) || [];
  if(!cands.length){ el.innerHTML = '<span class="sub">——</span>'; return; }
  const tot = cands.reduce((s,c)=>s+c.n,0) || 1;
  el.innerHTML = cands.map(c=>{
    const r = Math.floor(c.a/st.size), cc = c.a%st.size;
    const pct = (c.n/tot*100);
    return '<div class="cand"><span class="mono">('+r+','+cc+')</span>'+
      '<span class="cbar"><i style="width:'+Math.max(2,pct).toFixed(1)+'%"></i></span>'+
      '<span class="mono sub">'+c.n+' · '+pct.toFixed(0)+'%</span></div>';
  }).join("");
}

function render(){
  if(!st) return;
  draw(); drawChart(); renderCands();
  $("sel-human").value = st.human || "p1";
  $("inp-sims").value = st.sims;
  $("cfg-line").textContent = st.spec + " · " + st.size + "×" + st.size;
  const last = st.last;
  if(last && last.value_black !== null){
    $("v-now").textContent = (last.value_black>=0?"+":"") + last.value_black.toFixed(2);
    $("v-txt").textContent = "黑方 " + (last.value_black>=0?"占优":"劣势") +
      " · 模拟 " + last.visits + " 次 · q=" + last.best_q.toFixed(2);
  }
  const human = st.human;
  let msg;
  if(st.terminal) msg = st.winner === null ? "平局" :
      (st.winner === 1 ? "黑胜" : "白胜");
  else if(human === "none") msg = "AI 互下中 · 点「AI 走一步」";
  else msg = (st.cells.filter(v=>v).length % 2 === 0 ? "轮到黑" : "轮到白");
  $("status").textContent = msg + " · 已下 " + st.history.length + " 手";
  ["btn-undo","btn-hint","btn-step"].forEach(id=>{
    $(id).disabled = !!(busy || st.terminal ||
      (id==="btn-step" && human!=="none") || (id==="btn-hint" && human==="none"));
  });
}

async function api(path, body){
  const opt = body === undefined ? {} :
    {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)};
  const r = await fetch(path, opt);
  return await r.json();
}

async function refresh(){ st = await api("/api/state"); render(); }

async function withBusy(fn){
  if(busy) return;
  busy = true; $("status").textContent = "AI 思考中…"; render();
  try{ await fn(); } finally { busy = false; }
  await refresh();
}

boardCv.addEventListener("click", async ev=>{
  if(!st || busy || st.terminal || st.human === "none") return;
  const rect = boardCv.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const c = Math.round((x - PAD)/CELL), r = Math.round((y - PAD)/CELL);
  if(r<0||c<0||r>=st.size||c>=st.size) return;
  await withBusy(async ()=>{
    const res = await api("/api/move", {row:r, col:c});
    if(res.error){ $("status").textContent = res.error; }
  });
  hintCells = [];
});

$("btn-new").onclick = async ()=>{
  await withBusy(async ()=>{
    await api("/api/new", {human: $("sel-human").value,
                           sims: parseInt($("inp-sims").value,10)});
  });
  hintCells = [];
};
$("btn-undo").onclick = ()=> withBusy(async ()=>{ await api("/api/undo", {}); });
$("btn-step").onclick = ()=> withBusy(async ()=>{ await api("/api/step", {}); });
$("btn-hint").onclick = async ()=>{
  if(busy) return; busy = true; $("status").textContent = "AI 计算提示…";
  try{
    const res = await api("/api/hint", {});
    if(res.hint){
      hintCells = [res.hint.action];
      st.last = res.hint; render();
      const r = Math.floor(res.hint.action/st.size), c = res.hint.action%st.size;
      $("status").textContent = "建议落子 ("+r+","+c+")  价值 " + res.hint.value_black.toFixed(2);
    } else { $("status").textContent = res.error || "无提示"; }
  } finally { busy = false; render(); }
};
$("btn-export").onclick = ()=>{ window.open("/api/export", "_blank"); };
window.addEventListener("resize", ()=>{ layout(); render(); });

layout(); refresh();
</script>
</body>
</html>
"""


# ============================================================
# HTTP 服务
# ============================================================
def make_handler(game: Game):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gomoku-web"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 安静一点
            pass

        def _send(self, code: int, body: str, ctype: str):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def _read_body(self) -> Dict[str, Any]:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return {}
            raw = self.rfile.read(n) if n > 0 else b""
            try:
                out = json.loads(raw.decode("utf-8") or "{}")
                return out if isinstance(out, dict) else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(game.state())
            elif path == "/api/export":
                self._send(200, game.export(), "text/plain; charset=utf-8")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            body = self._read_body()
            if path == "/api/new":
                game.new_game(human=body.get("human"), sims=body.get("sims"))
                self._json(game.state())
            elif path == "/api/move":
                res = game.human_move(body.get("row"), body.get("col"))
                if res.get("error"):
                    self._json(res, 400)
                else:
                    self._json(game.state())
            elif path == "/api/step":
                res = game.ai_step()
                if res.get("error"):
                    self._json(res, 400)
                else:
                    self._json(game.state())
            elif path == "/api/undo":
                res = game.undo()
                if res.get("error"):
                    self._json(res, 400)
                else:
                    self._json(game.state())
            elif path == "/api/hint":
                res = game.hint()
                self._json(res, 400 if res.get("error") else 200)
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def serve(
    game: Game,
    host: str = "127.0.0.1",
    port: int = 8000,
    verbose: bool = True,
):
    httpd = ThreadingHTTPServer((host, port), make_handler(game))
    httpd.daemon_threads = True
    real_port = httpd.server_address[1]
    if verbose:
        print("=" * 54)
        print(f"五子棋 Web 界面已启动：http://{host}:{real_port}")
        print(f"  对手 = {game.spec} · 每步 {game.sims} 次模拟 · 棋盘 {game.size}×{game.size}")
        print("  按 Ctrl+C 停止")
        print("=" * 54, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return httpd
