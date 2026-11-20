"""命令行入口：play / arena / web / train / doctor。"""

from __future__ import annotations

import argparse
import random
import sys

from .arena import arena as run_arena
from .device import (
    configure_threads,
    device_spec,
    doctor_report,
    load_config,
    resolve_device,
)
from .env import BOARD_SIZE, P1, P2, Board, other, player_name
from .players import make_player


def _board_size(args, cfg):
    """棋盘尺寸优先级：CLI > 配置文件 board.size > 默认 9。"""
    if getattr(args, "board_size", None):
        return int(args.board_size)
    return int(cfg.get("board", {}).get("size", BOARD_SIZE))


# ============================================================
# 人机对局
# ============================================================
def _parse_move(text, board):
    text = text.strip().replace(",", " ").split()
    if len(text) != 2:
        raise ValueError("格式错误，应为「行 列」，例如 4 4")
    r, c = int(text[0]), int(text[1])
    if not (0 <= r < board.size and 0 <= c < board.size):
        raise ValueError(f"坐标越界（0..{board.size - 1}）")
    a = r * board.size + c
    if board.cells[a] != 0:
        raise ValueError("该位置已有棋子")
    return a


def _parse_color(text):
    """把用户输入解析成 P1/P2；无法识别返回 None。"""
    t = (text or "").strip().lower()
    if t in ("", "1", "p1", "x", "先手", "黑", "黑棋", "first"):
        return P1
    if t in ("2", "p2", "o", "后手", "白", "白棋", "second"):
        return P2
    return None


def _choose_color(args):
    """优先用 --human；未指定且是交互终端时询问，脚本管道模式默认先手。"""
    if args.human:
        return P1 if args.human == "p1" else P2
    if sys.stdin.isatty():
        while True:
            try:
                raw = input("你想执先手（P1/X，1）还是后手（P2/O，2）？[默认 1]: ")
            except EOFError:
                return P1
            color = _parse_color(raw)
            if color is not None:
                return color
            print("请输入 1 / 2（或 p1 / p2、先手 / 后手）")
    return P1


def cmd_play(args, cfg):
    from .mcts import MCTS

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    mcts_cfg = cfg.get("mcts", {})
    sims = int(args.sims or mcts_cfg.get("sims_play", 400))
    size = _board_size(args, cfg)
    device = resolve_device(device_spec(cfg.get("device", "auto"), args.device))
    configure_threads(cfg)

    if args.algo == "net":
        from .network import load_network

        net = load_network(args.model, device=device, size=size)
        player = make_player(
            f"net:{args.model}", cfg=cfg, net=net, device=device, seed=seed
        )
        algo_desc = f"神经网络 MCTS（{args.model}，device={device}）"
    else:
        player = make_player("heuristic", cfg=cfg, seed=seed)
        algo_desc = f"棋型启发式 MCTS（{sims} 次模拟）"
    player.mcts.simulations = sims

    human_color = _choose_color(args)
    ai_color = other(human_color)

    print("=== 五子棋人机对战 ===")
    print(f"棋盘: {size}x{size}")
    print(f"AI: {algo_desc}")
    print(f"你执 {player_name(human_color)}，AI 执 {player_name(ai_color)}")
    print("命令：行 列（如 4 4）、undo（悔棋）、hint（提示）、board（重绘）、quit（退出）\n")

    board = Board(size=size)
    stack = [board]
    while True:
        print(board.render())
        if board.is_terminal():
            if board.winner is None:
                print("🤝 平局")
            elif board.winner == human_color:
                print("🎉 你赢了")
            else:
                print("💻 AI 获胜")
            break

        if board.to_move == human_color:
            line = input(f"轮到你了（{player_name(human_color)}）：").strip()
            low = line.lower()
            if low in ("quit", "q", "exit"):
                print("已退出对局")
                return 0
            if low in ("undo", "u"):
                if len(stack) > 1:
                    stack.pop()
                    board = stack[-1]
                    print("已悔棋一步")
                else:
                    print("没有可悔的棋")
                continue
            if low in ("board", "b"):
                continue
            if low in ("hint", "h"):
                hint = MCTS(player.evaluator, simulations=max(100, sims // 4), rng=random.Random(seed))
                res = hint.search(board.cells, board.to_move, board.last_move)
                r, c = divmod(res.action, size)
                print(f"提示：({r}, {c})，预估价值 {res.value:+.3f}")
                continue
            try:
                a = _parse_move(line, board)
            except (ValueError, IndexError) as exc:
                print(f"⚠️ {exc}")
                continue
        else:
            print("AI 正在思考...")
            a = player.select(board)
            res = player.last_result
            r, c = divmod(a, size)
            if res is not None:
                print(
                    f"AI 落子 ({r}, {c})｜模拟 {res.visits} 次｜"
                    f"预估价值 {res.value:+.3f}｜最佳 q {res.best_q:+.3f}"
                )
            else:
                print(f"AI 落子 ({r}, {c})")

        board = board.play(a)
        stack.append(board)
    return 0


# ============================================================
# 对拉
# ============================================================
def cmd_arena(args, cfg):
    import copy

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    size = _board_size(args, cfg)
    device = resolve_device(device_spec(cfg.get("device", "auto"), args.device))
    configure_threads(cfg)
    if args.sims is not None:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("mcts", {})["sims_play"] = int(args.sims)

    a = make_player(args.a, cfg=cfg, seed=seed, device=device, size=size)
    b = make_player(args.b, cfg=cfg, seed=seed + 1, device=device, size=size)
    print(f"对拉：{args.a} vs {args.b}，{args.games} 局（{size}x{size}，交换先后手）")

    def progress(i, total, w, l, d):
        if i % max(1, total // 10) == 0 or i == total:
            print(f"  {i}/{total}  胜 {w} 负 {l} 平 {d}")

    res = run_arena(a, b, games=args.games, size=size, progress=progress)
    print(f"结果：{res}")
    return 0


# ============================================================
# 浏览器界面
# ============================================================
def cmd_web(args, cfg):
    from pathlib import Path

    from .web import Game, serve

    size = _board_size(args, cfg)
    device = resolve_device(device_spec(cfg.get("device", "auto"), args.device))
    configure_threads(cfg)

    algo = args.algo
    model = args.model
    if algo is None:
        algo = "net" if model else "heuristic"
    if algo == "net":
        if not model:
            for cand in ("models/latest.pt", "models/best.pt"):
                if Path(cand).is_file():
                    model = cand
                    break
        if not model or not Path(model).is_file():
            print(
                "错误：没找到模型文件。用 --model 指定，或加 --algo heuristic 走启发式。",
                file=sys.stderr,
            )
            return 1
        spec = f"net:{model}"
    else:
        spec = "heuristic"

    sims = int(args.sims or cfg.get("mcts", {}).get("sims_play", 400))
    game = Game(
        size=size,
        human=args.human,
        sims=sims,
        spec=spec,
        cfg=cfg,
        device=device,
        seed=int(cfg.get("seed", 0)),
    )
    serve(game, host=args.host, port=int(args.port))
    return 0


# ============================================================
# 训练
# ============================================================
def cmd_train(args, cfg):
    from .train import TrainConfig, train as run_train

    device = resolve_device(device_spec(cfg.get("device", "auto"), args.device))
    configure_threads(cfg)
    tcfg = TrainConfig.from_dict(cfg)
    if args.board_size is not None:
        tcfg.size = int(args.board_size)
    if args.iterations is not None:
        tcfg.iterations = args.iterations
    if args.games_per_iter is not None:
        tcfg.games_per_iter = args.games_per_iter
    if args.sims_train is not None:
        tcfg.sims_train = args.sims_train
    if args.seed is not None:
        tcfg.seed = args.seed
    if args.resume:
        tcfg.resume = args.resume
    if getattr(args, "fresh_buffer", False):
        tcfg.resume_buffer = False
    print(f"训练设备：{device}｜棋盘 {tcfg.size}x{tcfg.size}")
    run_train(tcfg, device=device)
    return 0


# ============================================================
# 自检
# ============================================================
def cmd_doctor(args, cfg):
    print(doctor_report(cfg, config_source=args.config or "configs/default.yaml"))
    return 0


# ============================================================
# 参数解析
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="gomoku", description="AlphaZero 风格五子棋（尺寸由配置 board.size 决定）"
    )
    p.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    sub = p.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="人机对局")
    play.add_argument("--algo", choices=["heuristic", "net"], default="heuristic")
    play.add_argument("--model", default="models/best.pt", help="--algo net 时的模型路径")
    play.add_argument(
        "--human",
        choices=["p1", "p2"],
        default=None,
        help="你执先手(p1)还是后手(p2)；不填则进入对局时询问",
    )
    play.add_argument("--sims", type=int, default=None, help="每步模拟次数")
    play.add_argument("--board-size", type=int, default=None, help="覆盖配置里的棋盘尺寸")
    play.add_argument("--device", default=None, help="cpu / cuda / cuda:0 / auto")
    play.add_argument("--seed", type=int, default=None)
    play.set_defaults(func=cmd_play)

    ar = sub.add_parser("arena", help="对拉评测")
    ar.add_argument("--a", default="heuristic", help="random / greedy / heuristic / net:path")
    ar.add_argument("--b", default="random")
    ar.add_argument("--games", type=int, default=100)
    ar.add_argument("--sims", type=int, default=None, help="每步模拟次数（默认取配置）")
    ar.add_argument("--board-size", type=int, default=None, help="覆盖配置里的棋盘尺寸")
    ar.add_argument("--device", default=None)
    ar.add_argument("--seed", type=int, default=None)
    ar.set_defaults(func=cmd_arena)

    wb = sub.add_parser("web", help="浏览器界面（下棋 + 可视化搜索）")
    wb.add_argument("--model", default=None, help="模型路径；省略则自动找 latest.pt")
    wb.add_argument("--algo", choices=["net", "heuristic"], default=None)
    wb.add_argument("--sims", type=int, default=None, help="AI 每步模拟次数")
    wb.add_argument("--port", type=int, default=8000)
    wb.add_argument("--host", default="127.0.0.1")
    wb.add_argument(
        "--human",
        choices=["p1", "p2", "none"],
        default="p1",
        help="你执黑(p1)/白(p2)；none = AI 互下",
    )
    wb.add_argument("--board-size", type=int, default=None)
    wb.add_argument("--device", default=None)
    wb.set_defaults(func=cmd_web)

    tr = sub.add_parser("train", help="自对弈训练")
    tr.add_argument("--iterations", type=int, default=None)
    tr.add_argument("--games-per-iter", type=int, default=None)
    tr.add_argument("--sims-train", type=int, default=None)
    tr.add_argument("--board-size", type=int, default=None, help="覆盖配置里的棋盘尺寸")
    tr.add_argument("--resume", default=None, help="从 checkpoint 继续训练")
    tr.add_argument(
        "--fresh-buffer",
        action="store_true",
        help="续训时不恢复回放池（只恢复网络权重）",
    )
    tr.add_argument("--device", default=None)
    tr.add_argument("--seed", type=int, default=None)
    tr.set_defaults(func=cmd_train)

    doc = sub.add_parser("doctor", help="环境自检")
    doc.set_defaults(func=cmd_doctor)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
