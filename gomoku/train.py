"""训练主循环：自对弈数据生成 + 策略/价值监督 + 评测 + 存档/续训。

关键约定：
  - 回放池只存真实对局路径；
  - 策略标签是 MCTS 访问分布 π（软分布，不是 one-hot）；
  - 价值标签 = (1-λ)·z + λ·V̄，z 为真实终局信号；
  - MCTS 内部不产生梯度，训练是独立的离线循环。
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .arena import arena as run_arena
from .device import DEFAULTS
from .env import BOARD_SIZE, P1, P2
from .evaluators import HeuristicEvaluator, NetEvaluator
from .network import Network, build_network
from .players import MCTSPlayer
from .replay import ReplayBuffer
from .selfplay import SelfPlayConfig, play_selfplay


def _torch():
    import torch

    return torch


@dataclass
class TrainConfig:
    iterations: int = 50
    games_per_iter: int = 20
    batch_size: int = 128
    updates_per_iter: int = 50
    lr: float = 0.002
    weight_decay: float = 0.0001
    c_policy: float = 1.0
    lambda_vbar: float = 0.2
    replay_capacity: int = 100_000
    eval_every: int = 100
    eval_games: int = 20
    sims_train: int = 150
    sims_eval: int = 100
    seed: int = 0
    size: int = BOARD_SIZE
    channels: int = 32
    blocks: int = 3
    c_puct: float = 1.5
    root_noise: bool = True
    dir_eps: float = 0.05          # 根节点噪声权重（越小越信网络先验）
    temp_moves: int = 15
    temp: float = 1.0
    opponents: Dict[str, float] = field(
        default_factory=lambda: {"self": 0.6, "heuristic": 0.3, "random": 0.1}
    )
    # 训练对手策略：前 N 轮用启发式陪练，之后纯自对弈（N=0 表示全程自对弈）
    heuristic_warmup_iters: int = 50
    # 训练对手（启发式）每步搜索次数：0=不走搜索；越大对手越强也越慢
    opponent_sims: int = 32
    # 可落子范围："棋子外接正方形再外扩 margin 圈"；<0 表示全棋盘
    search_margin: int = -1
    # 开局跳过：训练时先用启发式随机走 K 手（0 关闭）
    opening_moves: int = 0
    opening_max: int = 0
    # 策略标签截断：访问次数低于该值的动作清零后重归一化（0 关闭）
    pi_min_visits: int = 0
    # 评测对手（永久固定为启发式 MCTS），以及其搜索次数
    eval_vs_heuristic_sims: int = 300
    models_dir: str = "models"
    logs_dir: str = "logs"
    resume: Optional[str] = None
    save_every_iters: int = 5      # 每 N 轮额外保存一个带轮次的快照；0 表示关闭
    keep_checkpoints: int = 5      # 最多保留几个轮次快照（0 表示全部保留）

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TrainConfig":
        m = cfg.get("mcts", {})
        t = cfg.get("train", {})
        net = cfg.get("net", {})
        paths = cfg.get("paths", {})
        board = cfg.get("board", {})
        return cls(
            iterations=int(t.get("iterations", 50)),
            games_per_iter=int(t.get("games_per_iter", 20)),
            batch_size=int(t.get("batch_size", 128)),
            updates_per_iter=int(t.get("updates_per_iter", 50)),
            lr=float(t.get("lr", 0.002)),
            weight_decay=float(t.get("weight_decay", 0.0001)),
            c_policy=float(t.get("c_policy", 1.0)),
            lambda_vbar=float(t.get("lambda_vbar", 0.2)),
            replay_capacity=int(t.get("replay_capacity", 100_000)),
            eval_every=int(t.get("eval_every", 100)),
            eval_games=int(t.get("eval_games", 20)),
            sims_train=int(m.get("sims_train", 150)),
            sims_eval=max(20, int(m.get("sims_train", 150)) // 2),
            seed=int(cfg.get("seed", 0)),
            size=int(board.get("size", BOARD_SIZE)),
            channels=int(net.get("channels", 32)),
            blocks=int(net.get("blocks", 3)),
            c_puct=float(m.get("c_puct", 1.5)),
            root_noise=bool(m.get("root_noise", True)),
            dir_eps=float(m.get("dir_eps", 0.05)),
            temp_moves=int(m.get("temp_moves", 15)),
            temp=float(m.get("temp", 1.0)),
            opponents=dict(t.get("opponents", {"self": 0.6, "heuristic": 0.3, "random": 0.1})),
            heuristic_warmup_iters=int(t.get("heuristic_warmup_iters", 50)),
            opponent_sims=int(t.get("opponent_sims", 32)),
            search_margin=int(m.get("search_margin", -1)),
            opening_moves=int(t.get("opening_moves", 0)),
            opening_max=int(t.get("opening_max", 0)),
            pi_min_visits=int(t.get("pi_min_visits", 0)),
            eval_vs_heuristic_sims=int(t.get("eval_vs_heuristic_sims", 300)),
            models_dir=str(paths.get("models", "models")),
            logs_dir=str(paths.get("logs", "logs")),
            save_every_iters=int(t.get("save_every_iters", 5)),
            keep_checkpoints=int(t.get("keep_checkpoints", 5)),
        )


def _prune_checkpoints(ckpt_dir: Path, keep: int):
    """只保留最新的 keep 个轮次快照（文件名形如 ckpt_000010.pt）。"""
    if keep <= 0:
        return
    files = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _pick_opponent(opponents: Dict[str, float], rng: random.Random) -> str:
    names = list(opponents.keys())
    weights = [max(0.0, float(opponents[n])) for n in names]
    total = sum(weights)
    if total <= 0:
        return "self"
    return rng.choices(names, weights=[w / total for w in weights], k=1)[0]


def _opponent_for_iteration(tcfg: "TrainConfig", iteration: int, rng: random.Random) -> str:
    """训练对手策略：前 warmup_iters 轮用启发式陪练，之后纯自对弈。"""
    n = tcfg.heuristic_warmup_iters
    if n > 0 and iteration <= n:
        return "heuristic"
    return "self"


def _format_config(tcfg: "TrainConfig", device: str, start_iteration: int = 0) -> str:
    """把本次训练用到的全部关键参数打印成一块，便于事后复盘。"""
    lines = [
        "=" * 62,
        f"训练配置（起始迭代 {start_iteration}，以下参数均为本次实际生效值）",
        "-" * 62,
        f"设备        device={device}",
        f"棋盘        board.size={tcfg.size}",
        f"网络        channels={tcfg.channels}, blocks={tcfg.blocks}",
        f"搜索        sims_train={tcfg.sims_train}, sims_eval={tcfg.sims_eval}, "
        f"c_puct={tcfg.c_puct}",
        f"根噪声      root_noise={tcfg.root_noise}, dir_eps={tcfg.dir_eps}",
        f"落子范围    search_margin="
        + (
            f"{tcfg.search_margin}（外接矩形外扩，-1=全棋盘）"
            if tcfg.search_margin >= 0
            else "-1（全棋盘）"
        ),
        f"温度        temp_moves={tcfg.temp_moves}, temp={tcfg.temp}",
        f"对局        iterations={tcfg.iterations}, games_per_iter={tcfg.games_per_iter}, "
        f"先手比例=50%",
        f"对手        heuristic_warmup_iters={tcfg.heuristic_warmup_iters}, "
        f"opponent_sims={tcfg.opponent_sims}",
        f"评测        固定 vs 启发式 MCTS, eval_vs_heuristic_sims={tcfg.eval_vs_heuristic_sims}, "
        f"eval_every={tcfg.eval_every} 局, eval_games={tcfg.eval_games}",
        f"更新        batch_size={tcfg.batch_size}, updates_per_iter={tcfg.updates_per_iter}, "
        f"lr={tcfg.lr}, weight_decay={tcfg.weight_decay}",
        f"损失        c_policy={tcfg.c_policy}, lambda_vbar={tcfg.lambda_vbar}",
        f"回放池      capacity={tcfg.replay_capacity}",
        f"数据选项    opening_moves={tcfg.opening_moves}, opening_max={tcfg.opening_max}, "
        f"pi_min_visits={tcfg.pi_min_visits}",
        f"存档        save_every_iters={tcfg.save_every_iters}, "
        f"keep_checkpoints={tcfg.keep_checkpoints}, models={tcfg.models_dir}, logs={tcfg.logs_dir}",
        f"随机种子    seed={tcfg.seed}",
        "=" * 62,
    ]
    return "\n".join(lines)


def evaluate(
    net: Network,
    device: str,
    games: int,
    sims: int,
    seed: int,
    vs_heuristic_sims: int = 300,
    margin: int = -1,
) -> Dict[str, float]:
    """评测：固定使用启发式 MCTS（启发式评估器 + 搜索）作为对手。

    返回 {"heuristic": 得分率}（胜 1 / 平 0.5 / 负 0）。先后手交换由 arena 负责。
    注意：`margin` 是“可落子范围”设置，不是比赛规则——比赛仍按全棋盘合法落子，
    这里只让双方的**搜索**在该范围内分配算力，与训练保持一致。
    """
    agent = MCTSPlayer(
        NetEvaluator(net, device=device),
        simulations=sims,
        margin=margin,
        rng=random.Random(seed),
    )
    opp = MCTSPlayer(
        HeuristicEvaluator(),
        simulations=vs_heuristic_sims,
        margin=margin,
        rng=random.Random(seed + 1),
    )
    res = run_arena(agent, opp, games=games, size=net.size)
    return {"heuristic": res.score}


def train(tcfg: Optional[TrainConfig] = None, device: str = "cpu") -> Network:
    tcfg = tcfg or TrainConfig()
    torch = _torch()

    set_seed(tcfg.seed)
    rng = random.Random(tcfg.seed)
    Path(tcfg.models_dir).mkdir(parents=True, exist_ok=True)
    Path(tcfg.logs_dir).mkdir(parents=True, exist_ok=True)

    net = build_network(tcfg.channels, tcfg.blocks, size=tcfg.size, device=device)
    opt = torch.optim.Adam(
        net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    iteration = 0
    best_score = -1.0
    buffer = ReplayBuffer(tcfg.replay_capacity, seed=tcfg.seed)

    if tcfg.resume:
        net = Network.load(tcfg.resume, device=device, size=tcfg.size)
        meta = getattr(net, "meta", {}) or {}
        extra = meta.get("extra", {})
        iteration = int(extra.get("iteration", 0))
        best_score = float(extra.get("best_score", -1.0))
        if "opt" in extra:
            opt = torch.optim.Adam(
                net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
            )
            opt.load_state_dict(extra["opt"])
        print(f"续训：从迭代 {iteration} 继续，best_score={best_score:.3f}")

    sp_cfg = SelfPlayConfig(
        simulations=tcfg.sims_train,
        c_puct=tcfg.c_puct,
        root_noise=tcfg.root_noise,
        dir_eps=tcfg.dir_eps,
        temp_moves=tcfg.temp_moves,
        temp=tcfg.temp,
        size=tcfg.size,
        opponent_sims=tcfg.opponent_sims,
        search_margin=tcfg.search_margin,
        opening_moves=tcfg.opening_moves,
        opening_max=tcfg.opening_max,
        pi_min_visits=tcfg.pi_min_visits,
    )

    log_path = Path(tcfg.logs_dir) / "train_log.csv"
    write_header = not log_path.exists() or iteration == 0
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["iteration", "games", "buffer", "loss_policy", "loss_value",
             "eval_heuristic", "elapsed_s"]
        )

    total_games = 0
    t_start = time.time()
    print(_format_config(tcfg, device, iteration))

    try:
        for it in range(1, tcfg.iterations + 1):
            iteration += 1
            net.eval()
            opp_name = _opponent_for_iteration(tcfg, iteration, rng)
            for _ in range(tcfg.games_per_iter):
                # 每局 50% 先手 / 50% 后手
                agent_first = rng.random() < 0.5
                records, winner, steps = play_selfplay(
                    net,
                    device=device,
                    cfg=sp_cfg,
                    opponent=opp_name,
                    seed=rng.randrange(2**31),
                    agent_first=agent_first,
                )
                buffer.add_game(records, winner)
                total_games += 1

            # ---------- 网络更新 ----------
            loss_p = loss_v = None
            updated = 0
            min_samples = min(tcfg.batch_size, 8)
            if len(buffer) >= min_samples:
                net.train()
                for _ in range(tcfg.updates_per_iter):
                    batch = buffer.sample(
                        tcfg.batch_size, augment=True, lambda_vbar=tcfg.lambda_vbar
                    )
                    x = torch.from_numpy(batch["states"]).to(device)
                    logits, v = net(x)

                    legal = torch.from_numpy(
                        (batch["states"][:, 2] > 0.5)
                        .reshape(batch["states"].shape[0], -1)
                        .astype(np.float32)
                    ).to(device)
                    masked = logits.masked_fill(legal <= 0, -1e9)
                    log_probs = torch.log_softmax(masked, dim=1)
                    pi_t = torch.from_numpy(batch["pis"]).to(device)
                    loss_policy = -(pi_t * log_probs).sum(dim=1).mean()

                    target = torch.from_numpy(batch["targets"]).to(device)
                    loss_value = torch.nn.functional.mse_loss(v, target)

                    loss = loss_value + tcfg.c_policy * loss_policy
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(net.parameters()), 5.0)
                    opt.step()

                    loss_p = float(loss_policy.item())
                    loss_v = float(loss_value.item())
                    updated += 1
            else:
                print(
                    f"  （回放池 {len(buffer)} 条 < batch_size {tcfg.batch_size}，"
                    f"本迭代跳过梯度更新）"
                )

            elapsed = time.time() - t_start
            eval_h = None
            loss_desc = (
                f"loss_p={loss_p:.4f} loss_v={loss_v:.4f}"
                if loss_p is not None
                else "loss=n/a（样本不足）"
            )
            if total_games % max(1, tcfg.eval_every) < tcfg.games_per_iter:
                net.eval()
                scores = evaluate(
                    net,
                    device,
                    tcfg.eval_games,
                    tcfg.sims_eval,
                    seed=tcfg.seed + iteration,
                    vs_heuristic_sims=tcfg.eval_vs_heuristic_sims,
                )
                eval_h = scores["heuristic"]
                if eval_h > best_score:
                    best_score = eval_h
                    net.save(
                        Path(tcfg.models_dir) / "best.pt",
                        extra={"iteration": iteration, "best_score": best_score},
                    )
                print(
                    f"[迭代 {iteration}] opp={opp_name} buffer={len(buffer)} {loss_desc} "
                    f"vs heuristic={eval_h:.1%} "
                    f"用时 {elapsed / 60:.1f}m"
                )
            else:
                print(
                    f"[迭代 {iteration}] opp={opp_name} buffer={len(buffer)} {loss_desc} "
                    f"用时 {elapsed / 60:.1f}m"
                )

            net.eval()
            net.save(
                Path(tcfg.models_dir) / "latest.pt",
                extra={
                    "iteration": iteration,
                    "best_score": best_score,
                    "opt": opt.state_dict(),
                },
            )
            # ---------- 定期保存带轮次的快照 ----------
            if tcfg.save_every_iters > 0 and iteration % tcfg.save_every_iters == 0:
                ckpt_path = Path(tcfg.models_dir) / f"ckpt_{iteration:06d}.pt"
                net.save(
                    ckpt_path,
                    extra={
                        "iteration": iteration,
                        "best_score": best_score,
                        "opt": opt.state_dict(),
                    },
                )
                _prune_checkpoints(Path(tcfg.models_dir), tcfg.keep_checkpoints)
                print(f"  已保存检查点：{ckpt_path}")
            log_writer.writerow(
                [
                    iteration,
                    total_games,
                    len(buffer),
                    "" if loss_p is None else f"{loss_p:.6f}",
                    "" if loss_v is None else f"{loss_v:.6f}",
                    "" if eval_h is None else f"{eval_h:.4f}",
                    f"{elapsed:.1f}",
                ]
            )
            log_file.flush()
    finally:
        log_file.close()

    net.save(Path(tcfg.models_dir) / "final.pt", extra={"iteration": iteration})
    print(f"训练完成，模型保存在 {tcfg.models_dir}/final.pt")
    return net
