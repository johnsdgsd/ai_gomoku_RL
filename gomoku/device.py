"""配置加载与设备解析。

优先级：CLI 参数 > GOMOKU_DEVICE 环境变量 > --config 文件 > 内置默认。
device 取值：auto | cpu | cuda | cuda:1 | mps。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "device": "auto",
    "seed": 0,
    "num_threads": 8,
    "board": {"size": 15},
    "paths": {"models": "models", "logs": "logs"},
    "net": {"channels": 32, "blocks": 3},
    "mcts": {
        "sims_play": 400,
        "sims_train": 150,
        "c_puct": 1.5,
        "root_noise": True,
        "dir_alpha": 0.3,
        "dir_eps": 0.25,
        "temp_moves": 15,
        "temp": 1.0,
    },
    "train": {
        "iterations": 50,
        "games_per_iter": 20,
        "batch_size": 128,
        "updates_per_iter": 50,
        "lr": 0.002,
        "weight_decay": 0.0001,
        "c_policy": 1.0,
        "lambda_vbar": 0.2,
        "replay_capacity": 200000,
        "eval_every": 100,
        "eval_games": 20,
        "save_every_iters": 5,
        "keep_checkpoints": 5,
        "opponents": {"self": 0.6, "heuristic": 0.3, "random": 0.1},
    },
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """读取 YAML 配置并与内置默认合并。缺失文件时回退到 configs/default.yaml。"""
    cfg: Dict[str, Any] = copy.deepcopy(DEFAULTS)
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(DEFAULT_CONFIG_PATH)

    for p in candidates:
        if p.is_file():
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("读取配置文件需要 PyYAML：pip install pyyaml") from exc
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, data)
            break
    return cfg


def _torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:
        return None


def device_spec(config_value: Optional[str] = None, cli_value: Optional[str] = None) -> str:
    """按优先级合并设备配置：CLI > GOMOKU_DEVICE 环境变量 > 配置文件 > auto。"""
    return (
        (cli_value or "").strip()
        or (os.environ.get("GOMOKU_DEVICE") or "").strip()
        or (config_value or "").strip()
        or "auto"
    )


def resolve_device(spec: Optional[str] = None) -> str:
    """解析设备字符串，返回 torch 设备字符串。

    没有安装 torch 时返回 'cpu'（无模型路径仍然可用）。
    """
    torch = _torch()
    spec = device_spec(spec).lower()

    if torch is None:
        return "cpu"

    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if spec in ("cpu", "mps"):
        return spec

    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定的 device=%s 不可用：当前 torch 是 CPU 构建（torch=%s, cuda_available=False）。\n"
                "两种解决方式：\n"
                "  1) 用 CPU：--device cpu，或配置文件写 device: auto\n"
                "  2) 安装 CUDA 版 torch：\n"
                "     pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "     （安装后用 python -m gomoku doctor 确认 cuda 可用）"
                % (spec, getattr(torch, "__version__", "unknown"))
            )
        return spec

    raise ValueError(f"无法识别的 device 配置: {spec!r}（可选 auto/cpu/cuda/cuda:N/mps）")


def configure_threads(cfg: Dict[str, Any]):
    """设置 torch 线程数（尽力而为）。"""
    torch = _torch()
    if torch is None:
        return
    n = int(cfg.get("num_threads") or 0)
    if n > 0:
        torch.set_num_threads(n)


def doctor_report(cfg: Dict[str, Any], config_source: str = "") -> str:
    import platform
    import sys

    lines = ["=== 环境自检 ==="]
    lines.append(f"python      : {sys.version.split()[0]} ({platform.machine()})")
    lines.append(f"工作目录    : {Path.cwd()}")
    torch = _torch()
    if torch is None:
        lines.append("torch       : 未安装（无模型路径不可用，启发式/MCTS 仍可运行）")
        lines.append("torch threads / cuda : n/a")
    else:
        lines.append(f"torch       : {torch.__version__}")
        lines.append(f"cuda 可用    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda 设备数  : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"  [{i}] {torch.cuda.get_device_name(i)}")
        lines.append(f"cuda 构建    : {torch.version.cuda}")
        lines.append(f"torch 线程数 : {torch.get_num_threads()}")
    spec = device_spec(cfg.get("device", "auto"))
    try:
        resolved = resolve_device(spec)
        reason = "配置文件/环境变量指定" if spec != "auto" else "自动选择"
        lines.append(f"device 配置  : {spec} -> {resolved}（{reason}）")
    except RuntimeError as exc:
        lines.append(f"device 配置  : {spec} -> 不可用")
        lines.append(str(exc))
    lines.append(f"配置文件     : {config_source or '内置默认'}")
    board = cfg.get("board", {})
    lines.append(f"棋盘         : {board.get('size')}x{board.get('size')}（board.size）")
    lines.append(f"seed         : {cfg.get('seed')}")
    return "\n".join(lines)
