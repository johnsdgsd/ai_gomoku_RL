"""策略/价值网络（AlphaZero 式）。尺寸由 size 参数决定（9/13/15...）。

结构：
  Conv(4→CH, 3x3, pad=1) + ReLU
  ResidualBlock(CH) x blocks
  策略头: Conv(CH→2, 1x1) → Flatten(2*N) → Linear(2N→N)          → logits
  价值头: Conv(CH→1, 1x1) → Flatten(N)   → Linear(N→128) → ReLU → Linear(128→1) → tanh
  其中 N = size*size

说明：
  - 不使用 BatchNorm（浅网络、小 batch、推理确定性）。
  - 策略头输出 logits，softmax 只在 MCTS 中针对合法动作做。
  - 价值头 tanh，输出 [-1, 1]，与终局 z ∈ {-1, 0, 1} 对齐。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import BOARD_SIZE, WIN_LEN, side_of

FORMAT_VERSION = 1


def _torch():
    import torch

    return torch


def _nn():
    from torch import nn

    return nn


def build_model(channels: int = 32, blocks: int = 3, size: int = BOARD_SIZE):
    size = int(size)
    n = size * size
    torch = _torch()
    nn = _nn()

    class ResidualBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            return self.relu(out + x)

    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(4, channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                *[ResidualBlock(channels) for _ in range(blocks)]
            )
            self.policy_conv = nn.Conv2d(channels, 2, 1)
            self.policy_fc = nn.Linear(2 * n, n)
            self.value_conv = nn.Conv2d(channels, 1, 1)
            self.value_fc1 = nn.Linear(n, 128)
            self.value_fc2 = nn.Linear(128, 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and m.bias is None:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            h = self.blocks(self.stem(x))
            p = self.policy_conv(h).reshape(x.shape[0], -1)
            logits = self.policy_fc(p)
            v = self.value_conv(h).reshape(x.shape[0], -1)
            v = torch.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v)).squeeze(1)
            return logits, v

    return PolicyValueNet()


class Network:
    """网络封装：设备、尺寸、保存/加载、前向。"""

    def __init__(
        self,
        channels: int = 32,
        blocks: int = 3,
        size: int = BOARD_SIZE,
        device: str = "cpu",
    ):
        self.device = device
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.size = int(size)
        self.model = build_model(self.channels, self.blocks, self.size).to(device)

    def __call__(self, x):
        return self.model(x)

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        try:
            self.model.load_state_dict(sd)
        except RuntimeError as exc:
            raise ValueError(
                f"模型与当前网络尺寸不匹配（模型 {self.size}x{self.size}，"
                f"{self.channels} 通道 / {self.blocks} 块）：{exc}"
            ) from exc

    def forward_numpy(self, planes: np.ndarray):
        import torch

        single = planes.ndim == 3
        if single:
            planes = planes[None, ...]
        x = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).to(
            self.device
        )
        with torch.no_grad():
            logits, value = self.model(x)
        logits = logits.float().cpu().numpy()
        value = value.float().cpu().numpy()
        if single:
            return logits[0], float(value[0])
        return logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ---------- 存档 ----------
    def save(self, path, extra: dict | None = None):
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT_VERSION,
            "board_size": self.size,
            "win_len": WIN_LEN,
            "channels": self.channels,
            "blocks": self.blocks,
            "model": self.model.state_dict(),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, device: str = "cpu", size: int | None = None):
        import torch

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到模型文件: {path}\n"
                "提示：先训练（python -m gomoku train），或用 --algo heuristic 走启发式对局。"
            )
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError(f"模型文件格式不正确: {path}")
        fmt = ckpt.get("format", 0)
        if fmt != FORMAT_VERSION:
            raise ValueError(f"模型格式版本不支持: {fmt}（当前 {FORMAT_VERSION}）")
        file_size = int(ckpt.get("board_size", BOARD_SIZE))
        if size is not None and int(size) != file_size:
            raise ValueError(
                f"模型是在 {file_size}x{file_size} 棋盘上训练的，"
                f"当前请求 {size}x{size}；请用 --board-size {file_size} 或改用匹配尺寸的模型。"
            )
        net = cls(
            ckpt.get("channels", 32),
            ckpt.get("blocks", 3),
            size=file_size,
            device=device,
        )
        net.load_state_dict(ckpt["model"])
        net.eval()
        net.meta = ckpt
        return net


def load_network(path, device: str = "cpu", size: int | None = None) -> Network:
    return Network.load(path, device=device, size=size)


def build_network(
    channels: int = 32, blocks: int = 3, size: int = BOARD_SIZE, device: str = "cpu"
) -> Network:
    return Network(channels, blocks, size=size, device=device)
