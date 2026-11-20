# 五子棋 AlphaZero 式 AI（尺寸可配：9×9 / 15×15，无禁手）

从"DQN + MCTS"重构而来：修正了原实现中 MCTS 价值符号取反、终局节点继续扩展、防守数据不可解等问题，
改为 **策略/价值双头网络 + PUCT MCTS + 自对弈训练**，并保留一个零额外依赖的棋型启发式搜索作为开箱即用的对手。

棋盘尺寸由 `configs/default.yaml` 的 `board.size` 决定（默认 15），网络、编码、回放池、搜索全部按尺寸自适应；
15×15 只需改一行配置（或 `--board-size 15`）。

## 环境

| 项 | 当前实测值 |
|---|---|
| Python | 3.11.11（`MyRL` conda 环境） |
| torch | 2.8.0+cpu（CUDA 不可用） |
| numpy / PyYAML | 2.3.4 / 6.0.2 |
| 逻辑核 | 28 |

```powershell
# 下文命令都在仓库根目录执行
$py = "D:\WYJ\Anaconda\Anaconda3\envs\MyRL\python.exe"
```

依赖：`numpy`、`PyYAML`（核心与启发式）；`torch`（网络与训练）。测试用标准库 `unittest`，**不需要 pytest**。

## 快速开始

```powershell
# 0) 环境自检：torch 版本、CUDA、设备解析、线程数
& $py -m gomoku doctor

# 1) 人机对局（无需模型，走棋型启发式 + MCTS）
& $py -m gomoku play --algo heuristic --sims 400
#   启动时会问"先手（1）还是后手（2）"，默认先手；
#   也可以直接指定：--human p1（先手） / --human p2（后手，AI 先走）
#   对局中支持 undo 悔棋、hint 提示、board 重绘、quit 退出

# 2) 对拉评测
& $py -m gomoku arena --a heuristic --b random --games 100 --sims 60

# 3) 训练（小规模冒烟）
& $py -m gomoku train --iterations 2 --games-per-iter 5 --sims-train 20 --device cpu

# 4) 续训（从上次的存档继续；--iterations 是"再跑多少轮"，不是总数）
& $py -m gomoku train --resume models/latest.pt --iterations 100 --games-per-iter 20 --sims-train 200

# 5) 用训练好的模型对局
& $py -m gomoku play --algo net --model models/best.pt --human p2 --device cpu

# 6) 浏览器界面（下棋 + 可视化搜索过程）
& $py -m gomoku web --model models/latest.pt --sims 1000
#   然后打开 http://127.0.0.1:8000
```

> 通过管道/脚本喂输入时不会弹交互提问（`stdin` 不是终端），此时请显式加 `--human p1|p2`。

## 浏览器界面

```powershell
& $py -m gomoku web --model models/latest.pt --sims 1000
#   → http://127.0.0.1:8000
```

**完全独立于训练流程**：只读取指定模型文件，不写任何训练产物，也不影响正在跑的训练。
不给 `--model` 时会自动找 `models/latest.pt` / `models/best.pt`；加 `--algo heuristic` 则不用模型。

| 参数 | 说明 |
|---|---|
| `--model` | 模型路径 |
| `--algo` | `net` / `heuristic`，省略时按有没有模型自动判断 |
| `--sims` | AI 每步模拟次数，默认取配置 `mcts.sims_play` |
| `--human` | `p1`（你执黑）/ `p2`（你执白）/ `none`（AI 互下） |
| `--port` / `--host` | 监听地址，默认 `127.0.0.1:8000` |

界面上能看到的不只是棋盘：

| 区域 | 内容 |
|---|---|
| 棋盘 | 点击落子；**蓝色热力圈** = MCTS 在该点的访问占比，越亮越重视 |
| 局面评估 | 网络对该局的判断（黑方视角，−1 ~ +1）+ **整盘价值曲线** |
| MCTS 候选 | 前 8 个候选点的访问次数、占比、q 值 |
| 按钮 | 新局 / 悔棋 / 提示（只算不落子）/ AI 走一步 / 导出棋谱 |

「导出棋谱」会给出每手坐标、黑白标记、该手前的评估值，以及一行可以直接复制的坐标串——
排查「它哪一手开始下崩」时比看终端方便得多。

> 每次 AI 落子是同步计算的（1000 次模拟约 1~2 秒），页面会显示「AI 思考中…」。
> AI 互下模式不是自动播放，而是每次点「AI 走一步」前进一手，方便逐手观察热力图。

## 续训

训练每轮结束都会写 `models/latest.pt`（含网络、优化器状态、迭代号），中断后可以直接续：

```powershell
# 从 latest.pt 继续，再跑 100 轮
& $py -m gomoku train --resume models/latest.pt --iterations 100 --games-per-iter 20 --sims-train 200

# 服务器后台续训（关掉终端也不停）
nohup python -m gomoku train --resume models/latest.pt --iterations 100 \
  --games-per-iter 20 --sims-train 200 > logs/resume.log 2>&1 &
```

要点：

| 项 | 说明 |
|---|---|
| `--resume` | 路径写 `models/latest.pt`（也接受任意 `ckpt_0000NN.pt`） |
| `--iterations` | **本次还要跑多少轮**；起始迭代号会从存档里接着累加 |
| 优化器状态 | 一起恢复，学习率/动量不会重置 |
| 回放池 | **默认自动恢复** `models/buffer.npz`；加 `--fresh-buffer` 可只恢复网络、清空经验 |
| 尺寸校验 | 存档里记录了 `board_size`，与配置不一致会报错并提示改用 `--board-size` |
| 快照 | 每 `save_every_iters`（默认 5）轮额外写 `models/ckpt_0000NN.pt`，可指定任一快照续训 |

### 回放池持久化

每 `save_every_iters` 轮会连回放池一起落盘，续训时自动读回来接着用：

```yaml
train:
  replay_capacity: 30000   # 池子上限：约 1500 局 / 75 轮，保证经验不过期
  save_buffer: true        # 定期写 models/buffer.npz（原子写入，中途打断不会损坏）
  resume_buffer: true      # 续训时自动恢复
```

```powershell
# 正常续训：网络 + 优化器 + 回放池 全部恢复
& $py -m gomoku train --resume models/latest.pt --iterations 100

# 想换一批经验重来：只恢复网络
& $py -m gomoku train --resume models/latest.pt --iterations 100 --fresh-buffer
```

`buffer.npz` 里局面是 0/1 平面，压缩后 3 万条约 2–20 MB（局面越满压缩率越低）。删除该文件等同于 `--fresh-buffer`。

续训时终端会先打印完整配置，第一行显示 `起始迭代 N`，可以确认接得上：

```
==============================================================
训练配置（起始迭代 50，以下参数均为本次实际生效值）
--------------------------------------------------------------
设备        device=cuda
...
```


## 棋盘尺寸配置

尺寸只有一个入口：`configs/default.yaml` 的 `board.size`，当前为 **15**。

```yaml
# configs/default.yaml
board:
  size: 15              # 改成 9 / 13 / 19 即可切换，其余参数不用动
```

```powershell
# 按配置文件运行（默认就是 15x15）
& $py -m gomoku play --algo heuristic --sims 300

# 临时覆盖（优先级高于配置文件，不改文件也能试）
& $py -m gomoku play --board-size 9
& $py -m gomoku arena --a heuristic --b random --games 20 --sims 40
& $py -m gomoku train --iterations 5 --games-per-iter 5 --sims-train 60
```

- 模型存档会记录训练时的尺寸；用不匹配的尺寸加载会给出明确错误（提示改用 `--board-size`）。
- 15×15 的候选点更多，CPU 上单步耗时约为 9×9 的 1.5–2 倍：
  实测启发式评估 **0.81 ms/次模拟**（9×9 为 0.49），300 次模拟 ≈ 0.24 s/步。
- 连珠长度固定为 **5 连获胜**；15×15 为标准"无禁手自由五子棋"（黑棋长连/双四/双三不判负）。

## 设备配置

优先级：**CLI 参数 > 环境变量 `GOMOKU_DEVICE` > 配置文件 > 内置默认**。

```yaml
# configs/default.yaml
device: auto          # auto | cpu | cuda | cuda:1 | mps
num_threads: 8
```

- `auto`：按 `cuda → mps → cpu` 顺序选择，并打印实际结果。
- 强制 `cuda` 但当前是 CPU 版 torch 时，报错并给出安装 CUDA 轮子的命令。
- `python -m gomoku doctor` 可随时确认解析结果。

## 设计要点

### 1. 状态编码（4 通道）

```
平面 0  己方棋子（to_move）
平面 1  对方棋子
平面 2  空位
平面 3  上一手落点
```

视角永远是"轮到该走的一方"，**不传黑/白身份**；`last_move` 让网络聚焦对手刚形成的威胁。
8 种旋转/镜像对称变换在棋盘、动作、策略分布、上一手上共用同一张置换表（用于数据增广与测试）。

### 2. 网络（9×9 约 6 万参数，CPU 单次前向 < 1ms；尺寸自适应）

```
Conv(4→32, 3×3) + ReLU
ResidualBlock(32) × 3           # Conv→ReLU→Conv→(+x)→ReLU
策略头: Conv(32→2, 1×1) → Flatten(2N) → Linear(2N→N)      → logits
价值头: Conv(32→1, 1×1) → Flatten(N)  → Linear(N→128) → ReLU → Linear(128→1) → tanh
其中 N = board.size²（9×9 为 81，15×15 为 225）
```

- 策略头输出 **logits**；softmax 只在 MCTS 中针对合法动作做。
- 价值头 `tanh`，与终局信号 `z ∈ {-1,0,1}` 对齐。
- 不使用 BatchNorm：浅网络 + 小 batch + 推理确定性。

### 3. MCTS（PUCT）

```
选择: a = argmax[ q(a) + c·P[a]·√N_total/(1+N[a]) ]；N_total==0 时按 P 选
估值: 终局节点 → 规则值(-1/0)，不调网络
      未展开叶子 → 一次前向得 (P,V)，登记该节点全部合法动作的"空壳"子节点
      已展开非终局 → 复用缓存 V
回溯: v = -v; parent.W[a] += v; parent.N[a] += 1; parent.N_total += 1
收口: a* = argmax N[a] + 硬规则保险（能连五必上、唯一必堵点必堵）
      π = N/N_total，V̄ = ΣW/ΣN
```

不变量：`W[a]` 恒为父节点视角、`N_total == ΣN[a]`、终局节点零前向、每节点最多一次评估、子节点惰性创建。

### 4. 训练闭环

```
真实对局（每半回合都记）: (s, a, π, V̄, to_move)
一局结束回填:            z = +1 / -1 / 0（按该条记录的 to_move）
采样 batch:              logits, V = net(s)
策略损失:                CE(masked_logits, π)          # π 是软分布
价值损失:                MSE(V, (1-λ)z + λV̄)，λ=0.2
反向传播:                clip_grad_norm_(5.0) + Adam
```

**MCTS 内部不产生梯度**，只输出决策；回放池只存真实路径，不存仿真数据。

## 目录

```
gomoku/
  env.py          棋盘规则（纯函数落子、终局、赢家、可视化）
  encoder.py      4 通道编码、8 对称变换、masked softmax
  evaluators.py   棋型评估器（启发式）+ 网络评估器
  mcts.py         PUCT MCTS
  players.py      random / greedy / MCTS 玩家
  arena.py        对拉评测
  replay.py       回放池（uint8 压缩 + 对称增广 + npz 存取）
  selfplay.py     自对弈生成数据
  network.py      策略/价值网络与存档
  train.py        训练主循环（评测、best 模型、续训、CSV 日志）
  device.py       配置加载 + 设备解析 + doctor
  web.py          浏览器界面（HTTP 服务 + 单页前端 + 棋谱导出）
  cli.py          play / arena / web / train / doctor
configs/default.yaml     # 唯一配置：board.size、device、mcts、train 等全部参数
tests/             unittest（零额外依赖）
```

## 测试

```powershell
& $py -m unittest discover -s tests -t .
```

覆盖：规则与边界（四方向五连、跨行不回绕、平局、悔棋）、编码与对称往返、
MCTS 必胜必堵与回溯符号、网络形状/梯度/存读档、回放池 z 回填与增广对齐、自对弈与训练冒烟、
回放池 npz 存取与续训恢复、Web 接口（新局/落子/悔棋/提示/导出/404），
以及 **9×9 / 15×15 双尺寸**（编码形状、对称置换、网络输出、MCTS 战术、回放池、arena/自对弈、配置读取）。

## 已知限制

- 支持任意 N×N 棋盘（`board.size`），当前只实现 **无禁手自由五子棋**（15×15 黑棋长连/双四/双三不判负）；
  连珠禁手、15×15 以外的开局规则未实现。
- **连珠长度固定为 5**（`env.WIN_LEN`），棋型评估器按 5 连设计；改这个值需要同步扩展棋型表。
- CPU 训练较慢：单局自对弈（150 次模拟/步）约数秒级，完整训练需数小时；15×15 更慢，
  建议降低 `sims_train`/`sims_play` 再逐步加码，或先看 `arena` 结果。
- 没有图形/网页界面，终端交互。
- 默认配置只保证闭环可跑，不承诺短期超过启发式基线。

## 相对旧版本修掉的问题

1. `mcts_dqn.py` 中评估值用"刚落子方视角"却存入"对手视角"节点，父节点读取时再取负 → 胜招被记成 -1；
2. MCTS 无终局标记，已连五节点继续扩展，胜线被稀释；
3. 叶子价值用 `tanh(max Q)`，与塑形奖励尺度不可比；
4. `train_defense.py` 构造的残局数学上无解（连三/连四两端全空），奖励恒为 -1；
5. 对手只会挡立即五连，学不到活三/双威胁防守；
6. 三个脚本硬编码不存在的 `dqn_checkpoint_12000.pt`；
7. 棋型评分用 `.replace` 把 5 字符棋型替换成 6 个字符，长度漂移导致重复计数。
