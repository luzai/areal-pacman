# AReaL PacMan RL 配方

本仓库是 MaaPacman Level 1 的 AReaL 强化学习配方层，负责数据集、训练
workflow、奖励塑形、训练配置、轨迹审计和评估工具。

真实游戏环境由独立的 `maapacman` Python 包提供；分布式训练、vLLM rollout、
FSDP actor/reference engine 和 checkpoint 管理由官方 AReaL 提供。

## 架构文档

- [AReaL Pacman RL 配方设计](docs/architecture/AREAL_RECIPE_DESIGN.md)
- [MaaPacman 环境接口设计](../MaaPacman/PACMAN_ENV_DESIGN.md)
- [运行产物说明](RUN_ARTIFACTS.md)

## 系统边界

```text
MaaPacman 仓库 / maapacman 包
  maapacman.env.PygamePacmanEnv
    拥有真实 pygame Level 1 状态转移、RGB 渲染、合法动作、
    奖励、终止条件和游戏指标。

areal-pacman 仓库（本仓库）
  areal_pacman/level1/workflow.py
    将截图和真实游戏状态连接到 AReaL rollout。
  areal_pacman/level1/level1_dataset.py
    生成由 PygamePacmanEnv 支持的确定性训练/验证 episode。
  areal_pacman/level1/rewards.py
    定义配方侧的奖励塑形与审计。
  configs/level1/
    生产训练、评估配置以及历史配置。
  scripts/level1/
    数据集、训练、评估和报告工具。

官方 AReaL checkout
  提供 trainer、rollout workers、vLLM、FSDP actor/reference engines、
  调度和 checkpoint 发布。
```

这些边界是有意设计的：

- MaaPacman 拥有游戏环境；
- 本仓库拥有 RL 配方；
- 官方 AReaL 拥有分布式训练系统。

`areal_pacman.synthetic.*` 仅用于历史文本/合成迷宫实验，不能替代真实
Level 1 生产环境。

## Sibling 仓库与环境安装

生产 Level 1 不能只安装本仓库，还需要三个独立 checkout：

```text
<workspace>/
  AReaL/          # 官方分布式训练框架，需要作为 Python 包安装
  MaaPacman/      # 提供 maapacman.env.PygamePacmanEnv，需要安装
  pacman-python/  # 原版 pygame 游戏源码，只需 checkout，不作为 pip 包安装
  areal-pacman/   # 本仓库，需要安装
```

推荐将 `MaaPacman`、`pacman-python` 和 `areal-pacman` 放在同一个父目录。
`pacman-python` 必须保持为独立、固定 revision、未修改的 checkout。

### 新 Linux/H100 服务器（从零安装）

node1 当前采用的目录布局是：

```text
/mnt/data/z00819216/xinglu/AReaL
/mnt/data/z00819216/maapacman-stack/MaaPacman
/mnt/data/z00819216/maapacman-stack/pacman-python
/mnt/data/z00819216/maapacman-stack/<areal-pacman-checkout>
```

node1 已验证的关键环境版本：

```text
NVIDIA driver==590.48.01
CUDA used by PyTorch==13.0
python==3.12.13
areal==1.0.4
torch==2.11.0+cu130
torchvision==0.26.0
transformers==5.7.0
vllm==0.22.1
datasets==5.0.0
accelerate==1.14.0
peft==0.18.1
tokenizers==0.22.2
safetensors==0.8.0
numpy==2.2.6
pillow==12.2.0
pygame==2.6.1
flashinfer-python==0.6.11.post2
torch-memory-saver==0.0.9
```

从零创建环境并安装四层依赖：

```bash
# 1. 创建并激活 Conda 环境
conda create -n maapacman-rl python=3.12.13 pip -y
conda activate maapacman-rl

# 2. 安装固定版本的 GPU/Python 依赖
# 使用上方记录的 node1 已验证版本。
# PyTorch wheel 必须兼容服务器 CUDA；node1 使用 torch 2.11.0+cu130。

# 3. 安装三个 Python 项目
python -m pip install -e "/path/to/AReaL"
python -m pip install -e "/path/to/MaaPacman[pygame]"
python -m pip install -e "/path/to/areal-pacman[dev,dataset,agent]"

# 4. pacman-python 不需要 pip 安装，只需指向固定 revision 的 checkout
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
```

在 node1 上，`/path/to/...` 分别对应：

```bash
export AREAL_ROOT=/mnt/data/z00819216/xinglu/AReaL
export MAAPACMAN_ROOT=/mnt/data/z00819216/maapacman-stack/MaaPacman
export AREAL_PACMAN_ROOT=/mnt/data/z00819216/maapacman-stack/<areal-pacman-checkout>
export MAAPACMAN_PACMAN_PYTHON_ROOT=/mnt/data/z00819216/maapacman-stack/pacman-python
```

确认四层依赖均解析正确：

```bash
python - <<'PY'
from pathlib import Path

import areal
import areal_pacman
from maapacman.env import PygamePacmanEnv

print("areal:", Path(areal.__file__).resolve())
print("areal_pacman:", Path(areal_pacman.__file__).resolve())

with PygamePacmanEnv() as env:
    image, info = env.reset(seed=0)
    print("env_id:", info["env_id"])
    print("backend:", info["backend"])
    print("image:", image.shape)
    print("renderer:", env.spec.renderer_revision)
PY
```

预期环境 ID 和 backend：

```text
env_id: pacman-python-level1-pygame-v1
backend: original-pygame
```

## 准备 Level 1 数据集

短 episode（32 步）：

```bash
python scripts/level1/dataset/prepare_level1_dataset.py \
  --output-root artifacts/datasets/level1_dataset_step32 \
  --train-episodes 8 \
  --validation-episodes 2 \
  --max-steps 32 \
  --write-hf
```

长 episode（256 步）：

```bash
python scripts/level1/dataset/prepare_level1_dataset.py \
  --output-root artifacts/datasets/level1_dataset_step256 \
  --train-episodes 8 \
  --validation-episodes 2 \
  --max-steps 256 \
  --write-hf
```

`max_steps` 会写入每一条 episode 数据，因此不能把旧的 step32 目录简单改名为
step256。

## Linux/H100 训练入口

生产训练要求：

- 官方 AReaL checkout；
- MaaPacman 和 pacman-python；
- 本地 Qwen 模型 checkpoint；
- 与配置一致的数据集；
- 配置要求数量的空闲 GPU。

标准启动方式：

```bash
export AREAL_ROOT=/home/ubuntu/z00819216/xinglu/AReaL
export MODEL_PATH=/mnt/data/z00819216/models/Qwen3.5-9B
export CONFIG="$PWD/configs/level1/train/level1_live_state_step256_100update_group12_8gpu.yaml"
export DATASET_OUTPUT_ROOT="$PWD/artifacts/datasets/level1_dataset_step256"
export DATASET_MAX_STEPS=256

bash scripts/level1/train/run_level1_training.sh
```

启动器会依次：

1. 验证 Python、模型和官方 AReaL checkout；
2. 验证所选 GPU 没有 compute process；
3. 生成训练/验证数据集；
4. 执行 config dry-run；
5. 保存 config 和数据 manifest；
6. 启动正式训练；
7. 将 checkpoint、trajectory 和日志写入 `ARTIFACT_ROOT`。

建议为每次正式运行显式设置独立目录：

```bash
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID="reward-v2-step256-${RUN_TS}"
export ARTIFACT_ROOT="/mnt/data/z00819216/run_artifacts/maapacman-rl/${RUN_ID}"
```

## 256-step / 100-update 配置

配置文件：

[configs/level1/train/level1_live_state_step256_100update_group12_8gpu.yaml](configs/level1/train/level1_live_state_step256_100update_group12_8gpu.yaml)

主要参数：

```text
模型：Qwen/Qwen3.5-9B
GPU：8
训练 episode：8
验证 episode：2
batch size：4
epochs：50
optimizer updates：100
episode max_steps：256
n_samples：12
temperature：0.7
top_p：1.0
thinking：false
open_action_mask：true
```

## Reward v2

奖励公式：

```text
R_t = base_reward
      - 0.05
      - 1.0 * wall
      - 0.2 * revisit
      + distance_reward
```

其中：

```text
base_reward = current_game_score - previous_game_score
revisit = next_position in recent_positions[-8:]
```

普通豆子的原始游戏分数是 `+10`。未吃东西的普通移动和撞墙的
`base_reward` 为 `0`；能量豆和幽灵保留原版游戏分数变化。

当普通豆剩余比例不超过 `25%`，并且当前步骤没有吃普通豆时：

```text
distance_reward =
    0.2 * (nearest_distance_before - nearest_distance_after)
```

否则：

```text
distance_reward = 0
```

跳过吃豆步骤的 distance shaping，是为了避免吃掉当前目标后，下一颗最近豆子
突然变远而错误处罚吃豆行为。

示例：

| 行为 | 最终 reward |
| --- | ---: |
| 吃普通豆 | `+9.95` |
| 普通移动 | `-0.05` |
| 回到最近 8 步访问过的格子 | `-0.25` |
| 后期向最近豆子靠近一格 | `+0.15` |
| 后期远离最近豆子一格 | `-0.25` |
| 撞墙并触发 revisit | `-1.25` |

## 训练配置

生产配置位于 `configs/level1/train/`：

- `level1_live_state_step32_16update_group12_8gpu.yaml`：已完成的短训练。
- `level1_live_state_step32_200update_group12_8gpu.yaml`：32-step 长训练。
- `level1_live_state_step256_100update_group12_8gpu.yaml`：256-step、
  50-epoch、Reward v2 配置。
- `level1_logprob_alignment_probe_8gpu.yaml`：log-prob 对齐探针。

历史实验保存在 `configs/level1/archive/`、`configs/archive/text/` 和
`configs/archive/vision/`。这些配置仅用于复现旧实验，不应作为新的生产默认值。

## 最小验证

```bash
python -m pytest -q tests/test_level1_recipe.py

python train_areal.py \
  --config configs/level1/train/level1_live_state_step256_100update_group12_8gpu.yaml \
  --dry-run \
  --validate-areal
```

正式训练前还必须验证真实 `PygamePacmanEnv`、vLLM vision 请求、reference
log-prob、optimizer update 和 checkpoint save。

## 安全约束

- 启动训练前确认配置要求的全部 GPU 空闲。
- 不要停止未知 Python、Ray、AReaL、Verl、vLLM 或其他训练任务。
- 不要使用 `pkill python` 等宽泛命令。
- 训练脚本检测到 GPU compute process 时必须退出，而不是抢占。
- 每次运行使用唯一的 `ARTIFACT_ROOT`，不要覆盖历史 checkpoint。
- 模型、数据集、config 和代码 revision 必须写入运行 manifest。
- 报告训练完成前，必须复查 PID、GPU、日志和 checkpoint。

## Appendix

### Appendix A：8×A6000 复现建议

8 张 RTX A6000 可以用于功能复现和训练验证，但不属于与 8×H100 完全等价的
硬件复现。A6000 每张 48 GB；当前 Qwen3.5-9B 生产配置是在 8×H100 80 GB
上验证的，因此必须建立独立配置和 artifact 目录，并先通过短训练 gate。

建议保留 4+4 GPU 拓扑，降低 group size、并发和 vLLM 显存比例：

```yaml
cluster:
  n_nodes: 1
  n_gpus_per_node: 8

rollout:
  backend: "vllm:d4p1t1"
  max_concurrent_rollouts: 2

actor:
  backend: "fsdp:d4p1t1"

train_dataset:
  batch_size: 4

gconfig:
  n_samples: 4

eval_gconfig:
  n_samples: 4

vllm:
  gpu_memory_utilization: 0.55
```

约束：

- `batch_size` 必须能被 actor degree `d4` 整除；
- `n_samples` 必须能被 rollout degree `d4` 整除；
- 保持 `gradient_checkpointing: true`、`attn_impl: sdpa`；
- 初次 gate 保持 `max_head_offpolicyness: 0`；
- 当前 `sampled12_uniform_shaped` contract 固定要求 `n_samples=12`，
  使用 group 4 前必须增加对应 validator，不能只改 YAML。

建议先跑 2-update gate，并验证 rollout、reference log-prob、optimizer update、
checkpoint save 和每张 GPU 峰值显存。OOM 时依次尝试：

```text
1. max_concurrent_rollouts: 2 -> 1
2. vllm.gpu_memory_utilization: 0.55 -> 0.50
3. actor.mb_spec.max_tokens_per_mb: 640 -> 512 或 384
4. 使用更小模型完成系统 smoke
```

通过 A6000 gate 可以证明功能和训练链路可复现，不能证明吞吐或最终指标与
H100 完全一致。

### Appendix B：官方 AReaL checkout 策略

node1 和 node5 上的框架开发与机器人开发必须保持分离：

```text
/home/ubuntu/z00819216/xinglu/AReaL
  branch: areal-main
  tracks: https://github.com/inclusionAI/AReaL.git main

/home/ubuntu/z00819216/xinglu/AReaL-VLA
  branch: robotics/morgan-vla
  保留 Morgan/robotics 工作和现有 dirty state
```

`maapacman-rl` Conda 环境和
`scripts/level1/train/run_level1_training.sh` 必须从干净的官方 `AReaL`
worktree 解析 `areal`。不要从 `AReaL-VLA` 启动 Pacman 训练。

### Appendix C：Native multimodal workflow

`areal_pacman.level1.workflow.PacmanNativeVisionWorkflow` 实现配方侧原生
AReaL `RolloutWorkflow`。

每个动作请求只处理一次，并返回对齐的：

```text
input_ids
mm_token_type_ids
pixel_values
image_grid_thw
rollout log-probs
versions
masks
environment reward
```

生产 workflow 不使用历史 OpenAI proxy 轨迹路径，以避免遗漏
`mm_token_type_ids` 和 `multi_modal_input`。

动态 open-action mask 同时应用于：

- rollout generation；
- actor log-prob；
- reference log-prob。

因此配置使用：

```yaml
top_p: 1.0
logprobs_mode: processed_logprobs
```

### Appendix D：完整测试

运行完整 CPU 测试：

```bash
python -m pytest -q
```

运行 Level 1 核心测试：

```bash
python -m pytest -q tests/test_level1_recipe.py
```

执行生产 config dry-run：

```bash
python train_areal.py \
  --config configs/level1/train/level1_live_state_step256_100update_group12_8gpu.yaml \
  --dry-run \
  --validate-areal
```

### Appendix E：轨迹记录与审计

设置 `PACMAN_TRAJECTORY_DIR` 后，每个 rollout episode 会保存为独立 JSON：

```bash
export PACMAN_TRAJECTORY_DIR="$ARTIFACT_ROOT/training/trajectories"
```

文件名同时包含 dataset row ID 和随机 `trajectory_sample_id`，因此 GRPO 的重复
sample 不会互相覆盖。

每条轨迹记录：

- 环境和代码 revision；
- prompt 与 decoding contract；
- 模型原始响应；
- base reward 和 shaped reward 明细；
- score、豆子、撞墙、revisit 和终止状态；
- observation image hash。

汇总轨迹：

```bash
python scripts/level1/report/summarize_level1_trajectories.py \
  "$ARTIFACT_ROOT/training/trajectories"
```

### Appendix F：Base 与 checkpoint A/B demo

`scripts/level1/report/run_level1_ab_demo.sh` 使用相同 Level 1 seed、prompt、
decoding 和动态 open-action mask，分别录制 base model 与最新完整 checkpoint。

正常终点是死亡或清关。为了避免报告任务无限运行，还设置：

```text
最大 2,000 步
最大 600 秒
连续 100 步没有 score 或普通豆进展时停止
```

运行方式：

```bash
SOURCE_RUN=/path/to/training-run \
BASE_MODEL=/path/to/Qwen3.5-9B \
OUTPUT_ROOT=/path/to/ab-demo \
GPU_ID=0 \
bash scripts/level1/report/run_level1_ab_demo.sh
```

脚本会选择最高的完整 `globalstep`，补回训练 checkpoint 中省略的冻结
base-model tensors，每次只启动一个私有 vLLM server，并导出带有
step/action/reward/score/pellet overlay 的 PowerPoint 兼容 H.264 MP4。

脚本在 GPU 忙碌时会拒绝启动，不会抢占现有任务。

### Appendix G：评估工具

- `scripts/level1/evaluate/evaluate_level1.py`：独立 Level 1 evaluator。
- `scripts/level1/evaluate/evaluate_level1_run.sh`：base 与 checkpoint 顺序评估。
- `scripts/level1/evaluate/evaluate_level1_sampled_run.sh`：sampled 评估。
- `scripts/level1/evaluate/compare_level1_run.py`：生成确定性对比报告。
- `scripts/level1/report/summarize_level1_trajectories.py`：训练/验证轨迹审计。
- `scripts/level1/report/audit_level1_rewards.py`：reward 公式审计。
- `scripts/level1/report/build_complete_vlm_checkpoint.py`：构建完整 VLM checkpoint。

### Appendix H：Node1 完整环境安装与验证

本仓库自身的基础依赖定义在 [pyproject.toml](pyproject.toml)：

```text
Python >= 3.10
maapacman >= 0.4, < 0.5
numpy >= 1.24
Pillow >= 10
```

以上范围不足以完全复现训练。下面是 2026-07-30 从 node1 实际
`maapacman-rl` 环境读取的版本：

```text
NVIDIA driver==590.48.01
CUDA used by PyTorch==13.0

python==3.12.13
areal==1.0.4
torch==2.11.0+cu130
torchvision==0.26.0
transformers==5.7.0
vllm==0.22.1
datasets==5.0.0
accelerate==1.14.0
peft==0.18.1
tokenizers==0.22.2
safetensors==0.8.0
numpy==2.2.6
pillow==12.2.0
pygame==2.6.1
flashinfer-python==0.6.11.post2
torch-memory-saver==0.0.9
```

其中 `areal==1.0.4` 是从官方 AReaL checkout editable 安装的，不应替换为
同名但代码 revision 不同的其他包：

```text
/mnt/data/z00819216/xinglu/AReaL
```

#### 推荐的 Conda 安装顺序

在 Linux/H100 节点上创建独立环境：

```bash
export ENV_ROOT=/mnt/data/z00819216/conda_env/maapacman-rl

conda create --prefix "${ENV_ROOT}" \
  python=3.12.13 \
  pip \
  -y

conda activate "${ENV_ROOT}"
python -m pip install --upgrade pip setuptools wheel
```

安装与 node1 一致的 Python/GPU 包。PyTorch wheel 必须是 CUDA 13.0
兼容构建；安装后 `torch.__version__` 应显示 `2.11.0+cu130`：

```bash
python -m pip install \
  "torch==2.11.0" \
  "torchvision==0.26.0" \
  "transformers==5.7.0" \
  "vllm==0.22.1" \
  "datasets==5.0.0" \
  "accelerate==1.14.0" \
  "peft==0.18.1" \
  "tokenizers==0.22.2" \
  "safetensors==0.8.0" \
  "numpy==2.2.6" \
  "Pillow==12.2.0" \
  "pygame==2.6.1" \
  "flashinfer-python==0.6.11.post2" \
  "torch-memory-saver==0.0.9"
```

然后安装固定 revision 的三个本地 checkout：

```bash
python -m pip install -e /mnt/data/z00819216/xinglu/AReaL
python -m pip install -e /mnt/data/z00819216/maapacman-stack/MaaPacman
python -m pip install -e "/path/to/areal-pacman[dev,dataset,agent]"
```

实际 checkout 路径可不同，但 AReaL、MaaPacman、pacman-python 和
areal-pacman 的 Git revision 必须与目标运行 manifest 一致。

安装后验证：

```bash
python - <<'PY'
from importlib import metadata
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
for package in (
    "areal",
    "torchvision",
    "transformers",
    "vllm",
    "datasets",
    "accelerate",
    "peft",
    "tokenizers",
    "safetensors",
    "numpy",
    "pillow",
    "pygame",
    "flashinfer-python",
    "torch-memory-saver",
):
    print(f"{package}: {metadata.version(package)}")
PY
```

仅有相同 package version 仍不足以保证完全复现；正式运行还必须保存：

- 四个代码仓库的 Git commit；
- AReaL 本地 patch；
- 模型 revision/hash；
- dataset manifest；
- GPU、driver 和 CUDA 信息；
- 最终解析后的训练 config。
