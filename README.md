# AReaL PacMan RL 配方

本仓库同时发布 MaaPacman Level 1 的 headless 游戏环境和 AReaL 强化学习配方：
`maapacman` 包负责真实游戏环境，`areal_pacman` 包负责数据集、训练 workflow、
奖励塑形、训练配置、轨迹审计和评估工具。分布式训练、vLLM rollout、FSDP
actor/reference engine 和 checkpoint 管理由现有 AReaL fork 提供。

## 架构文档

- [AReaL Pacman RL 配方设计](docs/architecture/AREAL_RECIPE_DESIGN.md)
- [运行产物说明](RUN_ARTIFACTS.md)

## 系统边界

```text
areal-pacman 仓库（本仓库）
  maapacman/
    提供 maapacman.env.PygamePacmanEnv，拥有真实 pygame Level 1
    状态转移、RGB 渲染、合法动作、奖励、终止条件和游戏指标。
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

AReaL fork checkout
  提供 trainer、rollout workers、vLLM、FSDP actor/reference engines、
  调度和 checkpoint 发布。

pacman-python checkout
  提供原版游戏规则、资源和 pygame renderer。
```

这些边界是有意设计的：

- 本仓库内置的 `maapacman` 包拥有游戏环境；
- 本仓库的 `areal_pacman` 包拥有 RL 配方；
- 现有 AReaL fork 拥有分布式训练系统；
- `pacman-python` 提供固定 revision 的原版游戏源码和资源。

`areal_pacman.synthetic.*` 仅用于历史文本/合成迷宫实验，不能替代真实
Level 1 生产环境。

## 三层源码与环境安装

生产 Level 1 需要固定下面三层源码；不再需要独立的 MaaPacman checkout：

```text
<workspace>/
  AReaL/          # 现有 AReaL fork，需要作为 Python 包安装
  areal-pacman/   # 同时包含 areal_pacman 和 maapacman，需要安装
  pacman-python/  # 原版 pygame 游戏源码，只需 checkout，不作为 pip 包安装
```

实际路径可以不同，但必须锁定这三个 Git revision。`pacman-python` 必须保持为
独立、固定 revision 的 checkout，并在训练期间按只读源码使用。

正式训练与数据 manifest 的发布合约是 **Git clone + editable install**。不要用
GitHub `Download ZIP` 或仅安装 wheel 代替源码 checkout：这两种形式不含 `.git`
revision，只适合 import/环境 smoke test，无法生成可审计的三仓库 provenance。

### 新 Linux/H100 服务器（从零安装）

推荐使用三个并列、固定 revision 的 checkout；实际绝对路径不构成合约：

```text
  /path/to/AReaL
  /path/to/areal-pacman
  /path/to/pacman-python
```

下面是历史 node1 gate 已验证的关键环境版本，可作为锁定环境的参考：

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

从零创建环境并安装三层源码依赖：

```bash
# 1. 创建并激活 Conda 环境
conda create -n maapacman-rl python=3.12.13 pip -y
conda activate maapacman-rl

# 2. 将启动器绑定到当前激活的环境和当前用户可写目录
export OWNER_ROOT="$HOME"
export ENV_ROOT="${CONDA_PREFIX:?activate maapacman-rl first}"
export PYTHON="${ENV_ROOT}/bin/python"

# 3. 安装固定版本的 GPU/Python 依赖
# 使用上方记录的 node1 已验证版本。
# PyTorch wheel 必须兼容服务器 CUDA；node1 使用 torch 2.11.0+cu130。

# 4. 如果复用曾安装独立 maapacman 的旧环境，先移除旧 distribution
"${PYTHON}" -m pip uninstall -y maapacman

# 5. 安装两个 Python 项目；areal-pacman 会同时安装两个本地包
"${PYTHON}" -m pip install -e "/path/to/AReaL"
"${PYTHON}" -m pip install -e "/path/to/areal-pacman[dev,dataset,agent]"

# 6. pacman-python 不需要 pip 安装，只需指向固定 revision 的 checkout
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
```

为当前 checkout 显式设置路径：

```bash
export OWNER_ROOT="$HOME"
export ENV_ROOT="${CONDA_PREFIX:?activate maapacman-rl first}"
export PYTHON="${ENV_ROOT}/bin/python"
export AREAL_ROOT=/path/to/AReaL
export AREAL_PACMAN_ROOT=/path/to/areal-pacman
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
```

确认三个源码层均解析正确，并确认两个本地 Python 包来自同一
`areal-pacman` checkout：

```bash
"${PYTHON}" - <<'PY'
from pathlib import Path

import areal
import areal_pacman
import maapacman
from maapacman.env import PygamePacmanEnv

print("areal:", Path(areal.__file__).resolve())
print("areal_pacman:", Path(areal_pacman.__file__).resolve())
print("maapacman:", Path(maapacman.__file__).resolve())

recipe_package = Path(areal_pacman.__file__).resolve().parent
environment_package = Path(maapacman.__file__).resolve().parent
assert recipe_package.parent == environment_package.parent, (
    "stale standalone maapacman installation",
    recipe_package,
    environment_package,
)

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
env_id: pacman-python-level1-ghostdoor-v3
api_version: 3.0
backend: original-pygame
```

## 准备 Level 1 数据集

下面两条命令用于独立生成和审计数据集。标准训练启动器也会生成自己的不可变
数据集，因此不要再把这里已经存在的输出目录传给启动器。

短 episode（32 步）：

```bash
"${PYTHON}" scripts/level1/dataset/prepare_level1_dataset.py \
  --output-root artifacts/datasets/level1_dataset_step32 \
  --config configs/level1/train/level1_edward_step512_2update_group12_8gpu.yaml \
  --train-episodes 8 \
  --validation-episodes 2 \
  --max-steps 32 \
  --write-hf
```

长 episode（256 步）：

```bash
"${PYTHON}" scripts/level1/dataset/prepare_level1_dataset.py \
  --output-root artifacts/datasets/level1_dataset_step256 \
  --config configs/level1/train/level1_edward_step512_2update_group12_8gpu.yaml \
  --train-episodes 8 \
  --validation-episodes 2 \
  --max-steps 256 \
  --write-hf
```

`max_steps` 会写入每一条 episode 数据，因此不能把旧的 step32 目录简单改名为
step256。

## Linux/H100 训练入口

生产训练要求：

- 现有 AReaL fork checkout；
- 本仓库（包括内置的 `maapacman` 包）和 `pacman-python` checkout；
- 本地 Qwen 模型 checkpoint；
- 与配置一致的数据集；
- 配置要求数量的空闲 GPU。

标准启动方式：

```bash
export OWNER_ROOT="$HOME"
export ENV_ROOT="${CONDA_PREFIX:?activate maapacman-rl first}"
export PYTHON="${ENV_ROOT}/bin/python"
export AREAL_ROOT=/path/to/AReaL
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
export MODEL_PATH=/path/to/Qwen3.5-9B
export CONFIG="$PWD/configs/level1/train/level1_edward_step512_2update_group12_8gpu.yaml"
export DATASET_MAX_STEPS=512
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID="edward-v3-step512-${RUN_TS}"
export DATASET_OUTPUT_ROOT="$PWD/artifacts/datasets/${RUN_ID}"
export ARTIFACT_ROOT="$PWD/run_artifacts/${RUN_ID}"

bash scripts/level1/train/run_level1_training.sh
```

启动器会依次：

1. 验证 Python、模型和固定 revision 的 AReaL fork checkout；
2. 验证所选 GPU 没有 compute process；
3. 对 Edward 配置使用真实 Qwen processor 验证完整 10-candidate 图文输入小于 `vllm.max_model_len`；
4. 生成训练/验证数据集；
5. 执行 config dry-run；
6. 保存 config 和数据 manifest；
7. 启动正式训练；
8. 将 checkpoint、trajectory 和日志写入 `ARTIFACT_ROOT`。

上面的标准命令会用同一个唯一 `RUN_ID` 创建独立的数据集和训练产物目录。
`DATASET_OUTPUT_ROOT` 在启动前必须不存在；若要改到其他磁盘，请把它和
`ARTIFACT_ROOT` 都改为当前用户可写的新目录。

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

### Appendix B：AReaL fork checkout 策略

每个训练节点上的框架 checkout 与不相关的开发 worktree 必须保持分离：

```text
${AREAL_ROOT}
  branch: areal-main
  tracks: https://github.com/inclusionAI/AReaL.git main

${UNRELATED_AREAL_ROOT}
  branch: <unrelated-development-branch>
  保留不相关的开发工作和现有 dirty state
```

`maapacman-rl` Conda 环境和
`scripts/level1/train/run_level1_training.sh` 必须从干净、固定 revision 的 `AReaL`
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
maapacman（随 areal-pacman 打包，使用同一个 Git revision）
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

其中 `areal==1.0.4` 是从固定 revision 的 AReaL fork checkout editable 安装的，不应替换为
同名但代码 revision 不同的其他包：

```text
${AREAL_ROOT}
```

#### 推荐的 Conda 安装顺序

在 Linux/H100 节点上创建独立环境：

```bash
export ENV_ROOT="${ENV_ROOT:-${HOME}/.conda/envs/maapacman-rl}"
export AREAL_ROOT="${AREAL_ROOT:-/path/to/AReaL}"

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

然后安装固定 revision 的两个 Python checkout。若该环境曾安装独立
`maapacman` distribution，必须先卸载它；`maapacman` 随 `areal-pacman`
一起安装，`pacman-python` 只需提供固定 revision 的源码目录：

```bash
python -m pip uninstall -y maapacman
python -m pip install -e "${AREAL_ROOT}"
python -m pip install -e "/path/to/areal-pacman[dev,dataset,agent]"
```

实际 checkout 路径可不同，但 AReaL、pacman-python 和 areal-pacman 的 Git
revision 必须与目标运行 manifest 一致；内置 `maapacman` 的 revision 就是
areal-pacman 的 revision。

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

- 三个代码仓库的 Git commit；
- AReaL 本地 patch；
- 模型 revision/hash；
- dataset manifest；
- GPU、driver 和 CUDA 信息；
- 最终解析后的训练 config。
