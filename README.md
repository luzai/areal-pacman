# AReaL Pacman RL 配方

本仓库提供 MaaPacman Level 1 的 headless 游戏环境和 AReaL 强化学习配方：

- `maapacman`：真实 pygame Level 1 环境；
- `areal_pacman`：数据集、原生多模态 workflow、Reward v3、轨迹审计和评估；
- `configs/level1/` 与 `scripts/level1/`：可复现的训练、验证和报告入口。

固定 revision 的 AReaL fork 提供分布式调度、vLLM rollout、FSDP actor/reference engine 和 checkpoint 管理。生产路径不再需要独立 MaaPacman checkout。

## 源码边界

| 源码层 | 作用 | 当前兼容 pin |
| -- | -- | -- |
| 本仓库 | `areal_pacman` 配方与内置 `maapacman` 环境 | `release/maapacman-v0.1.0` |
| [luzai/AReaL](https://github.com/luzai/AReaL) | 训练、rollout、FSDP 和 checkpoint | `pacman/open-action-mask` @ `b40314ff84ae9422c9065f76065f880f599a0bb4` |
| [luzai/pacman-python](https://github.com/luzai/pacman-python) | 原版规则、资源和 pygame renderer | `release/maapacman-v0.1.0` @ `01eff954d4ee09bcc0937b77fea7d382496e891b` |

`b40314ff` 是本发布候选已验证的 AReaL 兼容 commit。以后发布 clean AReaL release branch 时，必须同时更新 branch、SHA 和运行 manifest，并重新执行 smoke test。

`areal_pacman.synthetic.*` 仅保留给历史合成迷宫实验，不是 Level 1 生产环境。

## 获取固定源码

```bash
export WORKSPACE_ROOT=/path/to/pacman-release
mkdir -p "$WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"

git clone --branch pacman/open-action-mask --single-branch \
  https://github.com/luzai/AReaL.git
git -C AReaL checkout b40314ff84ae9422c9065f76065f880f599a0bb4

git clone --branch release/maapacman-v0.1.0 --single-branch \
  https://github.com/luzai/areal-pacman.git

git clone --branch release/maapacman-v0.1.0 --single-branch \
  https://github.com/luzai/pacman-python.git
git -C pacman-python checkout 01eff954d4ee09bcc0937b77fea7d382496e891b
```

目录可以放在任意可写磁盘，但三层源码必须彼此独立：

```text
${WORKSPACE_ROOT}/
  AReaL/
  areal-pacman/
  pacman-python/
```

正式运行前记录三个实际 revision：

```bash
git -C "$WORKSPACE_ROOT/AReaL" rev-parse HEAD
git -C "$WORKSPACE_ROOT/areal-pacman" rev-parse HEAD
git -C "$WORKSPACE_ROOT/pacman-python" rev-parse HEAD
```

不要用 GitHub `Download ZIP` 代替训练 checkout；ZIP 不含 `.git` provenance。

## 环境安装

当前验证过的关键环境为 Linux、Python `3.12.13`、PyTorch `2.11.0+cu130`、Transformers `5.7.0`、vLLM `0.22.1`、pygame `2.6.1` 和 torch-memory-saver `0.0.9`。GPU wheel 必须与目标节点的 driver/CUDA 兼容。

先按照固定 AReaL checkout 准备其 vLLM/FSDP GPU 依赖，再注册两个 Python 项目：

```bash
export AREAL_ROOT="$WORKSPACE_ROOT/AReaL"
export AREAL_PACMAN_ROOT="$WORKSPACE_ROOT/areal-pacman"
export MAAPACMAN_PACMAN_PYTHON_ROOT="$WORKSPACE_ROOT/pacman-python"
export ENV_ROOT=/path/to/conda/envs/maapacman-rl

conda create --prefix "$ENV_ROOT" python=3.12.13 pip -y
conda activate "$ENV_ROOT"
export PYTHON="$ENV_ROOT/bin/python"

"$PYTHON" -m pip install -e "$AREAL_ROOT"
"$PYTHON" -m pip install -e "${AREAL_PACMAN_ROOT}[dev,dataset,agent]"
```

如果复用曾安装独立 `maapacman` 的旧环境，先卸载旧 distribution，再重新安装本仓库。`pacman-python` 只作为固定、只读的源码 checkout 使用。

## 环境 smoke test

```bash
cd "$AREAL_PACMAN_ROOT"
export PYTHONPATH="$AREAL_ROOT:$AREAL_PACMAN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

"$PYTHON" - <<'PY'
from pathlib import Path

import areal
import areal_pacman
import maapacman
from maapacman.env import PygamePacmanEnv

print("areal:", Path(areal.__file__).resolve())
print("areal_pacman:", Path(areal_pacman.__file__).resolve())
print("maapacman:", Path(maapacman.__file__).resolve())

with PygamePacmanEnv() as env:
    image, info = env.reset(seed=0)
    assert info["env_id"] == "pacman-python-level1-ghostdoor-v3"
    assert info["backend"] == "original-pygame"
    assert image.shape == (400, 336, 3)
    print(info["env_id"], info["backend"], image.shape)
PY
```

运行完整 headless 测试与配置 dry-run：

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q

AREAL_ADMIN_API_KEY=local-dry-run-only "$PYTHON" train_areal.py \
  --config configs/level1/train/level1_edward_step512_2update_group12_8gpu.yaml \
  --dry-run \
  --validate-areal
```

占位 `AREAL_ADMIN_API_KEY` 只允许用于不启动服务的 dry-run。正式启动器会生成随机 key。

2026-09-04 的发布 smoke 在 `H100_2_1` 别名对应的 8×NVIDIA H800 节点完成。测试时隐藏 GPU，未启动训练：

```text
258 passed, 25 subtests passed, 0 failed
config dry-run: passed
```

该结果证明导入、环境契约、CPU/headless 测试和启动配置可用，不代表完成了一次分布式 GPU 训练。

## Step512 / 2-update gate

默认入口使用 [`level1_edward_step512_2update_group12_8gpu.yaml`](configs/level1/train/level1_edward_step512_2update_group12_8gpu.yaml)：

| 项目 | 当前值 |
| -- | -- |
| 模型 | Qwen3.5-9B |
| GPU | 8 |
| 训练 / 验证 episode | 4 / 2 |
| optimizer updates | 2 |
| rollout samples / update | 48（4×12） |
| episode 上限 | 512 environment steps |
| decoding | `n_samples=12`、`temperature=0.7`、`top_p=1.0`、thinking disabled |
| 动作协议 | Edward options；`open_action_mask=false` |
| reward | `maapacman-level1-event-reward-v3` |
| objective | `option_return_raw_v1` |
| validation | `sampled12_uniform_shaped` |

Reward v3 的唯一准确信息源是 [`areal_pacman/level1/rewards.py`](areal_pacman/level1/rewards.py) 和上述 YAML。README 不再复制容易漂移的奖励公式。

确认配置所需的全部 GPU 空闲后启动：

```bash
cd "$AREAL_PACMAN_ROOT"
export OWNER_ROOT=/path/to/writable/owner-root
export ENV_ROOT="$CONDA_PREFIX"
export PYTHON="$ENV_ROOT/bin/python"
export AREAL_ROOT="$WORKSPACE_ROOT/AReaL"
export MAAPACMAN_PACMAN_PYTHON_ROOT="$WORKSPACE_ROOT/pacman-python"
export MODEL_PATH=/path/to/Qwen3.5-9B
export CONFIG="$AREAL_PACMAN_ROOT/configs/level1/train/level1_edward_step512_2update_group12_8gpu.yaml"
export DATASET_MAX_STEPS=512
export RUN_ID="edward-v3-step512-$(date -u +%Y%m%dT%H%M%SZ)"
export ARTIFACT_ROOT="/path/to/run-artifacts/$RUN_ID"

bash scripts/level1/train/run_level1_training.sh
```

启动器会验证源码导入、模型、GPU 空闲状态、AReaL 补丁和 prompt budget，然后生成不可变数据集、保存 manifest/config 并启动训练。不要复用已有 `DATASET_OUTPUT_ROOT` 或覆盖历史 `ARTIFACT_ROOT`。

## 安全与可审计性

- 不停止未知 Python、Ray、AReaL、Verl 或 vLLM 进程；不要使用 `pkill python`。
- 每次运行使用唯一 artifact 目录，并保存三仓 SHA、模型 revision、dataset manifest、解析后的 config、driver 和 CUDA 信息。
- 只有 PID/GPU、日志、checkpoint 和评估结果都复核后，才报告训练或恢复完成。

## 详细文档

- [配置说明](configs/README.md)
- [配方架构与历史验证](docs/architecture/AREAL_RECIPE_DESIGN.md)
- [AReaL 必需补丁](patches/README.md)
- [数据集、训练、评估和报告脚本](scripts/README.md)
- [运行产物与保留策略](RUN_ARTIFACTS.md)
- [第三方来源与署名](THIRD_PARTY_NOTICES.md)

本仓库目前尚未选择项目级开源许可证；第三方来源和署名要求见 `THIRD_PARTY_NOTICES.md`。
