# AReaL Pacman RL 配方

本仓库提供 MaaPacman Level 1 的 headless 游戏环境和 AReaL 强化学习配方：

- `maapacman`：真实 pygame Level 1 环境；
- `areal_pacman`：数据集、原生多模态 workflow、Reward v3、轨迹审计和评估工具；
- `configs/level1/` 与 `scripts/level1/`：训练配置和运行入口。

固定 revision 的 AReaL fork 提供分布式调度、vLLM rollout、FSDP actor/reference engine 和 checkpoint 管理。环境已内置在本仓库，无需独立的 MaaPacman checkout。

日常开发和交付统一使用 `release/maapacman-v0.1.0`。`backup/2026-09-04/*` 仅用于保留历史。

## 源码边界

| 源码层                                                        | 作用                                       | 当前配方使用的版本                                                         |
| ------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| 本仓库                                                        | `areal_pacman` 配方与内置 `maapacman` 环境 | `release/maapacman-v0.1.0`；运行时记录实际 SHA                             |
| [luzai/AReaL](https://github.com/luzai/AReaL)                 | 训练、rollout、FSDP 和 checkpoint          | `pacman/open-action-mask` @ `b40314ff84ae9422c9065f76065f880f599a0bb4`     |
| [luzai/pacman-python](https://github.com/luzai/pacman-python) | 游戏规则、资源和 pygame renderer           | `release/maapacman-v0.1.0` 中的 `cbb97115e407abc86a44adc82a1b8f360b3e8da0` |

复现时以固定 SHA 为准，不能仅依赖会继续更新的分支名。更新依赖 revision 后，需要同步运行 manifest 并重新验证。`areal_pacman.synthetic.*` 保留用于历史合成迷宫实验。

## 获取固定源码

在 Linux 服务器上执行，替换示例路径：

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
git -C pacman-python checkout cbb97115e407abc86a44adc82a1b8f360b3e8da0
```

三个 checkout 可以放在任意可写磁盘；以下命令使用并列目录：

```text
${WORKSPACE_ROOT}/
  AReaL/
  areal-pacman/
  pacman-python/
```

正式运行前保存实际 revision：

```bash
git -C "$WORKSPACE_ROOT/AReaL" rev-parse HEAD
git -C "$WORKSPACE_ROOT/areal-pacman" rev-parse HEAD
git -C "$WORKSPACE_ROOT/pacman-python" rev-parse HEAD
```

正式复现使用 Git clone + editable install；GitHub `Download ZIP` 不含数据 manifest 所需的 `.git` provenance。

## 环境安装

历史 8×H100 gate 的关键版本为 Python `3.12.13`、PyTorch `2.11.0+cu130`、Transformers `5.7.0`、vLLM `0.22.1`、pygame `2.6.1` 和 torch-memory-saver `0.0.9`。这些是历史环境记录，GPU wheel 仍需与目标节点的 driver/CUDA 匹配。

创建环境并注册两个本地 Python 项目：

```bash
export AREAL_ROOT="$WORKSPACE_ROOT/AReaL"
export AREAL_PACMAN_ROOT="$WORKSPACE_ROOT/areal-pacman"
export MAAPACMAN_PACMAN_PYTHON_ROOT="$WORKSPACE_ROOT/pacman-python"
export ENV_ROOT=/path/to/conda/envs/maapacman-rl

conda create --prefix "$ENV_ROOT" python=3.12.13 pip -y
conda activate "$ENV_ROOT"
export PYTHON="$ENV_ROOT/bin/python"

# 按固定 AReaL checkout 的安装说明准备 vLLM/FSDP GPU 依赖。
"$PYTHON" -m pip install -e "$AREAL_ROOT"
"$PYTHON" -m pip install -e "${AREAL_PACMAN_ROOT}[dev,dataset,agent]"
```

上面的 editable install 不替代 GPU 依赖安装。AReaL 兼容要求见[必需补丁说明](patches/README.md)。如果复用安装过独立 `maapacman` 的旧环境，先卸载旧 distribution，再安装本仓库。`pacman-python` 只作为固定、只读的源码 checkout 使用。

## 环境检查

以下命令检查导入和真实 headless 游戏环境，不启动 GPU 训练：

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

recipe_package = Path(areal_pacman.__file__).resolve().parent
environment_package = Path(maapacman.__file__).resolve().parent
assert recipe_package.parent == environment_package.parent
print("areal:", Path(areal.__file__).resolve())
print("areal_pacman:", recipe_package)
print("maapacman:", environment_package)

with PygamePacmanEnv() as env:
    image, info = env.reset(seed=0)
    assert info["env_id"] == "pacman-python-level1-ghostdoor-v3"
    assert info["backend"] == "original-pygame"
    assert image.shape == (400, 336, 3)
    print(info["env_id"], info["backend"], image.shape)
PY

CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q
```

## 两份正式训练配置

| 配置                                                      | 初始化模型                  | 默认完整训练            |
| --------------------------------------------------------- | --------------------------- | ----------------------- |
| [curriculum1.yaml](configs/level1/train/curriculum1.yaml) | `Qwen/Qwen3.5-9B`           | 100 次 optimizer update |
| [curriculum2.yaml](configs/level1/train/curriculum2.yaml) | `${CURRICULUM1_CHECKPOINT}` | 100 次 optimizer update |

两阶段使用相同关卡、画面尺寸与 prompt 字段，但具有明确不同的幽灵模式：

| 项目                     | Curriculum 1       | Curriculum 2           |
| ------------------------ | ------------------ | ---------------------- |
| `environment.ghost_mode` | `disabled`，无幽灵 | `normal`，正常移动幽灵 |
| 输入中的 `ghosts`        | `[]`               | 正常幽灵状态列表       |
| 每局步数                 | 32                 | 256                    |
| 训练 / 验证 seeds        | 0–7 / 8–9          | 100–139 / 140–147      |
| Epochs                   | 50                 | 10                     |
| 学习率                   | `1e-6`             | `5e-7`                 |
| `nearest_pellet_alpha`   | `0.1`              | `0.02`                 |

两者默认均为 8 张 GPU、batch size 4、100 次更新，每个 prompt 采样 12 条轨迹。
C1 学导航与吃豆，C2 加入避敌、能量豆和通关。地图与初始位置不变，seeds 不是不同地图；这些是发布默认值，不代表已验证最优超参数。
独立 ghost 开关不关闭水果、不改变奖励事件定义；C1 的能量豆仍得分，但没有幽灵易受攻击计时。
启动器从 YAML 读取数据规模与步数；环境模式同时写入 dataset、run manifest 和每个原子帧，规则 hash 按模式区分。配置与数据不匹配时拒绝训练。
旧数据没有 `ghost_mode`，须用当前固定源码重新生成；不要手工补字段或复用旧规则 hash。

动作协议为 Edward option code，使用 `live_state_v3` 输入、`temperature=0.7`、`top_p=1.0` 和 `episode_return_group_v1` 目标。当前 YAML 未启用定期验证 rollout。奖励细节以 [rewards.py](areal_pacman/level1/rewards.py) 和配置为准。

## 启动训练

使用上面的路径与已激活环境，在仓库根目录设置一次：

```bash
cd "$AREAL_PACMAN_ROOT"
export OWNER_ROOT=/path/to/writable/owner-root
export ENV_ROOT="$CONDA_PREFIX"
export PYTHON="$ENV_ROOT/bin/python"
export MODEL_PATH=/path/to/Qwen3.5-9B
# 使用选中 YAML 的数据默认值，避免上次运行的环境变量覆盖本次配置。
unset TRAIN_EPISODES VALIDATION_EPISODES DATASET_MAX_STEPS

# 让启动器为每次运行自动生成新的名称、产物和数据集目录。
unset RUN_ID ARTIFACT_ROOT DATASET_OUTPUT_ROOT
```

先对 Curriculum 1 做两次更新的 smoke test：

```bash
CONFIG=configs/level1/train/curriculum1.yaml \
bash scripts/level1/train/run_level1_training.sh --smoke-updates 2
```

确认 smoke 结果后，从 Qwen3.5-9B 开始完整训练：

```bash
CONFIG=configs/level1/train/curriculum1.yaml \
bash scripts/level1/train/run_level1_training.sh
```

Curriculum 1 完成后，指定其完整模型 checkpoint，启动 Curriculum 2：

```bash
export CURRICULUM1_CHECKPOINT=/path/to/curriculum1-complete-checkpoint

CONFIG=configs/level1/train/curriculum2.yaml \
bash scripts/level1/train/run_level1_training.sh
```

`CURRICULUM1_CHECKPOINT` 必须包含模型权重、配置和所需 tokenizer/processor 文件；不能直接指向训练产物根目录或不完整的恢复目录。需要补齐冻结视觉权重时，参见[完整 VLM checkpoint 工具](scripts/level1/report/build_complete_vlm_checkpoint.py)。

Curriculum 2 继承模型权重并重新初始化 optimizer/scheduler。两份配置首次运行均为 `recover.mode: disabled`。
注意：在当前 AReaL 中，`disabled` 同时禁止保存恢复状态；中断后才改为 `auto`，无法补回之前未保存的 optimizer/dataloader 状态。
需要完整中断恢复的任务应在首次启动时就显式启用 `auto`，并保留同一 run 的数据、名称与路径；当前发布启动器面向新 run，不会覆盖或重建旧数据。
只有确实存在完整恢复状态时，才应使用原始训练命令加 `recover.mode=auto` 恢复；只有模型 checkpoint 时属于权重初始化新 run。

Smoke 使用同一份 YAML，通过 `total_train_steps=2` 限制更新次数，不需要第三份配置。也可以直接向 `train_areal.py --config ...` 传入 `--smoke-updates 2`，但需要先准备对应的数据集和运行环境。

启动器会检查源码导入、模型文件、GPU 空闲状态、AReaL 补丁和 prompt budget，生成不可变数据集并执行配置 dry-run，然后启动训练。默认产物位于 `${OWNER_ROOT}/run_artifacts/maapacman-rl/<recipe>-<timestamp>/`，数据集保存在其中的 `dataset/`。指定自定义输出路径时应使用新目录。

两阶段已新增真实 headless 环境、确定性、能量豆后无幽灵、数据生成与篡改拒绝测试；分布式 GPU smoke 尚待运行，CPU 环境测试不能代替它。
Windows 可运行这些 CPU 检查；完整 AReaL workflow 的依赖包含 `uvloop`，仍需在 Linux 环境验证。

## 评估与产物

[评估工具目录](scripts/level1/evaluate/) 提供单模型评估、checkpoint 对比和结果汇总工具。目前旧批量评估脚本仍假设恰好 4 个 checkpoint，参数也需要与当前 Edward options 训练协议对齐，尚不能作为这两份 recipe 已验证的一键评估入口。

正式对比 Base、Curriculum 1 和 Curriculum 2 时，应固定环境 revision、seed、episode 步数和解码设置，记录通关率、清豆率和回报。轨迹审计工具与 checkpoint 工具见[脚本说明](scripts/README.md)。

## 运行约定

- 启动前确认所需 GPU 空闲；保留其他训练、推理和服务进程。
- 每次运行保存三仓 SHA、模型 revision、dataset manifest、实际配置及 driver/CUDA 信息。
- 根据日志中的 update 标记、checkpoint 和评估产物确认结果，不能仅凭启动器退出或进程存在判断完成。

## 详细文档

- [配置说明](configs/README.md)
- [配方架构与历史验证](docs/architecture/AREAL_RECIPE_DESIGN.md)
- [AReaL 必需补丁](patches/README.md)
- [数据集、训练、评估和报告脚本](scripts/README.md)
- [运行产物与保留策略](RUN_ARTIFACTS.md)
- [第三方来源与署名](THIRD_PARTY_NOTICES.md)

本仓库目前尚未选择项目级开源许可证；第三方来源和署名说明见 `THIRD_PARTY_NOTICES.md`。
