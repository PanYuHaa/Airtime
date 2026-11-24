<div align="center">

# Airtime

**从一段滑雪视频，重建滑雪者的 3D 骨架。**

单目视频 → 目标锁定 → 2D 关键点 → 3D 人体 → 融合骨架

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 这是什么

给定一段自由式滑雪（空中技巧 / 坡面障碍）的视频，Airtime 输出**逐帧的 3D 人体骨架**——
不是 2D 关节点，而是能在三维空间里看姿态、算转体角度、做动作分解的骨架。

```
输入视频 (mp4/mov)
      │
      ▼
   M0 预处理 ──── 转码 1080p30 / 旋转校正 / 镜头切分
      │
      ▼
   M1 跟踪 ────── YOLOX 检测 + SAM2 分割，锁定「正在做动作的人」
      │
      ▼
   M2 2D 关键点 ─ RTMPose-x (Halpe26)，26 个关节点
      │
      ▼
   M3 3D 重建 ─── GEM-X，SOMA 77 关节
      │
      ▼
   M3F 融合 ───── 2D/3D 对齐 + 逐关节校正 → 最终骨架
      │
      ▼
   输出 skeleton.json / tracking.json / 叠加视频
```

### 特点

- **锁定做动作的人**：不按「出现时长」选目标，按速度、腾空、跨越幅度打分，
  避免锁到旁边扛雪板走路的人
- **2D/3D 双模型互证**：两个独立模型对同一关节的观测互相印证，剔除错误点
- **骨架不撕裂**：融合后做骨长检查，超差的骨头退回原始投影
- **确定性输出**：同一输入重复运行，结果逐位一致

---

## 安装

### 依赖总览

| 组件 | 用途 | 要求 |
|---|---|---|
| Python | 主流程（M0/M1/M2/M3F/M8） | 3.10+ |
| ffmpeg | 视频转码与切分 | 4.x+，需在 PATH |
| CUDA | 推理加速 | 11.8+（CPU 也能跑，很慢） |
| GEM-X | M3 三维重建 | **独立环境**，见下 |

### 1. 系统依赖

**Ubuntu / WSL2**

```bash
sudo apt update
sudo apt install -y ffmpeg python3-pip python3-venv git
```

**Windows**（建议在 WSL2 下运行，原生 Windows 未经完整测试）

装 [WSL2](https://learn.microsoft.com/windows/wsl/install) 后按上面 Ubuntu 步骤操作。

### 2. 主流程环境

```bash
git clone https://github.com/PanYuHaa/Airtime.git
cd Airtime

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

如果要用 GPU 版 onnxruntime：

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### 3. GEM-X 环境（M3 三维重建）

GEM-X 需要独立的 torch + CUDA 环境。按官方仓库安装：

```bash
# 克隆 GEM-X（NVlabs 官方）
git clone https://github.com/NVlabs/GEM-X.git ~/GEM-X

# 建独立环境
conda create -n sp-gem python=3.10 -y
conda activate sp-gem
cd ~/GEM-X
pip install -r requirements.txt
# 按官方说明装 CUDA 版 torch
```

装好后告诉 Airtime 去哪里找它：

```bash
export AIRTIME_GEMX=~/GEM-X
export AIRTIME_GEMX_PYTHON=~/miniconda3/envs/sp-gem/bin/python
```

### 4. 下载模型权重

权重体积大，不进仓库。都放到 `models/` 下：

```bash
mkdir -p models

# (1) YOLOX 人体检测 —— rtmlib 会自动下载；也可手动放到 models/
#     yolox_x_8xb8-300e_humanart-a39d44ed.onnx

# (2) RTMPose-x Halpe26 关键点
#     https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/
#     rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288.onnx

# (3) SAM2 分割
#     https://github.com/facebookresearch/sam2
#     models/sam2/          整个代码目录
#     models/sam2_hiera_large.pt

# (4) GEM-X 权重
#     按 GEM-X 官方说明下载到 models/GEM-X/
```

目录应该长这样：

```
models/
├── yolox_x_8xb8-300e_humanart-a39d44ed.onnx
├── rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288.onnx
├── sam2_hiera_large.pt
├── sam2/                      ← SAM2 代码
└── GEM-X/                     ← GEM-X 代码与权重
    └── checkpoints/gem_x.ckpt
```

### 5. 检查环境

```bash
airtime-check --strict
```

全部通过会打印 `✓ 环境自检通过`。有问题会列出缺什么、该放哪里。

---

## 快速开始

```bash
# 一行跑完
airtime input.mp4

# 指定输出目录
airtime input.mp4 --out runs/my_ski

# 只要骨架数据，不要叠加视频（更快）
airtime input.mp4 --no-overlay

# 只跑到某个阶段
airtime input.mp4 --stage fuse
```

或者用模块方式：

```bash
python -m airtime.run input.mp4
```

### 输出

```
runs/my_ski/
├── clips/            预处理后的视频
├── meta/             每个视频的元信息
├── tracking/         逐帧跟踪框与状态
├── segments/         骨架段划分
├── pose2d/           2D 关键点 (npz)
├── gem/              3D 模型原始输出
├── skeleton/         ★ 最终骨架 (json)
└── overlay/          骨架叠加视频
```

`skeleton/<name>.json` 是主产物：

```json
{
  "video": "my_ski",
  "fps": 30,
  "joint_names": ["Hips", "Spine1", "..."],
  "frames_data": [
    {
      "i": 0,
      "t": 0.0,
      "status": "skeleton",
      "joints": { "Hips": [812.3, 645.1], "Spine1": [815.0, 620.4], "...": [] }
    },
    {
      "i": 1,
      "t": 0.033,
      "status": "track_only",
      "joints": null
    }
  ]
}
```

`status` 有三种：

| 值 | 含义 |
|---|---|
| `skeleton` | 这一帧有可靠骨架 |
| `track_only` | 只跟到了人，但骨架不可信（太小 / 遮挡 / 模糊） |
| `none` | 这一帧没跟到 |

---

## 单独调用各阶段

每个阶段都能独立跑，方便调试和替换模型：

```bash
# M0 预处理
python -m airtime.pipeline.m0_prep --videos data/videos --out runs/demo

# M1 跟踪
python -m airtime.pipeline.m1_track --pkg runs/demo

# M2 2D 关键点
python -m airtime.pipeline.m2_pose2d --track runs/demo --device cuda

# M3 三维重建（需 GEM-X 环境）
python -m airtime.pipeline.m3_gemx --pkg runs/demo

# M3F 融合
python -m airtime.pipeline.m3f_fuse --out runs/demo

# M8 叠加视频
python -m airtime.pipeline.m8_overlay --pkg runs/demo
```

各阶段的 `--help` 有完整参数说明。

---

## 配置

所有门限和参数集中在 **`configs/pipeline.json`**，代码里不硬编码。

```jsonc
{
  "m1_track": {
    "min_person_height_px": 80,      // 检测框最小高度（低于此视为噪声）
    "lock": {
      "weight_duration": 0.20,        // 目标打分里「出现时长」的权重
      "sam2_drift_ratio": 3.0         // 漂移判据
    }
  },
  "m3f_fuse": {
    "reliability": {
      "time_continuity_ratio": 0.08,  // 时间连续性门限
      "two_model_agree_ratio": 0.05   // 两模型互证门限
    }
  }
}
```

几个常改的参数：

| 参数 | 默认 | 什么时候改 |
|---|---|---|
| `m1_track.min_person_height_px` | 80 | 素材里人特别小时调低 |
| `m1_track.lock.weight_duration` | 0.20 | 目标锁错人时调低（更看重「做动作」） |
| `m3f_fuse.reliability.two_model_agree_ratio` | 0.05 | 两个模型分歧大导致骨架稀疏时调大 |

### 环境变量

| 变量 | 作用 |
|---|---|
| `AIRTIME_MODELS` | 模型目录（默认 `./models`） |
| `AIRTIME_CONFIGS` | 配置目录（默认 `./configs`） |
| `AIRTIME_DATA` | 数据目录（默认 `./data`） |
| `AIRTIME_OUTPUT` | 输出目录（默认 `./runs`） |
| `AIRTIME_GEMX` | GEM-X 代码位置 |
| `AIRTIME_GEMX_PYTHON` | GEM-X 环境的 python 路径 |

---

## 设计取舍

**为什么用 GEM-X 而不是直接 2D 抬升？**
单目视频没有深度信息，纯 2D 抬升在转体 / 腾空时会出现严重的姿态歧义。
GEM-X 是在大规模人体数据上训练的三维重建模型，能给出合理的深度先验。

**为什么还要 2D 那一趟？**
3D 模型在远景、遮挡、运动模糊下会飘。2D 关键点在画面里更准。
融合阶段用 2D 观测去校正 3D 投影——两者互补。

**为什么融合要做骨长检查？**
逐关节校正可能把骨架拉到骨头长度明显不合理的地步（撕裂）。
超过容差就退回原始投影：宁可少校正，也不要撕裂。

**为什么去掉了鼻子对头顶的校正？**
Halpe26 的 `nose` 和 `head` 都曾被映射到 SOMA 的 `HeadEnd`。
实测 GEM-X 自己的 `HeadEnd` 观测距 `head`（头顶）4.69%、距 `nose` 14.27%——
`HeadEnd` 在解剖学上是头顶，不是鼻子。用鼻子去校正它会把头顶点拉到护目镜附近。

---

## 知道的问题

- **远景小目标**：人在画面里只有几十像素高时，2D 关键点噪声大，骨架会抖。
  这类帧会被标成 `track_only`，不进骨架。
- **快速转体**：转体速度超过 2D 模型跟踪能力时会丢帧。
- **整体对齐的稳健性**：融合的整体对齐在可靠点很少的帧上可能大幅晃动，
  会带动全身关节位移。这是已知待改进项。

---

## 项目结构

```
Airtime/
├── src/airtime/
│   ├── cfg.py              配置与路径解析
│   ├── run.py              一键入口
│   ├── m1_lock.py          M1 目标锁定打分
│   └── pipeline/
│       ├── m0_prep.py      预处理
│       ├── m1_track.py     跟踪
│       ├── m2_pose2d.py    2D 关键点
│       ├── m3_gemx.py      3D 重建
│       ├── m3_export.py    GEM-X 输出导出
│       ├── m3f_fuse.py     2D/3D 融合
│       └── m8_overlay.py   叠加可视化
├── configs/
│   ├── pipeline.json       主配置（门限都在这里）
│   ├── style.json          可视化样式
│   └── joint_names.json    SOMA 77 关节定义
├── models/                 模型权重（不入库）
├── data/                   输入视频（不入库）
└── runs/                   输出（不入库）
```

---

## 引用

如果用在本项目在研究/产品里，请一并引用底层模型：

- **GEM-X** — [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)，三维人体重建
- **RTMPose** — [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)，2D 关键点
- **SAM 2** — [facebookresearch/sam2](https://github.com/facebookresearch/sam2)，视频分割
- **rtmlib** — [open-mmlab/rtmlib](https://github.com/open-mmlab/rtmlib)，推理封装

---

## 许可证

[Apache-2.0](LICENSE)

注意：底层依赖各有自己的许可证——GEM-X、SAM 2、RTMPose 等，
商用前请分别确认。
