"""Airtime 全局配置与路径解析。

设计原则：**仓库里没有任何硬编码的绝对路径**。
所有路径按下面优先级解析：

  1. 环境变量（AIRTIME_MODELS / AIRTIME_CONFIGS / AIRTIME_DATA / AIRTIME_OUTPUT）
  2. 仓库内默认位置（models/ configs/ data/ runs/）
  3. 找不到时报错，并打印明确的修复指引

所以 `git clone` + 把权重放进 models/ 就能直接跑。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 仓库根目录：本文件在 src/airtime/cfg.py → 上溯两级
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _dir(env: str, default: Path) -> Path:
    """读环境变量指定的路径，未设置则用默认值。"""
    v = os.environ.get(env)
    return Path(v).expanduser().resolve() if v else default


# ------------------------------------------------------------------ 目录
MODELS_DIR = _dir("AIRTIME_MODELS", REPO_ROOT / "models")
CONFIGS_DIR = _dir("AIRTIME_CONFIGS", REPO_ROOT / "configs")
DATA_DIR = _dir("AIRTIME_DATA", REPO_ROOT / "data")
OUTPUT_DIR = _dir("AIRTIME_OUTPUT", REPO_ROOT / "runs")


# ------------------------------------------------------------------ 权重
# 2D 人体检测（YOLOX，rtmlib 使用）
YOLOX_ONNX = _dir(
    "AIRTIME_YOLOX",
    MODELS_DIR / "yolox_x_8xb8-300e_humanart-a39d44ed.onnx",
)
# 2D 关键点（RTMPose-x，Halpe26）
RTMPOSE_ONNX = _dir(
    "AIRTIME_RTMPOSE",
    MODELS_DIR / "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288.onnx",
)
# SAM2（M1 跟踪用的视频分割）
SAM2_DIR = _dir("AIRTIME_SAM2", MODELS_DIR / "sam2")
SAM2_CKPT = _dir("AIRTIME_SAM2_CKPT", MODELS_DIR / "sam2_hiera_large.pt")
# GEM-X（M3 三维人体重建）
GEMX_DIR = _dir("AIRTIME_GEMX", MODELS_DIR / "GEM-X")
GEMX_CKPT = _dir(
    "AIRTIME_GEMX_CKPT",
    MODELS_DIR / "GEM-X" / "checkpoints" / "gem_x.ckpt",
)

# 跑 GEM-X 的独立解释器（需要 torch + CUDA，与 2D 那套依赖分开装）
GEMX_PYTHON = os.environ.get("AIRTIME_GEMX_PYTHON", "")
# 跑 SAM2 的独立解释器（缺省复用 GEMX_PYTHON）
SAM2_PYTHON = os.environ.get("AIRTIME_SAM2_PYTHON", "") or GEMX_PYTHON


def resolve_sam2_config(ckpt: Path) -> str:
    """根据 SAM2 权重文件名自动匹配对应的 Hydra 配置路径。"""
    override = os.environ.get("AIRTIME_SAM2_CONFIG", "").strip()
    if override:
        return override
    name = ckpt.name.lower()
    is_v21 = (SAM2_DIR / "sam2" / "configs" / "sam2.1").is_dir() or "2.1" in name
    prefix = "configs/sam2.1/sam2.1_hiera_" if is_v21 else "configs/sam2/sam2_hiera_"
    if "large" in name or name.endswith("_l.pt"):
        suffix = "l.yaml"
    elif "base" in name or "b+" in name:
        suffix = "b+.yaml"
    elif "tiny" in name or name.endswith("_t.pt"):
        suffix = "t.yaml"
    else:
        suffix = "s.yaml"
    return prefix + suffix


# ------------------------------------------------------------------ 配置
def config_path(name: str) -> Path:
    """取 configs/ 下的文件路径，不存在则报错并给出指引。"""
    p = CONFIGS_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"配置文件缺失: {p}\n"
            f"  · 确认仓库 configs/ 目录完整\n"
            f"  · 或设置环境变量 AIRTIME_CONFIGS 指向配置目录"
        )
    return p


# SOMA 77 关节名表（GEM-X 的关节定义，随仓库分发，不属模型权重）
SOMA_JOINT_NAMES = CONFIGS_DIR / "joint_names.json"


def load_config(name: str) -> dict:
    """读 configs/ 下的 JSON 配置。"""
    return json.loads(config_path(name).read_text(encoding="utf-8"))


def load_pipeline_config() -> dict:
    """主流水线配置：所有阶段的门限都在这里。"""
    return load_config("pipeline.json")


def load_videos_config() -> dict:
    """视频清单配置。"""
    return load_config("videos.json")


# ------------------------------------------------------------------ 自检
def check_setup(strict: bool = False) -> list[str]:
    """返回问题列表，空表示就绪。strict=True 时连模型权重一起检查。"""
    problems: list[str] = []

    if not CONFIGS_DIR.is_dir():
        problems.append(f"configs 目录不存在: {CONFIGS_DIR}")
    try:
        config_path("pipeline.json")
    except FileNotFoundError as e:
        problems.append(str(e))

    if strict:
        for nm, p in (
            ("YOLOX 检测模型", YOLOX_ONNX),
            ("RTMPose 关键点模型", RTMPOSE_ONNX),
            ("SAM2 目录", SAM2_DIR),
            ("GEM-X 目录", GEMX_DIR),
            ("SOMA 关节名表", SOMA_JOINT_NAMES),
        ):
            if not p.exists():
                problems.append(f"{nm} 不存在: {p}")
    return problems


def main() -> int:
    print(f"仓库根目录: {REPO_ROOT}")
    print(f"配置目录  : {CONFIGS_DIR}")
    print(f"模型目录  : {MODELS_DIR}")
    print(f"数据目录  : {DATA_DIR}")
    print(f"输出目录  : {OUTPUT_DIR}")
    print()
    probs = check_setup(strict="--strict" in sys.argv)
    if probs:
        print("发现问题：")
        for p in probs:
            print(f"  ✗ {p}")
        return 1
    print("✓ 环境自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
