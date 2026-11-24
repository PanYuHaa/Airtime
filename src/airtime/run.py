#!/usr/bin/env python3
"""Airtime 一键入口：从一段滑雪视频生成骨架数据。

用法：
    python -m airtime.run input.mp4                    # 全流程
    python -m airtime.run input.mp4 --out runs/demo    # 指定输出
    python -m airtime.run input.mp4 --stage fuse       # 只跑到某个阶段

阶段：
    prep     M0  预处理（转码 / 旋转校正 / 镜头切分 / 统一 30fps 1080）
    track    M1  目标跟踪（YOLOX 检测 + SAM2 分割，锁定滑雪者）
    pose2d   M2  2D 关键点（RTMPose-x，Halpe26）
    gemx     M3  3D 人体重建（GEM-X，SOMA 77 关节）
    fuse     M3F 2D/3D 融合（得到最终骨架）
    overlay  M8  骨架叠加视频（可选，需要源视频）
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import cfg

STAGES = ["prep", "track", "pose2d", "gemx", "fuse", "overlay"]
PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"


def _run_stage(script: str, args: list[str]) -> int:
    """在子进程里跑一个阶段，返回退出码。"""
    cmd = [sys.executable, str(PIPELINE_DIR / script), *args]
    print(f"\n{'=' * 70}\n  {script}\n{'=' * 70}")
    return subprocess.call(cmd)


def _check_deps() -> list[str]:
    """检查运行依赖。"""
    missing = []
    for mod in ("cv2", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="airtime",
        description="Airtime — 从单段视频重建滑雪者 3D 骨架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("video", help="输入视频（mp4 / mov）")
    ap.add_argument("--out", default=str(cfg.OUTPUT_DIR / "default"),
                    help="输出目录")
    ap.add_argument("--stage", default="overlay",
                    choices=STAGES + ["all"],
                    help="跑到哪个阶段停止（默认 overlay = 全流程）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="推理设备")
    ap.add_argument("--no-overlay", action="store_true",
                    help="跳过叠加视频（更快，只要骨架数据时用）")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"错误：找不到视频 {video}", file=sys.stderr)
        return 1

    missing = _check_deps()
    if missing:
        print(f"错误：缺少依赖 {missing}，请先 pip install -e .", file=sys.stderr)
        return 1

    probs = cfg.check_setup(strict=False)
    if probs:
        print("配置有问题：", file=sys.stderr)
        for p in probs:
            print(f"  ✗ {p}", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stop = "overlay" if args.stage == "all" else args.stage
    if args.no_overlay and stop == "overlay":
        stop = "fuse"
    upto = STAGES.index(stop) + 1
    todo = STAGES[:upto]

    print(f"输入视频: {video}")
    print(f"输出目录: {out}")
    print(f"执行阶段: {' -> '.join(todo)}")

    t0 = time.time()
    pkg = out

    if "prep" in todo:
        rc = _run_stage("m0_prep.py", ["--videos", str(video.parent),
                                       "--out", str(pkg), "--only", video.stem])
        if rc != 0:
            return rc

    if "track" in todo:
        rc = _run_stage("m1_track.py", ["--pkg", str(pkg), "--out", str(pkg)])
        if rc != 0:
            return rc

    if "pose2d" in todo:
        rc = _run_stage("m2_pose2d.py", ["--track", str(pkg),
                                         "--out", str(pkg),
                                         "--device", args.device])
        if rc != 0:
            return rc

    if "gemx" in todo:
        rc = _run_stage("m3_gemx.py", ["--pkg", str(pkg), "--out", str(pkg)])
        if rc != 0:
            return rc

    if "fuse" in todo:
        rc = _run_stage("m3f_fuse.py", ["--out", str(pkg)])
        if rc != 0:
            return rc

    if "overlay" in todo:
        rc = _run_stage("m8_overlay.py", ["--pkg", str(pkg), "--out", str(pkg)])
        if rc != 0:
            return rc

    dt = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ✓ 完成，用时 {dt:.1f}s")
    print(f"  输出: {pkg}")
    skel = pkg / "skeleton"
    if skel.is_dir():
        js = sorted(skel.glob("*.json"))
        print(f"  骨架文件: {len(js)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
