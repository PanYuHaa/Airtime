"""Airtime 精确时序契约（PTS / VFR 检测 / 慢放倍率 / 物理时间换算）单测。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from airtime import time_contract as tc


def test_cfr_unconfirmed_slowmo_blocks_fake_real_time() -> None:
    # 60fps 源视频 60 帧（1 秒），转码 30fps 30 帧，未指定 slowmo_factor
    src_tl = {
        "source_frames": 60,
        "source_time_base": "1/60000",
        "source_nominal_fps": 60.0,
        "source_avg_fps": 60.0,
        "is_vfr": False,
        "pts": [k * 1000 for k in range(60)],
        "pts_time_s": [round(k / 60.0, 6) for k in range(60)],
    }
    c = tc.build_time_contract(src_tl, clip_frames=30, clip_fps=30.0, slowmo_factor=None)
    assert c["summary"]["real_time_confirmed"] is False
    assert c["summary"]["slowmo_factor"] is None
    assert c["frames"][15]["source_frame_idx"] == 30
    assert c["frames"][15]["real_time_s"] is None

    dur = tc.span_duration(c, 0, 29)
    assert dur["playback_duration_s"] == pytest.approx(29 / 30.0, abs=1e-5)
    assert dur["real_duration_s"] is None


def test_slowmo_4x_computes_accurate_physical_airtime() -> None:
    # 4 倍慢放视频：播放滞空 72 帧（2.40s @ 30fps），真实物理滞空应为 0.60s
    n = 90
    src_tl = {
        "source_frames": n,
        "source_time_base": "1/30000",
        "source_nominal_fps": 30.0,
        "source_avg_fps": 30.0,
        "is_vfr": False,
        "pts": [k * 1000 for k in range(n)],
        "pts_time_s": [round(k / 30.0, 6) for k in range(n)],
    }
    c = tc.build_time_contract(src_tl, clip_frames=n, clip_fps=30.0, slowmo_factor=4.0)
    assert c["summary"]["real_time_confirmed"] is True
    assert c["summary"]["effective_capture_fps"] == 120.0

    dur = tc.span_duration(c, 10, 82)  # 72 帧间隔 = 2.40s 播放时长
    assert dur["playback_duration_s"] == pytest.approx(2.40, abs=1e-5)
    assert dur["real_duration_s"] == pytest.approx(0.60, abs=1e-5)


def test_vfr_detection_and_nearest_source_frame_mapping() -> None:
    # 构造可变帧率 PTS（前段 ~30fps，中段掉帧跳变 0.12s，后段 ~60fps）
    pts_sec = [0.0, 0.033, 0.067, 0.100, 0.220, 0.236, 0.253, 0.270]
    assert tc.detect_vfr(pts_sec) is True
    assert tc.detect_vfr([k / 30.0 for k in range(10)]) is False

    src_tl = {
        "source_frames": len(pts_sec),
        "source_time_base": "1/90000",
        "source_nominal_fps": 30.0,
        "source_avg_fps": 29.6,
        "is_vfr": True,
        "pts": [int(t * 90000) for t in pts_sec],
        "pts_time_s": pts_sec,
    }
    c = tc.build_time_contract(src_tl, clip_frames=9, clip_fps=30.0, slowmo_factor=1.0)
    # k=7 -> t_play = 7/30 = 0.2333s，离 pts_sec[5]=0.236s 最近
    assert c["frames"][7]["source_frame_idx"] == 5
    assert c["frames"][7]["source_pts_s"] == pytest.approx(0.236, abs=1e-6)


def test_probe_source_timeline_on_real_mp4(tmp_path: Path) -> None:
    mp4 = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    for k in range(12):
        writer.write(np.full((64, 64, 3), k * 10, dtype=np.uint8))
    writer.release()

    tl = tc.probe_source_timeline(mp4)
    assert tl["source_frames"] == 12
    assert len(tl["pts_time_s"]) == 12
    assert tl["is_vfr"] is False
