"""Airtime 配置契约、M1 锁人工具与基准评测模块的确定性回归单测。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from airtime import cfg, m1_lock
from airtime import eval as airtime_eval


def test_pipeline_config_and_soma_joints_contract() -> None:
    p = cfg.load_pipeline_config()
    for section in ("m0_preprocess", "m1_track", "m2_pose2d", "m3_gemx", "m3f_fusion"):
        assert section in p, f"configs/pipeline.json 缺少配置段: {section}"

    joints = json.loads(cfg.SOMA_JOINT_NAMES.read_text(encoding="utf-8"))
    assert isinstance(joints, list) and len(joints) == 77
    assert "Hips" in joints and "HeadEnd" in joints


def test_resolve_sam2_config_variants() -> None:
    assert cfg.resolve_sam2_config(Path("sam2_hiera_large.pt")).endswith("_l.yaml")
    assert cfg.resolve_sam2_config(Path("sam2.1_hiera_small.pt")).endswith("_s.yaml")
    assert cfg.resolve_sam2_config(Path("sam2.1_hiera_tiny.pt")).endswith("_t.yaml")
    assert cfg.resolve_sam2_config(Path("sam2.1_hiera_base_plus.pt")).endswith("_b+.yaml")


def test_m1_lock_iou_and_build_tracks() -> None:
    assert m1_lock.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0, rel=1e-6)
    assert m1_lock.iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0, abs=1e-6)

    # 模拟一个匀速平移目标，连续 5 帧应被串成 1 条轨迹
    all_dets = [[[100 + 10 * k, 200, 180 + 10 * k, 400]] for k in range(5)]
    tracks = m1_lock.build_tracks(all_dets)
    assert len(tracks) == 1
    assert len(tracks[0]["frames"]) == 5


def test_compute_skeleton_jitter_zero_for_linear_motion() -> None:
    # 匀速直线运动：二阶差分应严格为 0
    frames_data = [
        {
            "i": k,
            "t": round(k / 30.0, 3),
            "status": "skeleton",
            "joints": {"Hips": [100.0 + 5.0 * k, 200.0 + 2.0 * k]},
        }
        for k in range(6)
    ]
    tracking = [{"i": k, "box_h": 200.0} for k in range(6)]
    p50, p95, n = airtime_eval.compute_skeleton_jitter(frames_data, tracking=tracking)
    assert n == 4
    assert p50 == 0.0
    assert p95 == 0.0


def test_compute_skeleton_jitter_detects_oscillation() -> None:
    # 往复抖动 10px，box_h=200px：每帧二阶差分 = |(p[t+1]-p[t]) - (p[t]-p[t-1])| = 20px -> 10% box_h
    xs = [100.0, 110.0, 100.0, 110.0, 100.0]
    frames_data = [
        {"i": k, "t": k / 30.0, "status": "skeleton", "joints": {"Hips": [x, 200.0]}}
        for k, x in enumerate(xs)
    ]
    tracking = [{"i": k, "box_h": 200.0} for k in range(len(xs))]
    p50, p95, n = airtime_eval.compute_skeleton_jitter(frames_data, tracking=tracking)
    assert n == 3
    assert p50 == pytest.approx(10.0, abs=0.01)
    assert p95 == pytest.approx(10.0, abs=0.01)


def _write_fake_run(
    root: Path,
    vid: str,
    *,
    skel_count: int,
    jitter_amp: float,
    bad_proj: int,
    m3_sec: float,
) -> None:
    (root / "skeleton").mkdir(parents=True, exist_ok=True)
    (root / "tracking").mkdir(parents=True, exist_ok=True)
    total = 10
    frames_data = []
    tracking = []
    for k in range(total):
        status = "skeleton" if k < skel_count else "none"
        x = 100.0 + (jitter_amp if k % 2 else 0.0)
        frames_data.append(
            {
                "i": k,
                "t": round(k / 30.0, 3),
                "status": status,
                "joints": {"Hips": [x, 200.0]} if status == "skeleton" else None,
            }
        )
        tracking.append({"i": k, "box_h": 200.0, "status": status})

    (root / "skeleton" / f"{vid}.json").write_text(
        json.dumps({"video": vid, "frames": total, "frames_data": frames_data}),
        encoding="utf-8",
    )
    (root / "tracking" / f"{vid}.json").write_text(json.dumps(tracking), encoding="utf-8")
    (root / "index.json").write_text(json.dumps({vid: {"frames": total, "sec": 1.0}}), encoding="utf-8")
    (root / "pose2d_stats.json").write_text(json.dumps({vid: {"mean_conf": 0.8, "sec": 2.0}}), encoding="utf-8")
    (root / "gem_stats.json").write_text(json.dumps({vid: {"sec": m3_sec}}), encoding="utf-8")
    (root / "fusion_stats.json").write_text(
        json.dumps({vid: {"fused_ratio_fusable": 45.0, "bone_rollback": 5, "n_bad_projection": bad_proj, "sec": 0.1}}),
        encoding="utf-8",
    )


def test_evaluate_and_compare_runs_flags_regression(tmp_path: Path) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    _write_fake_run(before_dir, "clip_a", skel_count=10, jitter_amp=1.0, bad_proj=0, m3_sec=20.0)
    _write_fake_run(after_dir, "clip_a", skel_count=8, jitter_amp=6.0, bad_proj=2, m3_sec=15.0)

    rep_b = airtime_eval.evaluate_run(before_dir)
    rep_a = airtime_eval.evaluate_run(after_dir)
    cmp = airtime_eval.compare_runs(rep_b, rep_a)

    assert cmp["common_videos"] == ["clip_a"]
    assert cmp["per_video"]["clip_a"]["delta"]["skeleton_ratio_pct"] == -20.0
    assert cmp["per_video"]["clip_a"]["delta"]["n_bad_projection"] == 2
    assert len(cmp["regressions"]) == 3  # 覆盖下降 + 异常投影增加 + 抖动上升

    md = airtime_eval.format_compare_markdown(cmp)
    assert "clip_a" in md and "⚠" in md


def _write_dummy_overlay(path: Path, n_frames: int = 8, color: tuple[int, int, int] = (40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for k in range(n_frames):
        frame = np.full((120, 160, 3), color, dtype=np.uint8)
        cv2.putText(frame, str(k), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_render_visual_and_compare_sheets(tmp_path: Path) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    _write_fake_run(before_dir, "clip_a", skel_count=8, jitter_amp=1.0, bad_proj=0, m3_sec=10.0)
    _write_fake_run(after_dir, "clip_a", skel_count=8, jitter_amp=1.0, bad_proj=0, m3_sec=10.0)
    _write_dummy_overlay(before_dir / "render" / "clip_a" / "overlay.mp4", n_frames=8, color=(30, 60, 120))
    _write_dummy_overlay(after_dir / "render" / "clip_a" / "overlay.mp4", n_frames=8, color=(30, 120, 60))

    single = airtime_eval.render_visual_sheet(before_dir, "clip_a", tmp_path / "single.jpg", n_frames=4, cell_height=120)
    assert single is not None and single.is_file()
    img_s = cv2.imdecode(np.frombuffer(single.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img_s is not None and img_s.shape[0] == 120 and img_s.shape[1] == 160 * 4

    cmp_img = airtime_eval.render_compare_sheet(
        before_dir, after_dir, "clip_a", tmp_path / "cmp.jpg", n_frames=4, cell_height=120
    )
    assert cmp_img is not None and cmp_img.is_file()
    img_c = cv2.imdecode(np.frombuffer(cmp_img.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img_c is not None and img_c.shape[0] == 120 * 2 and img_c.shape[1] == 160 * 4
