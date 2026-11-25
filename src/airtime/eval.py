#!/usr/bin/env python3
"""Airtime 基准评测与改动前后对比工具。

用于量化单次运行质量，或在优化类 PR 中生成「改动前 vs 改动后」Markdown 对比表。

用法：
    # 汇总单次运行目录
    python -m airtime.eval runs/smoke_cli_fix

    # 对比两次运行（改动前 vs 改动后）
    python -m airtime.eval --before runs/base --after runs/exp

    # 导出 Markdown / JSON 报告
    python -m airtime.eval --before runs/base --after runs/exp --md runs/compare.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_skeleton_jitter(
    frames_data: list[dict[str, Any]],
    tracking: list[dict[str, Any]] | None = None,
    fallback_height_px: float | None = None,
) -> tuple[float | None, float | None, int]:
    """计算骨架时间抖动（二阶差分加速度残差，归一化为 %box_h）。

    仅在连续三帧 (t-1, t, t+1) 均为 ``status == "skeleton"`` 且同一关节坐标均有效时计算：
        ``|| (p[t+1] - p[t]) - (p[t] - p[t-1]) || / box_h[t] * 100``

    返回 ``(p50_pct, p95_pct, sample_count)``；无有效样本时返回 ``(None, None, 0)``。
    """
    box_h_by_idx: dict[int, float] = {}
    if isinstance(tracking, list):
        for item in tracking:
            idx = item.get("i")
            bh = item.get("box_h")
            if idx is not None and bh and bh > 1.0:
                box_h_by_idx[int(idx)] = float(bh)

    by_idx: dict[int, dict[str, Any]] = {}
    for frame in frames_data:
        idx = frame.get("i")
        if idx is not None:
            by_idx[int(idx)] = frame

    residuals: list[float] = []
    for idx in sorted(by_idx):
        f_prev = by_idx.get(idx - 1)
        f_curr = by_idx.get(idx)
        f_next = by_idx.get(idx + 1)
        if not (f_prev and f_curr and f_next):
            continue
        if (
            f_prev.get("status") != "skeleton"
            or f_curr.get("status") != "skeleton"
            or f_next.get("status") != "skeleton"
        ):
            continue

        j_prev = f_prev.get("joints")
        j_curr = f_curr.get("joints")
        j_next = f_next.get("joints")
        if not (isinstance(j_prev, dict) and isinstance(j_curr, dict) and isinstance(j_next, dict)):
            continue

        scale = box_h_by_idx.get(idx) or fallback_height_px
        if not scale or scale <= 1.0:
            continue

        for name, p_curr in j_curr.items():
            p_prev = j_prev.get(name)
            p_next = j_next.get(name)
            if not (
                isinstance(p_prev, (list, tuple))
                and isinstance(p_curr, (list, tuple))
                and isinstance(p_next, (list, tuple))
                and len(p_prev) >= 2
                and len(p_curr) >= 2
                and len(p_next) >= 2
            ):
                continue
            dx = (float(p_next[0]) - float(p_curr[0])) - (float(p_curr[0]) - float(p_prev[0]))
            dy = (float(p_next[1]) - float(p_curr[1])) - (float(p_curr[1]) - float(p_prev[1]))
            if math.isfinite(dx) and math.isfinite(dy):
                residuals.append(math.hypot(dx, dy) / scale * 100.0)

    if not residuals:
        return None, None, 0

    arr = np.asarray(residuals, dtype=np.float64)
    return round(float(np.percentile(arr, 50)), 2), round(float(np.percentile(arr, 95)), 2), len(residuals)


def _discover_videos(run_dir: Path) -> list[str]:
    vids: set[str] = set()
    for stat_name in (
        "index.json",
        "tracking_stats.json",
        "pose2d_stats.json",
        "gem_stats.json",
        "fusion_stats.json",
        "render_stats.json",
    ):
        doc = _read_json(run_dir / stat_name)
        if isinstance(doc, dict):
            vids.update(doc.keys())

    skel_dir = run_dir / "skeleton"
    if skel_dir.is_dir():
        for p in skel_dir.glob("*.json"):
            vids.add(p.stem)
    return sorted(vids)


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """从一次 Airtime 输出目录汇总各视频指标与全局汇总。"""
    run_dir = Path(run_dir).resolve()
    index_doc = _read_json(run_dir / "index.json") or {}
    track_stats = _read_json(run_dir / "tracking_stats.json") or {}
    pose2d_stats = _read_json(run_dir / "pose2d_stats.json") or {}
    gem_stats = _read_json(run_dir / "gem_stats.json") or {}
    fusion_stats = _read_json(run_dir / "fusion_stats.json") or {}
    render_stats = _read_json(run_dir / "render_stats.json") or {}

    videos: dict[str, dict[str, Any]] = {}
    for vid in _discover_videos(run_dir):
        idx_m = index_doc.get(vid, {}) if isinstance(index_doc, dict) else {}
        trk_m = track_stats.get(vid, {}) if isinstance(track_stats, dict) else {}
        p2d_m = pose2d_stats.get(vid, {}) if isinstance(pose2d_stats, dict) else {}
        gem_m = gem_stats.get(vid, {}) if isinstance(gem_stats, dict) else {}
        fus_m = fusion_stats.get(vid, {}) if isinstance(fusion_stats, dict) else {}
        rnd_m = render_stats.get(vid, {}) if isinstance(render_stats, dict) else {}

        skel_doc = _read_json(run_dir / "skeleton" / f"{vid}.json")
        trk_doc = _read_json(run_dir / "tracking" / f"{vid}.json")

        frames_data = skel_doc.get("frames_data", []) if isinstance(skel_doc, dict) else []
        frames = (
            int(skel_doc["frames"])
            if isinstance(skel_doc, dict) and skel_doc.get("frames") is not None
            else int(trk_m.get("frames") or idx_m.get("frames") or len(frames_data))
        )

        if frames_data:
            skel_frames = sum(1 for f in frames_data if f.get("status") == "skeleton")
            track_only_frames = sum(1 for f in frames_data if f.get("status") == "track_only")
            none_frames = sum(1 for f in frames_data if f.get("status") == "none")
        else:
            skel_frames = int(fus_m.get("skeleton_frames") or trk_m.get("skeleton") or 0)
            track_only_frames = int(trk_m.get("track_only") or 0)
            none_frames = int(trk_m.get("none") or 0)

        fallback_h = skel_doc.get("body_height_px_median") if isinstance(skel_doc, dict) else None
        jitter_p50, jitter_p95, jitter_samples = compute_skeleton_jitter(
            frames_data, tracking=trk_doc, fallback_height_px=fallback_h
        )

        stage_sec = {
            "m0": idx_m.get("sec"),
            "m1": trk_m.get("sec"),
            "m2": p2d_m.get("sec"),
            "m3": gem_m.get("sec"),
            "m3f": fus_m.get("sec"),
            "m8": rnd_m.get("sec"),
        }
        valid_secs = [float(v) for v in stage_sec.values() if v is not None]
        total_sec = round(sum(valid_secs), 1) if valid_secs else None

        skel_ratio = round(skel_frames / frames * 100.0, 1) if frames > 0 else 0.0
        videos[vid] = {
            "frames": frames,
            "skeleton_frames": skel_frames,
            "skeleton_ratio_pct": skel_ratio,
            "track_only_frames": track_only_frames,
            "none_frames": none_frames,
            "pose2d_mean_conf": p2d_m.get("mean_conf"),
            "fused_ratio_pct": fus_m.get("fused_ratio_fusable"),
            "bone_rollback": fus_m.get("bone_rollback"),
            "n_bad_projection": fus_m.get("n_bad_projection"),
            "jitter_p50_pct": jitter_p50,
            "jitter_p95_pct": jitter_p95,
            "jitter_samples": jitter_samples,
            "stage_sec": stage_sec,
            "total_sec": total_sec,
        }

    total_frames = sum(v["frames"] for v in videos.values())
    total_skel = sum(v["skeleton_frames"] for v in videos.values())
    p50s = [v["jitter_p50_pct"] for v in videos.values() if v["jitter_p50_pct"] is not None]
    p95s = [v["jitter_p95_pct"] for v in videos.values() if v["jitter_p95_pct"] is not None]
    confs = [v["pose2d_mean_conf"] for v in videos.values() if v["pose2d_mean_conf"] is not None]
    fuseds = [v["fused_ratio_pct"] for v in videos.values() if v["fused_ratio_pct"] is not None]
    secs = [v["total_sec"] for v in videos.values() if v["total_sec"] is not None]

    summary = {
        "video_count": len(videos),
        "total_frames": total_frames,
        "total_skeleton_frames": total_skel,
        "skeleton_ratio_pct": round(total_skel / total_frames * 100.0, 1) if total_frames > 0 else 0.0,
        "mean_pose2d_conf": round(float(np.mean(confs)), 4) if confs else None,
        "mean_fused_ratio_pct": round(float(np.mean(fuseds)), 1) if fuseds else None,
        "total_bone_rollback": sum(int(v["bone_rollback"] or 0) for v in videos.values()),
        "total_bad_projection": sum(int(v["n_bad_projection"] or 0) for v in videos.values()),
        "mean_jitter_p50_pct": round(float(np.mean(p50s)), 2) if p50s else None,
        "mean_jitter_p95_pct": round(float(np.mean(p95s)), 2) if p95s else None,
        "total_sec": round(sum(secs), 1) if secs else None,
    }
    return {"run_dir": str(run_dir), "summary": summary, "videos": videos}


def compare_runs(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """对比两次评测结果，计算差值并标出潜在退化项。"""
    common = sorted(set(before["videos"]).intersection(after["videos"]))
    only_before = sorted(set(before["videos"]) - set(after["videos"]))
    only_after = sorted(set(after["videos"]) - set(before["videos"]))

    per_video: dict[str, dict[str, Any]] = {}
    regressions: list[str] = []

    def _delta(a: float | None, b: float | None, ndigits: int = 2) -> float | None:
        if a is None or b is None:
            return None
        return round(float(b) - float(a), ndigits)

    for vid in common:
        vb = before["videos"][vid]
        va = after["videos"][vid]
        d_skel_ratio = _delta(vb["skeleton_ratio_pct"], va["skeleton_ratio_pct"], 1)
        d_p50 = _delta(vb["jitter_p50_pct"], va["jitter_p50_pct"], 2)
        d_p95 = _delta(vb["jitter_p95_pct"], va["jitter_p95_pct"], 2)
        d_fused = _delta(vb["fused_ratio_pct"], va["fused_ratio_pct"], 1)
        d_bad = _delta(vb["n_bad_projection"], va["n_bad_projection"], 0)
        d_sec = _delta(vb["total_sec"], va["total_sec"], 1)

        flags: list[str] = []
        if d_skel_ratio is not None and d_skel_ratio < -1.0:
            flags.append(f"骨架覆盖下降 {d_skel_ratio:+.1f}%")
        if d_bad is not None and d_bad > 0:
            flags.append(f"异常投影增加 +{int(d_bad)}")
        if d_p95 is not None and d_p95 > 0.5:
            flags.append(f"抖动p95上升 {d_p95:+.2f}%")
        for f in flags:
            regressions.append(f"{vid}: {f}")

        per_video[vid] = {
            "before": vb,
            "after": va,
            "delta": {
                "skeleton_ratio_pct": d_skel_ratio,
                "jitter_p50_pct": d_p50,
                "jitter_p95_pct": d_p95,
                "fused_ratio_pct": d_fused,
                "n_bad_projection": int(d_bad) if d_bad is not None else None,
                "total_sec": d_sec,
            },
            "flags": flags,
        }

    return {
        "before_dir": before["run_dir"],
        "after_dir": after["run_dir"],
        "common_videos": common,
        "only_before": only_before,
        "only_after": only_after,
        "per_video": per_video,
        "regressions": regressions,
    }


def _fmt(val: Any, suffix: str = "", signed: bool = False) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        sign = "+" if signed and val > 0 else ""
        return f"{sign}{val:.2f}{suffix}" if abs(val) < 10 and suffix == "%" else f"{sign}{val:g}{suffix}"
    if isinstance(val, int) and signed and val > 0:
        return f"+{val}{suffix}"
    return f"{val}{suffix}"


def format_single_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "### Airtime 基准评测报告",
        "",
        f"- 视频数：`{s['video_count']}` | 总骨架覆盖：`{s['total_skeleton_frames']}/{s['total_frames']}` (`{s['skeleton_ratio_pct']}%`)",
        f"- 抖动均值（%box_h）：`p50={_fmt(s['mean_jitter_p50_pct'], '%')}` / `p95={_fmt(s['mean_jitter_p95_pct'], '%')}` | 异常投影：`{s['total_bad_projection']}` | 总耗时：`{_fmt(s['total_sec'], 's')}`",
        "",
        "| 视频 | 有效骨架 | 2D置信度 | 融合率 | 骨长回退 | 异常投影 | 抖动 p50 | 抖动 p95 | 总耗时 |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for vid, v in report["videos"].items():
        lines.append(
            f"| `{vid}` | {v['skeleton_frames']}/{v['frames']} ({v['skeleton_ratio_pct']}%) "
            f"| {_fmt(v['pose2d_mean_conf'])} | {_fmt(v['fused_ratio_pct'], '%')} "
            f"| {_fmt(v['bone_rollback'])} | {_fmt(v['n_bad_projection'])} "
            f"| {_fmt(v['jitter_p50_pct'], '%')} | {_fmt(v['jitter_p95_pct'], '%')} "
            f"| {_fmt(v['total_sec'], 's')} |"
        )
    return "\n".join(lines)


def format_compare_markdown(cmp: dict[str, Any]) -> str:
    lines = [
        "### 改动前 vs 改动后基准对比",
        "",
        "| 视频 | 骨架覆盖（前 → 后） | 抖动 p50（%box_h） | 抖动 p95（%box_h） | 融合率 | 异常投影 | 总耗时 | 状态 |",
        "| :-- | --: | --: | --: | --: | --: | --: | :-- |",
    ]
    for vid in cmp["common_videos"]:
        item = cmp["per_video"][vid]
        vb, va, d = item["before"], item["after"], item["delta"]
        status = "⚠ " + "；".join(item["flags"]) if item["flags"] else "✅ 正常"
        lines.append(
            f"| `{vid}` "
            f"| {vb['skeleton_ratio_pct']}% → {va['skeleton_ratio_pct']}% ({_fmt(d['skeleton_ratio_pct'], '%', True)}) "
            f"| {_fmt(vb['jitter_p50_pct'], '%')} → {_fmt(va['jitter_p50_pct'], '%')} ({_fmt(d['jitter_p50_pct'], '%', True)}) "
            f"| {_fmt(vb['jitter_p95_pct'], '%')} → {_fmt(va['jitter_p95_pct'], '%')} ({_fmt(d['jitter_p95_pct'], '%', True)}) "
            f"| {_fmt(vb['fused_ratio_pct'], '%')} → {_fmt(va['fused_ratio_pct'], '%')} "
            f"| {_fmt(vb['n_bad_projection'])} → {_fmt(va['n_bad_projection'])} "
            f"| {_fmt(vb['total_sec'], 's')} → {_fmt(va['total_sec'], 's')} "
            f"| {status} |"
        )
    if cmp["regressions"]:
        lines.append("")
        lines.append("**⚠ 检测到潜在退化项**：")
        for r in cmp["regressions"]:
            lines.append(f"- {r}")
    return "\n".join(lines)


def _overlay_video_path(run_dir: Path, vid: str) -> Path | None:
    """查找 M8 生成的叠加视频路径。"""
    for cand in (
        run_dir / "render" / vid / "overlay.mp4",
        run_dir / "overlay" / f"{vid}.mp4",
    ):
        if cand.is_file():
            return cand
    return None


def _sample_video_frames(
    video_path: Path,
    frame_indices: list[int],
    cell_height: int = 360,
) -> list[tuple[int, np.ndarray]]:
    """按指定帧号从视频中抽帧并等比缩放到统一高度。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: list[tuple[int, np.ndarray]] = []
    try:
        for raw_idx in frame_indices:
            idx = max(0, min(int(raw_idx), max(total - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            scale = cell_height / max(h, 1)
            resized = cv2.resize(
                frame,
                (max(1, round(w * scale)), cell_height),
                interpolation=cv2.INTER_AREA,
            )
            out.append((idx, resized))
    finally:
        cap.release()
    return out


def _choose_frame_indices(
    frames_data: list[dict[str, Any]],
    total_frames: int,
    n_frames: int = 6,
) -> list[int]:
    """优先在有效骨架帧区间内等间隔抽帧，否则在整段视频内等间隔抽帧。"""
    skel_idxs = [int(f["i"]) for f in frames_data if f.get("status") == "skeleton" and f.get("i") is not None]
    pool = skel_idxs if len(skel_idxs) >= 2 else list(range(max(total_frames, 1)))
    if not pool:
        return []
    k = max(1, min(n_frames, len(pool)))
    pos = np.linspace(0, len(pool) - 1, k)
    return [pool[round(float(p))] for p in pos]


def _annotate_cell(img: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def _save_jpeg(path: Path, img: np.ndarray, quality: int = 90) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def render_visual_sheet(
    run_dir: Path,
    vid: str,
    out_path: Path,
    n_frames: int = 6,
    cell_height: int = 360,
) -> Path | None:
    """为单次运行生成一行等间隔关键帧目检拼图。"""
    vpath = _overlay_video_path(run_dir, vid)
    if vpath is None:
        return None
    skel_doc = _read_json(run_dir / "skeleton" / f"{vid}.json") or {}
    frames_data = skel_doc.get("frames_data", []) if isinstance(skel_doc, dict) else []
    total = int(skel_doc.get("frames") or 0)
    if total <= 0:
        cap = cv2.VideoCapture(str(vpath))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    indices = _choose_frame_indices(frames_data, total, n_frames=n_frames)
    sampled = _sample_video_frames(vpath, indices, cell_height=cell_height)
    if not sampled:
        return None
    cells = [_annotate_cell(im, f"f={idx}", (255, 255, 255)) for idx, im in sampled]
    sheet = np.hstack(cells)
    return out_path if _save_jpeg(out_path, sheet) else None


def render_compare_sheet(
    before_dir: Path,
    after_dir: Path,
    vid: str,
    out_path: Path,
    n_frames: int = 6,
    cell_height: int = 320,
) -> Path | None:
    """为改动前 vs 改动后生成上下双排同帧目检对比拼图（上=Before，下=After）。"""
    vb = _overlay_video_path(before_dir, vid)
    va = _overlay_video_path(after_dir, vid)
    if vb is None or va is None:
        return None

    skel_doc = _read_json(after_dir / "skeleton" / f"{vid}.json") or _read_json(
        before_dir / "skeleton" / f"{vid}.json"
    ) or {}
    frames_data = skel_doc.get("frames_data", []) if isinstance(skel_doc, dict) else []
    total = int(skel_doc.get("frames") or 0)
    if total <= 0:
        cap = cv2.VideoCapture(str(va))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    indices = _choose_frame_indices(frames_data, total, n_frames=n_frames)
    sb = _sample_video_frames(vb, indices, cell_height=cell_height)
    sa = _sample_video_frames(va, indices, cell_height=cell_height)
    n = min(len(sb), len(sa))
    if n == 0:
        return None

    row_b = [_annotate_cell(sb[k][1], f"Before f={sb[k][0]}", (180, 220, 255)) for k in range(n)]
    row_a = [_annotate_cell(sa[k][1], f"After  f={sa[k][0]}", (140, 255, 160)) for k in range(n)]
    target_w = min(row_b[0].shape[1], row_a[0].shape[1])
    row_b = [cv2.resize(im, (target_w, cell_height)) if im.shape[1] != target_w else im for im in row_b]
    row_a = [cv2.resize(im, (target_w, cell_height)) if im.shape[1] != target_w else im for im in row_a]
    sheet = np.vstack([np.hstack(row_b), np.hstack(row_a)])
    return out_path if _save_jpeg(out_path, sheet) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="airtime.eval",
        description="Airtime 基准指标汇总、前后对比与目检拼图工具",
    )
    ap.add_argument("run_dir", nargs="?", help="单次运行输出目录（如 runs/smoke_cli_fix）")
    ap.add_argument("--before", help="改动前的基线输出目录")
    ap.add_argument("--after", help="改动后的实验输出目录")
    ap.add_argument("--sheet-dir", help="导出关键帧目检拼图的目录（有 overlay.mp4 时自动生成）")
    ap.add_argument("--sheet-frames", type=int, default=6, help="每张目检拼图抽取的关键帧数（默认 6）")
    ap.add_argument("--json", dest="json_out", help="将结构化结果写入 JSON 文件")
    ap.add_argument("--md", dest="md_out", help="将 Markdown 报告写入文件")
    args = ap.parse_args(argv)

    sheets: dict[str, str] = {}
    if args.before and args.after:
        before_path = Path(args.before)
        after_path = Path(args.after)
        before_rep = evaluate_run(before_path)
        after_rep = evaluate_run(after_path)
        result: dict[str, Any] = compare_runs(before_rep, after_rep)
        if args.sheet_dir:
            sdir = Path(args.sheet_dir)
            for vid in result["common_videos"]:
                out_img = render_compare_sheet(
                    before_path, after_path, vid, sdir / f"{vid}_compare.jpg", n_frames=args.sheet_frames
                )
                if out_img is not None:
                    sheets[vid] = str(out_img)
            result["visual_sheets"] = sheets
        md_text = format_compare_markdown(result)
    elif args.run_dir:
        run_path = Path(args.run_dir)
        result = evaluate_run(run_path)
        if args.sheet_dir:
            sdir = Path(args.sheet_dir)
            for vid in result["videos"]:
                out_img = render_visual_sheet(
                    run_path, vid, sdir / f"{vid}_sheet.jpg", n_frames=args.sheet_frames
                )
                if out_img is not None:
                    sheets[vid] = str(out_img)
            result["visual_sheets"] = sheets
        md_text = format_single_markdown(result)
    else:
        ap.error("请提供单次目录 run_dir，或同时提供 --before 与 --after")
        return 2

    if sheets:
        md_text += "\n\n**目检拼图**：\n" + "\n".join(f"- `{vid}`: `{p}`" for vid, p in sheets.items())

    print(md_text)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md_out:
        out_p = Path(args.md_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
