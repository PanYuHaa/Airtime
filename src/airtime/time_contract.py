"""Airtime 精确时序契约（源帧号 / PTS / VFR 检测 / 慢放倍率与真实物理时间）。

设计目标：
1. M0 为适配下游模型会将视频转码为 30fps CFR，本模块在转码前保留源视频逐帧 PTS 时间轴，
   并建立「转码帧 k ↔ 源视频帧 source_frame_idx / source_pts_s」的精确双向可追溯映射。
2. 严格区分「播放时间 (playback_time_s)」与「真实物理动作时间 (real_time_s)」：
   - 仅当显式给定 slowmo_factor (>0) 时，标记 real_time_confirmed=True 并输出 real_time_s；
   - 未确认慢放倍率时 real_time_s 置为 None，防止下游物理模块（抛物线高度、角速度 deg/s）
     在慢动作素材上算出失真的物理量。
"""
from __future__ import annotations

import bisect
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "airtime.time_contract.v1"


def _parse_ratio(val: str | None) -> float:
    if not val or val in ("0/0", "N/A"):
        return 0.0
    if "/" in val:
        num_s, den_s = val.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if den != 0.0 else 0.0
    return float(val)


def probe_source_timeline(video_path: Path) -> dict[str, Any]:
    """用 ffprobe 提取源视频流级时间基准与逐帧精确 PTS 时间戳。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,time_base",
        "-show_entries",
        "frame=pts,best_effort_timestamp_time,pkt_duration_time",
        "-of",
        "json",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        tail = (r.stderr or "").strip()[-240:]
        raise RuntimeError(f"ffprobe 探测源时间轴失败: {tail}")

    doc = json.loads(r.stdout)
    stream = (doc.get("streams") or [{}])[0]
    time_base = (stream.get("time_base") or "").strip() or None
    tb_val = _parse_ratio(time_base)
    nominal_fps = _parse_ratio(stream.get("r_frame_rate"))
    avg_fps = _parse_ratio(stream.get("avg_frame_rate")) or nominal_fps

    raw_frames = doc.get("frames") or []
    pts_list: list[int | None] = []
    pts_sec_list: list[float] = []
    fallback_fps = avg_fps if avg_fps > 0 else 30.0

    for idx, f in enumerate(raw_frames):
        raw_pts = f.get("pts")
        pts_int = int(raw_pts) if raw_pts is not None else None
        t_str = f.get("best_effort_timestamp_time")
        if t_str is not None:
            t_sec = float(t_str)
        elif pts_int is not None and tb_val > 0:
            t_sec = pts_int * tb_val
        else:
            t_sec = idx / fallback_fps
        pts_list.append(pts_int)
        pts_sec_list.append(round(t_sec, 6))

    is_vfr = detect_vfr(pts_sec_list)
    return {
        "source_frames": len(pts_sec_list),
        "source_time_base": time_base,
        "source_nominal_fps": round(nominal_fps, 4) if nominal_fps else None,
        "source_avg_fps": round(avg_fps, 4) if avg_fps else None,
        "is_vfr": is_vfr,
        "pts": pts_list,
        "pts_time_s": pts_sec_list,
    }


def detect_vfr(pts_time_s: list[float], rel_jitter_thr: float = 0.05) -> bool:
    """根据逐帧 PTS 间隔判定源视频是否为可变帧率 (VFR)。"""
    if len(pts_time_s) < 4:
        return False
    diffs = np.diff(np.asarray(pts_time_s, dtype=np.float64))
    pos = diffs[diffs > 1e-6]
    if len(pos) < 3:
        return False
    med = float(np.median(pos))
    if med <= 0:
        return False
    # 若 95% 与 5% 分位帧间隔相对中位数偏差超过阈值（默认 5%），判为 VFR
    p05, p95 = np.percentile(pos, [5, 95])
    return bool((p95 - p05) / med > rel_jitter_thr)


def build_time_contract(
    source_timeline: dict[str, Any],
    clip_frames: int,
    clip_fps: float,
    slowmo_factor: float | None = None,
) -> dict[str, Any]:
    """构建转码后 CFR 帧序列到源视频帧序列与真实物理时间的完整契约。"""
    if slowmo_factor is not None and slowmo_factor <= 0:
        raise ValueError(f"slowmo_factor 必须为正数，收到: {slowmo_factor}")
    if clip_fps <= 0:
        raise ValueError(f"clip_fps 必须为正数，收到: {clip_fps}")

    src_times: list[float] = list(source_timeline.get("pts_time_s") or [])
    src_pts: list[int | None] = list(source_timeline.get("pts") or [])
    n_src = len(src_times)
    # 归零起点偏移（部分手机素材首帧 PTS 非 0）
    t0 = src_times[0] if n_src > 0 else 0.0
    norm_src_times = [t - t0 for t in src_times]

    confirmed = slowmo_factor is not None
    factor = float(slowmo_factor) if confirmed else None

    frames: list[dict[str, Any]] = []
    max_align_err_ms = 0.0
    for k in range(int(clip_frames)):
        t_play = k / float(clip_fps)
        src_idx: int | None = None
        src_pts_val: int | None = None
        src_t_val: float | None = None

        if n_src > 0:
            pos = bisect.bisect_left(norm_src_times, t_play)
            if pos <= 0:
                best = 0
            elif pos >= n_src:
                best = n_src - 1
            else:
                before_i, after_i = pos - 1, pos
                best = (
                    before_i
                    if abs(norm_src_times[before_i] - t_play) <= abs(norm_src_times[after_i] - t_play)
                    else after_i
                )
            src_idx = int(best)
            src_pts_val = src_pts[best] if best < len(src_pts) else None
            src_t_val = round(src_times[best], 6)
            err_ms = abs(norm_src_times[best] - t_play) * 1000.0
            max_align_err_ms = max(max_align_err_ms, err_ms)

        real_t = round(t_play / factor, 6) if (confirmed and factor is not None) else None
        frames.append(
            {
                "i": k,
                "playback_time_s": round(t_play, 6),
                "source_frame_idx": src_idx,
                "source_pts": src_pts_val,
                "source_pts_s": src_t_val,
                "real_time_s": real_t,
            }
        )

    summary = {
        "schema": SCHEMA_VERSION,
        "source_frames": int(source_timeline.get("source_frames") or n_src),
        "source_time_base": source_timeline.get("source_time_base"),
        "source_nominal_fps": source_timeline.get("source_nominal_fps"),
        "source_avg_fps": source_timeline.get("source_avg_fps"),
        "is_vfr": bool(source_timeline.get("is_vfr", False)),
        "clip_frames": int(clip_frames),
        "clip_fps": float(clip_fps),
        "slowmo_factor": factor,
        "real_time_confirmed": confirmed,
        "effective_capture_fps": round(float(clip_fps) * factor, 2) if (confirmed and factor) else None,
        "max_source_align_err_ms": round(max_align_err_ms, 2) if n_src > 0 else None,
    }
    return {"summary": summary, "frames": frames}


def span_duration(contract: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, float | None]:
    """计算区间 [start_idx, end_idx] 的播放时长与真实物理时长。

    当 ``real_time_confirmed=False`` 时，``real_duration_s`` 返回 ``None``，
    强制下游（如滞空高度、角速度计算）区分播放时间与真实物理时间。
    """
    frames = contract.get("frames") or []
    if not frames or start_idx < 0 or end_idx >= len(frames) or end_idx < start_idx:
        raise IndexError(f"无效帧区间 [{start_idx}, {end_idx}]（总帧数 {len(frames)}）")
    f0, f1 = frames[start_idx], frames[end_idx]
    play_dt = round(float(f1["playback_time_s"]) - float(f0["playback_time_s"]), 6)
    if contract.get("summary", {}).get("real_time_confirmed") and f0.get("real_time_s") is not None:
        real_dt: float | None = round(float(f1["real_time_s"]) - float(f0["real_time_s"]), 6)
    else:
        real_dt = None
    return {"playback_duration_s": play_dt, "real_duration_s": real_dt}
