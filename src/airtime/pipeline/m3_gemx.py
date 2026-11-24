#!/usr/bin/env python3
"""
GEM-X 按段推理

要求：
  - 每个 skeleton 段单独跑一次
  - 用我们的框替换 GEM-X 自动跟踪
  - **相机模式 background**（方案 P0 结论）
  - 输出 gem/<段号>.npz，格式同 P0，加 frame_offset

复用 P0 的 run_gemx 逻辑。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# 必须在 torch 初始化 CUDA 之前设置，否则 use_deterministic_algorithms
# 会报 "CUBLAS_WORKSPACE_CONFIG" 相关的错误。放模块顶层、torch 导入之前。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PY_SPGEM = (os.environ.get("AIRTIME_GEMX_PYTHON")
            or str(Path.home() / "miniconda3/envs/sp-gem/bin/python"))
M3_EXPORT = str(Path(__file__).resolve().parent / "m3_export.py")
PAD = 10   # 段两端扩帧（给时序模型上下文）


def crop_segment(video, start, end, out_path, fps=30):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = max(0, start - PAD)
    e = min(n - 1, end + PAD)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    frames = []
    for _ in range(e - s + 1):
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
         str(out_path)],
        stdin=subprocess.PIPE)
    for im in frames:
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    proc.wait()
    return {"start": s, "end": e, "n": len(frames), "W": w, "H": h}


def run_gemx_segment(clip, bbx, pt_out, gem, camera_mode="background"):
    """单个段跑 GEM-X"""
    sys.path.insert(0, str(gem))
    import copy

    import torch
    import gem.inference.body_video as BV

    # ---- 确定性开关（1.0 发布阻塞项，见方案第 9 节）----
    # GEM-X 是 torch 模型。同一输入两次跑，j3d_world 有 91% 元素不同、
    # 最大差 1.07e-6 m —— 这是 GPU 浮点归约顺序导致的，不是精度不足。
    # 关掉 cuDNN 的算法自动选择、固定种子后，归约顺序才固定。
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    from gem.common.detectors.tracks import (TrackResult, smooth_boxes,
                                             square_crops)
    from gem.inference.camera_motion import camera_angular_velocity
    from gem.inference.motion import load_predictor, run_motion
    from gem.inference.serialization import save_predictions, tensor_tree
    from gem.inference.video import VideoFrames, normalize_video

    cfg0 = BV.load_body_video_config(
        Path(gem) / "docs/configs/body_video_native77.yaml")
    predictor = load_predictor(cfg0["motion"]["checkpoint"], cfg0["motion"],
                               stats_path=cfg0["motion"].get("stats"),
                               device="cuda")

    def make_tracks(n_frames, H, W, bbx_in):
        B = np.asarray(bbx_in, dtype=np.float32)
        if len(B) != n_frames:
            src = np.linspace(0, 1, len(B))
            dst = np.linspace(0, 1, n_frames)
            B = np.stack([np.interp(dst, src, B[:, c]) for c in range(4)],
                         1).astype(np.float32)
        B[:, [0, 2]] = np.clip(B[:, [0, 2]], 0, W-1)
        B[:, [1, 3]] = np.clip(B[:, [1, 3]], 0, H-1)
        empty = [np.zeros((0, 4), np.float32) for _ in range(n_frames)]
        return TrackResult(
            np.arange(n_frames), B, square_crops(smooth_boxes(B), 1.2),
            np.zeros(n_frames, np.int32), np.ones(n_frames, bool),
            (0, n_frames-1), [b[None] for b in B], empty,
            [b[None] for b in B],
            [np.ones(1, np.float32) for _ in range(n_frames)],
            ["provided_by_ski_professor"] * n_frames)

    cfg = copy.deepcopy(cfg0)
    cfg.setdefault("camera", {})["mode"] = camera_mode

    BV.detect_track = lambda frames, det, **kw: make_tracks(
        len(frames), *frames[0].shape[:2], bbx)

    with tempfile.TemporaryDirectory() as tmp:
        meta = normalize_video(clip, Path(tmp) / "in.mp4")
        with VideoFrames(Path(tmp) / "in.mp4") as frames:
            seq, tracks, camera = BV.build_body_conditions(
                frames, cfg, device="cuda", progress=lambda m: None)
        meta["normalized"].pop("path", None)
        s = dict(seq)
        nf = int(seq["length"][0])
        if camera is None:
            s["cam_angvel"] = camera_angular_velocity(
                torch.eye(3).repeat(nf, 1, 1)).unsqueeze(0)
            cam = {"R_w2c": torch.eye(3).repeat(nf, 1, 1),
                   "diagnostics": {"method": "static"}}
        else:
            cam = camera
        res = run_motion(predictor, s, device="cuda")
        res = tensor_tree({**res, "format": "gem_body_video_predictions_v1",
                           "video": meta, "release": cfg, "conditions": s,
                           "tracking": {k: v for k, v in tracks.__dict__.items()},
                           "camera": cam})
        res["video"]["source"]["path"] = str(clip.resolve())
        pt_out.parent.mkdir(parents=True, exist_ok=True)
        save_predictions(res, pt_out)

    del predictor
    torch.cuda.empty_cache()
    return pt_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="")
    ap.add_argument("--out", default=str(Path.home() / "output_out"))
    ap.add_argument("--gem", default=str(Path.home() / "GEM-X"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not a.pkg:
        a.pkg = a.out

    # 参数从 configs/pipeline.json 读（工程原则）
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from .. import cfg as CFG
    _g = CFG.load_pipeline_config()["m3_gemx"]
    global PAD
    PAD = _g["pad_frames"]
    max_seg_sec = _g["max_segment_sec"]

    pkg = Path(a.pkg)
    out = Path(a.out)
    gem = Path(a.gem)

    index = json.load(open(pkg / "index.json", encoding="utf-8"))
    only = [x for x in a.only.split(",") if x]

    import time as _time
    gem_stats = {}

    for vid in sorted(index):
        if only and vid not in only:
            continue

        _t_vid = _time.time()
        video = pkg / "clips" / f"{vid}.mp4"
        seg_p = out / "segments" / f"{vid}.json"
        trk_p = out / "tracking" / f"{vid}.json"

        if not (video.exists() and seg_p.exists() and trk_p.exists()):
            print(f"  x {vid}: 缺输入")
            continue

        seg_doc = json.load(open(seg_p, encoding="utf-8"))
        segs = seg_doc["segments"]
        windows = seg_doc.get("inference_windows", [])
        tracking = json.load(open(trk_p, encoding="utf-8"))
        n = len(tracking)

        # 相机模式统一策略（方案要求）：
        #   三脚架 → static
        #   移动机位 → background，失败则本段标「不可信」，不当作正常结果
        is_tripod = bool(index[vid].get("static_cam"))
        camera = "static" if is_tripod else "background"

        gem_dir = out / "gem" / vid
        gem_dir.mkdir(parents=True, exist_ok=True)
        work = out / "work" / vid
        work.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*74}")
        print(f"  {vid}: {n} 帧, {len(segs)} 段, {len(windows)} 窗口, "
              f"相机={camera} ({'三脚架' if is_tripod else '移动机位'})")
        print(f"{'='*74}")

        # 按推理窗口跑（窗口内可能含多个骨架段）
        for wi, win in enumerate(windows):
            ws, we = win["win_start"], win["win_end"]
            members = win["segments"]
            npz_out = gem_dir / f"seg{wi}.npz"

            if npz_out.exists():
                print(f"  窗口{wi}: {ws}-{we} 已存在")
                continue

            dur = (we - ws + 1) / 30.0
            print(f"  窗口{wi}: {ws}-{we} ({we-ws+1}帧, {dur:.2f}s), "
                  f"含 {len(members)} 段", flush=True)

            # 段太长 → 再切（通用规则：限制段长，避免相机估计丢失）
            # 注意：最终送进 GEM-X 的长度 = (窗口长 + 2*PAD)，所以切块要扣掉 PAD
            sub_windows = []
            max_core = max(30, int(max_seg_sec * 30) - 2 * PAD)
            if (we - ws + 1) > max_core:
                s = ws
                while s <= we:
                    e = min(we, s + max_core - 1)
                    sub_windows.append((s, e))
                    s = e + 1
                print(f"    窗口 {we-ws+1} 帧超限 → 切成 {len(sub_windows)} 块"
                      f"（每块核心 <= {max_core} 帧，含 pad 后 <= "
                      f"{max_core + 2*PAD}）", flush=True)
            else:
                sub_windows = [(ws, we)]

            seg_results = []
            for si, (ss, se) in enumerate(sub_windows):
                tag = f"seg{wi}" + (f"_p{si}" if len(sub_windows) > 1 else "")
                clip = work / f"{tag}.mp4"
                pt = work / f"{tag}.pt"

                if not clip.exists():
                    info = crop_segment(video, ss, se, clip)
                    if info is None:
                        print(f"    x 裁片失败", flush=True)
                        continue
                else:
                    info = {"start": max(0, ss-PAD), "end": min(n-1, se+PAD)}

                bbx_slice = []
                for k in range(info["start"], info["end"]+1):
                    b = tracking[k]["bbox"]
                    if b is None:
                        idx = next((i for i in range(k+1, n)
                                    if tracking[i]["bbox"] is not None), None)
                        if idx is None:
                            idx = next((i for i in range(k-1, -1, -1)
                                        if tracking[i]["bbox"] is not None),
                                       None)
                        b = (tracking[idx]["bbox"] if idx is not None
                             else [0, 0, 50, 50])
                    bbx_slice.append(b)
                bbx_slice = np.array(bbx_slice, dtype=np.float32)

                t0 = time.time()
                cam_used = camera
                cam_trusted = True
                cam_note = ""

                if not pt.exists():
                    try:
                        run_gemx_segment(clip, bbx_slice, pt, gem, camera)
                        print(f"    {tag}: ✓ {time.time()-t0:.0f}s "
                              f"(相机={camera})", flush=True)
                    except Exception as ex:
                        err = f"{type(ex).__name__}: {str(ex)[:130]}"
                        print(f"    {tag}: background 失败: {err}", flush=True)

                        if camera == "background":
                            # 统一策略：失败 → 标不可信，用 static 兜底出数据，
                            # 但记录原因，下游世界坐标数据必须带低可信标记
                            cam_trusted = False
                            cam_note = err
                            print(f"    {tag}: → 用 static 兜底（相机状态标不可信）",
                                  flush=True)
                            try:
                                run_gemx_segment(clip, bbx_slice, pt, gem, "static")
                                cam_used = "static"
                                print(f"    {tag}: ✓ {time.time()-t0:.0f}s "
                                      f"(static 兜底, 不可信)", flush=True)
                            except Exception as ex2:
                                print(f"    {tag}: x static 也失败: "
                                      f"{str(ex2)[:120]}", flush=True)
                                continue
                        else:
                            continue
                else:
                    print(f"    {tag}: 已存在")

                # 导出
                tmp_npz = work / f"{tag}.npz"
                r = subprocess.run(
                    [PY_SPGEM, M3_EXPORT,
                     "--pt", str(pt), "--out", str(tmp_npz),
                     "--soma", str(gem / "assets/soma")],
                    capture_output=True, text=True)
                if r.returncode != 0 or not tmp_npz.exists():
                    print(f"    x 导出失败: {r.stderr[-150:]}", flush=True)
                    continue

                d = dict(np.load(tmp_npz))
                d["frame_offset"] = np.array(info["start"], dtype=np.int32)
                d["camera_mode"] = np.array(cam_used)
                d["camera_trusted"] = np.array(cam_trusted)
                d["camera_note"] = np.array(cam_note)
                d["win_segments"] = np.array(members, dtype=np.int32)
                save_p = gem_dir / f"{tag}.npz"
                np.savez_compressed(save_p, **d)
                tmp_npz.unlink(missing_ok=True)
                seg_results.append({"file": save_p.name,
                                    "camera": cam_used,
                                    "trusted": cam_trusted,
                                    "note": cam_note})

            if seg_results:
                # 窗口级元数据
                meta_p = gem_dir / f"seg{wi}_meta.json"
                meta_p.write_text(json.dumps({
                    "window": [ws, we],
                    "segments": members,
                    "camera_planned": camera,
                    "parts": seg_results,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

        gem_stats[vid] = {
            "frames": n,
            "n_windows": len(windows),
            "n_blocks": len(list(gem_dir.glob("seg*.npz"))) if gem_dir.exists() else 0,
            "sec": round(_time.time() - _t_vid, 1),
        }

    # 汇总
    print(f"\n{'='*74}")
    print(f"  完成")
    print(f"{'='*74}")
    for vid in sorted(index):
        vd = out / "gem" / vid
        if vd.exists():
            npzs = sorted(vd.glob("seg*.npz"))
            metas = sorted(vd.glob("seg*_meta.json"))
            cams = []
            for m in metas:
                md = json.loads(m.read_text(encoding="utf-8"))
                for p in md["parts"]:
                    cams.append(f"{p['camera']}{'' if p['trusted'] else '(不可信)'}")
            print(f"  {vid:<9} {len(npzs)} 块  相机: {cams}")

    (out / "gem_stats.json").write_text(
        json.dumps(gem_stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
