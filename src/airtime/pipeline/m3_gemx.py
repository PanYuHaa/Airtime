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
    """单个段跑 GEM-X。

    关键设计：**跨进程**调用。
    主流程环境（.venv-p1）没有 torch；GEM-X 需要独立环境（sp-gem）。
    所以在 sp-gem 解释器里跑 m3_run_gemx.py，主流程只传文件路径。
    """
    import tempfile as _tmp

    runner = Path(__file__).resolve().parent / "m3_run_gemx.py"
    with _tmp.TemporaryDirectory() as td:
        bbx_p = Path(td) / "bbx.npy"
        np.save(bbx_p, np.asarray(bbx, dtype=np.float32))
        r = subprocess.run(
            [PY_SPGEM, str(runner),
             "--clip", str(clip),
             "--bbx", str(bbx_p),
             "--out", str(pt_out),
             "--gem", str(gem),
             "--camera", camera_mode],
            capture_output=True, text=True, cwd=str(gem))
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-400:]
            raise RuntimeError(f"GEM-X 进程失败 rc={r.returncode}: {tail}")
    if not Path(pt_out).exists():
        raise RuntimeError(f"GEM-X 未产出 {pt_out}")
    return pt_out

def main():
    from .. import cfg as CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="")
    ap.add_argument("--out", default=str(CFG.OUTPUT_DIR / "default"))
    ap.add_argument("--gem", default=str(CFG.GEMX_DIR))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not a.pkg:
        a.pkg = a.out

    # 参数从 configs/pipeline.json 读（工程原则）
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
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
