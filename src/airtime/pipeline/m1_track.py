#!/usr/bin/env python3
"""
P1 T1.2 跟踪（通用修法版）

按方案「〇·五 工程原则」：
  - 目标锁定用通用规则（自动打分 + target_hint 标准输入），不含视频编号分支
  - 切段写成函数：段内只能是连续 skeleton，坏帧 > 15 帧断开
  - 相机模式统一策略（三脚架 static，移动 background）
  - 不再用旧框填充

输出：
  tracking/<vid>.json    每帧 t/bbox/source/box_h/status
  segments/<vid>.json    严格连续的 skeleton 段 + 推理窗口
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .. import cfg as CFG
from .. import det_deterministic as DET
from ..cfg import CONFIGS_DIR, SAM2_DIR, YOLOX_ONNX

from ..m1_lock import box_quality, iou, lock_target

# 全部参数从 configs/pipeline.json 读，代码里不硬编码（工程原则）
_P = CFG.load_pipeline_config()
_T = _P["m1_track"]
MIN_HEIGHT_PX = _T["min_person_height_px"]
REF_WIDTH = _T["height_ref_width"]
MAX_GAP_SEC = _T["max_gap_sec"]
FPS = _P["m0_preprocess"]["target_fps"]
MIN_SEG_FRAMES = _T["min_segment_frames"]
ASSOC_COST_MAX = _T["assoc"]["cost_max"]
ASSOC_MAX_MISS = _T["assoc"]["max_miss_frames"]
SAM2_CFG = _T["sam2"]
_LOCK = _T.get("lock", {})


def detect_all(video, score_thr=0.15):
    from rtmlib import YOLOX
    # 确定性：必须在建 YOLOX 之前打补丁，否则会话已经按默认（非确定）配置建好。
    # 实测同一段两个独立进程跑，boxes 最大差 0.0126 px、kpts 0.537 px；
    # 打补丁后跨进程 bit-identical（见 tests/run_det_xproc_det.sh）。
    # DET 已在模块顶部导入
    DET.patch_rtmlib(verbose=False)
    det = YOLOX(
        str(YOLOX_ONNX),
        model_input_size=(640, 640), score_thr=score_thr,
        backend="onnxruntime", device="cuda")
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_dets = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        r = det(im)
        all_dets.append([list(map(float, b[:4])) for b in (r if len(r) else [])])
    cap.release()
    return all_dets, W, H


def sam2_fill(video, seed_k, seed_box, W, H):
    """SAM2 从种子传播，返回 {frame_idx: box}

    只用单个种子（框最大的那一帧）向前后传播。
    试过在缺口段额外播种，会误把「路人」当主体补进来（v2606 帧 347-349），
    已回退 —— 详见 process() 里的说明。
    """
    import shutil
    import tempfile

    import torch
    sys.path.insert(0, str(SAM2_DIR))
    from sam2.build_sam import build_sam2_video_predictor

    ckpt = Path.home() / "A2_pkg/models/sam2.1_hiera_small.pt"
    if not ckpt.exists():
        return {}

    tmpdir = Path(tempfile.mkdtemp(prefix="sam2t_"))
    cap = cv2.VideoCapture(str(video))
    scale = min(1.0, 720.0 / min(W, H))
    tw, th = int(W*scale)//2*2, int(H*scale)//2*2
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(tmpdir / f"{n:05d}.jpg"),
                    cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()

    pred = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml", str(ckpt), device="cuda")
    sb = [seed_box[0]*scale, seed_box[1]*scale,
          seed_box[2]*scale, seed_box[3]*scale]
    out = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(str(tmpdir), offload_video_to_cpu=True)
        pred.add_new_points_or_box(st, frame_idx=seed_k, obj_id=1,
                                   box=np.array(sb, dtype=np.float32))
        for fidx, _, ml in pred.propagate_in_video(st, start_frame_idx=seed_k,
                                                   reverse=False):
            m = (ml[0, 0] > 0).cpu().numpy()
            if m.any():
                ys, xs = np.where(m)
                out[fidx] = [xs.min()/scale, ys.min()/scale,
                             xs.max()/scale, ys.max()/scale]
        for fidx, _, ml in pred.propagate_in_video(st, start_frame_idx=seed_k,
                                                   reverse=True):
            m = (ml[0, 0] > 0).cpu().numpy()
            if m.any():
                ys, xs = np.where(m)
                out[fidx] = [xs.min()/scale, ys.min()/scale,
                             xs.max()/scale, ys.max()/scale]
    pred.reset_state(st)
    del pred
    torch.cuda.empty_cache()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def build_status(boxes, sources, thr_h):
    """
    判定每帧状态。thr_h 是**进门门限**；出门用带迟滞的更低门限。

    为什么要迟滞（Mac 第二节第 2 步）：实测两次运行里 video5 切段从
    [0-150] 翻成 [0-151]，就是因为第 151 帧的框高正好压在 thr_h 这条线上，
    bbox 差 0.1 px 就让它从 skeleton 变 track_only。这种边界帧必须用
    「进门难、出门易」的双门限，否则任何数值抖动都会让判断结果翻转。

    进门：source 是 detect/sam2 且 框高 >= thr_h
    出门：已经处于 skeleton 时，只要 框高 >= thr_h * (1 - HYS) 就继续保持
    sam2_lost 帧一律 track_only（跟丢的段不能画骨架，2026-11-24）
    """
    n = len(boxes)
    HYS = _T.get("height_hysteresis", 0.10)   # 出门门限放宽 10%
    exit_h = thr_h * (1.0 - HYS)
    status = []
    prev_in = False
    for k, b in enumerate(boxes):
        if b is None:
            status.append("none")
            prev_in = False
            continue
        h = b[3] - b[1]
        good_src = sources[k] in ("detect", "sam2")
        # 当前帧进来用硬门限；若上一帧已经在骨架里，用放宽的出门门限
        lim = exit_h if prev_in else thr_h
        is_skel = good_src and h >= lim
        status.append("skeleton" if is_skel else "track_only")
        prev_in = is_skel
    return status


def split_segments(status, max_gap_frames, min_frames):
    """段 = 连续 skeleton。空档 > max_gap 断开。这个函数是唯一入口，全部视频共用。"""
    n = len(status)
    runs = []
    k = 0
    while k < n:
        if status[k] == "skeleton":
            s = k
            while k < n and status[k] == "skeleton":
                k += 1
            runs.append((s, k - 1))
        else:
            k += 1
    # 丢掉过短的段，同时把这些帧的状态降级（否则 skeleton 帧会在段外且无 3D 数据）
    kept = []
    for s, e in runs:
        if e - s + 1 >= min_frames:
            kept.append((s, e))
        else:
            for i in range(s, e + 1):
                status[i] = "track_only"
    return kept


def merge_windows(segs, max_gap_frames):
    """间隔 <= max_gap 的相邻段共用一次 GEM-X 调用（保时序连续）"""
    if not segs:
        return []
    groups = []
    cs, ce = segs[0]
    members = [segs[0]]
    for s, e in segs[1:]:
        if s - ce - 1 <= max_gap_frames:
            ce = e
            members.append((s, e))
        else:
            groups.append((cs, ce, members))
            cs, ce, members = s, e, [(s, e)]
    groups.append((cs, ce, members))
    return groups


def process(vid, video, meta, config):
    W, H = meta["output_size"]
    n = meta["frames"]
    thr_h = MIN_HEIGHT_PX * W / REF_WIDTH
    is_tripod = bool(meta.get("static_cam"))

    print(f"\n{'='*74}", flush=True)
    print(f"  {vid}: {W}x{H}, {n} 帧, {'三脚架' if is_tripod else '移动机位'}",
          flush=True)
    print(f"{'='*74}", flush=True)

    t0 = time.time()
    print(f"  [1/4] YOLOX 检测...", flush=True)
    all_dets, _, _ = detect_all(video)
    n_raw = sum(1 for d in all_dets if d)
    print(f"    有检测帧: {n_raw}/{n}", flush=True)

    # 目标锁定（通用规则 + target_hint）
    print(f"  [2/4] 目标锁定（通用规则）...", flush=True)
    hint = config.get("target_hint")
    boxes, lock_info = lock_target(all_dets, W, H, FPS, thr_h,
                                   target_hint=hint, verbose=True)
    if lock_info["method"] == "none":
        print(f"    x 无候选轨迹", flush=True)
        sources = ["none"] * n
        status = ["none"] * n
        return finish(vid, meta, boxes, sources, status, [], lock_info,
                      is_tripod, t0)

    n_track = sum(1 for b in boxes if b is not None)
    print(f"    锁定后: {n_track}/{n} 帧", flush=True)

    # SAM2 补缺。种子 = 锁定轨迹里**合格**且框最大的帧。
    sources = ["detect" if boxes[k] is not None else "none" for k in range(n)]
    missing = [k for k in range(n) if boxes[k] is None]
    if missing:
        areas = [(boxes[k][3]-boxes[k][1]) if boxes[k] else 0
                 for k in range(n)]
        # 起点必须合格：只能是检测器给出的完整人体框。
        #
        # v2615 的种子曾是 [0,1455,9,1630] —— 9px 宽、贴左边缘的残片，
        # SAM2 从那里起跟，一路滑到拍摄者的影子上。
        #
        # 但「不合格就整段跳过」是个更糟的做法（2026-11-24 被 Mac 抓到）：
        # 原来只取「框最大的一帧」当种子，而框最大的往往正好是贴边的近景帧
        # （v2617 帧 3 贴底、v2629 帧 230 贴底），于是整条视频的 SAM2 一次都没跑，
        # 把 v2606 88 帧 / v2617 60 帧 / v2629 42 帧**正确**的补缺帧全丢了。
        #
        # 现在改成：在所有**合格**的检测帧里挑框最大的那一帧当种子。
        # 合格 = 完整人体、不贴边（box_quality）。这样既不会从残片起跟，
        # 也不会因为「最大那帧恰好贴边」就整段不补。
        cand = [k for k in range(n)
                if boxes[k] is not None
                and box_quality(boxes[k], W, H, thr_h)[0]]
        if not cand:
            print(f"  [3/4] 无合格种子帧，跳过 SAM2", flush=True)
            seed_k = None
        else:
            seed_k = max(cand, key=lambda k: areas[k])
        if seed_k is not None and areas[seed_k] > 0:
            _big = int(np.argmax(areas))
            if _big != seed_k:
                print(f"  [3/4] 最大框在帧 {_big}（贴边，不作种子），"
                      f"改用合格的最大框 帧 {seed_k}", flush=True)
            print(f"  [3/4] SAM2 补 {len(missing)} 帧...", flush=True)
            # 只从主种子传播（单次播种）。
            #
            # 曾经试过「在缺口段里额外播种」，想让 SAM2 把断掉的段接回来。
            # **已回退**，原因：缺口段里检测到的往往是**画面里的另一个人**
            # （v2606 帧 347-349 检测框高 184px，是左边远处的小人；
            #  主体是右边那个框高 1310px 的人）。
            # 用那个框额外播种，SAM2 就把「路人」补成骨架，
            # 结果是骨架率从 98.8% "提升"到 100%，
            # 但框高在 184 ↔ 1310 之间反复跳（相邻帧突变 7 倍），
            # 跟踪的是两个人来回切换 —— 这是**假修复**，比留缺口更糟。
            #
            # 结论：缺口宁可如实保留，也不能用不确定的种子去填。
            try:
                sb = sam2_fill(video, seed_k, boxes[seed_k], W, H)
                filled = 0
                for k, b in sb.items():
                    if 0 <= k < n and boxes[k] is None:
                        boxes[k] = b
                        sources[k] = "sam2"
                        filled += 1
                print(f"    SAM2 补 {filled} 帧", flush=True)
            except Exception as e:
                print(f"    x SAM2: {e}", flush=True)
    else:
        print(f"  [3/4] 无需 SAM2 补缺", flush=True)

    # ---- SAM2 监督：长时间漂移就判跟丢（2026-11-24 加）----
    # v2615 原状态：锁到拍摄者影子，SAM2 连续 450 帧（13 秒）没有任何检测
    # 印证，框从拍摄者身上一路滑到影子上 —— 完整/互换/撕裂三项照样全过，
    # 只有目视能看出锁错人。
    #
    # 判据经过两次修正，两个坑都记在这：
    #   坑 1「要求 SAM2 帧必须有检测印证」→ 把正常补缺全判成跟丢。
    #        SAM2 本来就是检测器漏检时才补，那些帧本来就该没有检测框。
    #        实测 v2606 骨架帧从 405 掉到 318（误伤）。
    #   坑 2「连续一段时间没有检测器确认」→ 仍会误伤长时间补缺的条子
    #        （video4/9/10/11 等 10 条统计都变了）。
    #
    # 真正要抓的是**漂移**：正常的 SAM2 补缺，框基本停在原地（检测器暂时
    # 漏检而已）；跟丢时框会持续往一个方向滑走。所以判据是
    #   「连续 sam2 帧里，框中心累计位移 > sam2_drift_ratio × 起始框高」
    # 实测（tests/drift_analysis.py）：
    #   v2615 漂移 597% 起始框高、尺寸变化 7.5 倍  ← 真跟丢
    #   v2606 漂移 14%、v2617 44%、v2629 21%       ← 正常补缺
    sup_sec = _LOCK.get("sam2_supervise_sec", 1.0)
    sup_frames = max(1, int(sup_sec * FPS))
    drift_ratio = _LOCK.get("sam2_drift_ratio", 3.0)

    lost_runs = []
    k = 0
    while k < n:
        if sources[k] != "sam2":
            k += 1
            continue
        s = k
        while k < n and sources[k] == "sam2":
            k += 1
        e = k - 1
        if (e - s + 1) < sup_frames:
            continue
        # 段内逐帧相对于段起点框的累计位移
        b0 = boxes[s]
        if b0 is None:
            continue
        h0 = max(b0[3] - b0[1], 1.0)
        cx0, cy0 = (b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2
        max_drift = 0.0
        for i in range(s, e + 1):
            bi = boxes[i]
            if bi is None:
                continue
            d = np.hypot((bi[0] + bi[2]) / 2 - cx0,
                         (bi[1] + bi[3]) / 2 - cy0)
            max_drift = max(max_drift, d)
        if max_drift / h0 >= drift_ratio:
            lost_runs.append((s, e, round(max_drift / h0, 2)))

    n_lost = sum(e - s + 1 for s, e, _ in lost_runs)
    if lost_runs:
        print(f"    SAM2 漂移 {len(lost_runs)} 段、{n_lost} 帧"
              f"（位移 ≥ {drift_ratio}× 起始框高），改标 track_only",
              flush=True)
        for s, e, dr in lost_runs:
            print(f"      {s}-{e} ({e-s+1}帧, {s/FPS:.2f}-{e/FPS:.2f}s, "
                  f"漂移 {dr}× 框高)", flush=True)
            for i in range(s, e + 1):
                sources[i] = "sam2_lost"

    # 插值只用于轨迹显示，source 如实标 interp
    have = [k for k in range(n) if boxes[k] is not None]
    if len(have) >= 2:
        for k in range(n):
            if boxes[k] is None:
                prev = max([i for i in have if i < k], default=None)
                nxt = min([i for i in have if i > k], default=None)
                if prev is not None and nxt is not None:
                    w = (k - prev) / (nxt - prev)
                    boxes[k] = [boxes[prev][c]*(1-w) + boxes[nxt][c]*w
                                for c in range(4)]
                    sources[k] = "interp"

    status = build_status(boxes, sources, thr_h)
    max_gap = int(MAX_GAP_SEC * FPS)
    segs = split_segments(status, max_gap, MIN_SEG_FRAMES)

    from collections import Counter
    print(f"\n  结果:", flush=True)
    print(f"    来源: {dict(Counter(sources))}", flush=True)
    print(f"    状态: {dict(Counter(status))}", flush=True)
    print(f"    段: {len(segs)} 个", flush=True)
    for s, e in segs:
        print(f"      {s}-{e} ({e-s+1}帧, {s/FPS:.2f}-{e/FPS:.2f}s)", flush=True)

    return finish(vid, meta, boxes, sources, status, segs, lock_info,
                  is_tripod, t0)


def finish(vid, meta, boxes, sources, status, segs, lock_info, is_tripod, t0,
           lost_runs=None):
    n = meta["frames"]
    max_gap = int(MAX_GAP_SEC * FPS)

    tracking = []
    for k in range(n):
        b = boxes[k]
        tracking.append({
            "i": k, "t": round(k/FPS, 4),
            "bbox": [round(float(x), 1) for x in b] if b is not None else None,
            "source": sources[k],
            "box_h": round(float(b[3]-b[1]), 1) if b is not None else 0,
            "status": status[k],
        })

    # 相机模式统一策略：三脚架 → static；移动 → background
    cam_planned = "static" if is_tripod else "background"
    groups = merge_windows(segs, max_gap)

    seg_json = [{"start": int(s), "end": int(e), "frames": int(e-s+1),
                 "t_start": round(s/FPS, 3), "t_end": round(e/FPS, 3)}
                for s, e in segs]
    win_json = [{"win_start": int(ws), "win_end": int(we),
                 "win_frames": int(we-ws+1),
                 "segments": [[int(s), int(e)] for s, e in mem],
                 "camera_planned": cam_planned}
                for ws, we, mem in groups]

    from collections import Counter
    st_c = Counter(status)
    src_c = Counter(sources)

    return {
        "video": vid,
        "meta": meta,
        "tracking": tracking,
        "segments": seg_json,
        "inference_windows": win_json,
        "target_lock": lock_info,
        "stats": {
            "frames": n,
            "detect": src_c.get("detect", 0),
            "sam2": src_c.get("sam2", 0),
            "sam2_lost": src_c.get("sam2_lost", 0),
            "interp": src_c.get("interp", 0),
            "none": src_c.get("none", 0),
            "skeleton": st_c.get("skeleton", 0),
            "track_only": st_c.get("track_only", 0),
            "n_segments": len(segs),
            "n_windows": len(groups),
            "camera_planned": cam_planned,
            "sam2_lost_runs": [[int(a), int(b)]
                               for a, b in (lost_runs or [])],
            "sec": round(time.time()-t0, 1),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="")
    ap.add_argument("--out", default=str(Path.home() / "P1v2_out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not a.pkg:
        a.pkg = a.out

    pkg = Path(a.pkg)
    out = Path(a.out)
    (out / "tracking").mkdir(parents=True, exist_ok=True)
    (out / "segments").mkdir(parents=True, exist_ok=True)

    index = json.load(open(pkg / "index.json", encoding="utf-8"))

    # 读 videos.json 拿 target_hint（标准输入，所有视频都可选填）
    vcfg_p = CONFIGS_DIR / "videos.json"
    vcfg = {}
    if vcfg_p.exists():
        for v in json.load(open(vcfg_p, encoding="utf-8"))["videos"]:
            vcfg[v["id"]] = v

    only = [x for x in a.only.split(",") if x]
    results = {}

    for vid in sorted(index):
        if only and vid not in only:
            continue
        video = pkg / "clips" / f"{vid}.mp4"
        if not video.exists():
            print(f"  x {vid}: 视频不存在", flush=True)
            continue

        cfg = vcfg.get(vid, {})
        r = process(vid, video, index[vid], cfg)

        (out / "tracking" / f"{vid}.json").write_text(
            json.dumps(r["tracking"], ensure_ascii=False), encoding="utf-8")
        (out / "segments" / f"{vid}.json").write_text(
            json.dumps({"video": vid, "segments": r["segments"],
                        "inference_windows": r["inference_windows"],
                        "target_lock": r["target_lock"],
                        "stats": r["stats"]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        results[vid] = r["stats"]

    (out / "tracking_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*74}")
    print(f"  完成 {len(results)} 条")
    print(f"{'='*74}")
    print(f"\n{'视频':<9} {'帧':>4} {'detect':>7} {'sam2':>5} {'interp':>7} "
          f"{'none':>5} {'skel':>5} {'段':>3} {'窗口':>5} {'相机'}", flush=True)
    print("-" * 76, flush=True)
    for vid, s in results.items():
        print(f"{vid:<9} {s['frames']:>4} {s['detect']:>7} {s['sam2']:>5} "
              f"{s['interp']:>7} {s['none']:>5} {s['skeleton']:>5} "
              f"{s['n_segments']:>3} {s['n_windows']:>5} {s['camera_planned']}",
              flush=True)


if __name__ == "__main__":
    main()
