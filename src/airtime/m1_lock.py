#!/usr/bin/env python3
"""
M1 目标运动员锁定（v2：以「做动作」为主）

旧版存在这个问题：
  「可用帧占比」权重 0.5 → 整段都在画面里的旁观者得分最高。
  video10 锁定了扛雪板走路的人，真正跳台的运动员反而没有骨架。

修正原则：运动员的特征是「做动作」，不是「出现得久」。
  主特征：
    - 沿滑行方向的**速度**（运动员滑行比走路快）
    - 竖直方向的**加速度 / 腾空**（跳台腾空是强特征）
    - 轨迹**跨越区域的幅度**（从助滑道到落地区）
  次特征：
    - 可用帧数（框够大）
    - 出现时长
"""

import numpy as np

from . import cfg as CFG

_A = CFG.load_pipeline_config()["m1_track"]["assoc"]
COST_MAX = _A["cost_max"]
MAX_MISS = _A["max_miss_frames"]

# 目标锁定的通用参数（2026-11-24 加，锁到拍摄者影子后补的）
_LOCK = CFG.load_pipeline_config()["m1_track"].get("lock", {})


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
                    - inter + 1e-9)


def _assoc_cost(a, b, pred=None):
    """关联代价（0~1，越小越好）。速度预测 + 距离 + 尺寸。"""
    i = iou(a, b)
    cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    ha, hb = a[3]-a[1], b[3]-b[1]
    sz = max(ha, hb, 1.0)

    ref = pred if pred is not None else a
    cr = ((ref[0]+ref[2])/2, (ref[1]+ref[3])/2)
    d = np.hypot(cb[0]-cr[0], cb[1]-cr[1])
    d_cost = min(d / (2.5 * sz), 1.0)

    size_ratio = min(ha, hb) / max(ha, hb, 1e-6)
    s_cost = 1.0 - size_ratio
    iou_cost = 1.0 - i

    if i >= 0.4:
        return 0.45 * iou_cost + 0.35 * d_cost + 0.20 * s_cost
    return 0.75 * d_cost + 0.25 * s_cost


def build_tracks(all_dets, cost_max=None, max_miss=None):
    """
    串轨迹（速度预测 + 全局贪心）。
    轨迹结束后移到 closed 保留（不参与匹配，但参与打分）。
    """
    cost_max = COST_MAX if cost_max is None else cost_max
    max_miss = MAX_MISS if max_miss is None else max_miss

    active, closed = [], []
    for k, dets in enumerate(all_dets):
        preds = {}
        for ti, t in enumerate(active):
            fs = t["frames"]
            if len(fs) >= 2:
                (k1, b1), (k2, b2) = fs[-2], fs[-1]
                dk = max(k2 - k1, 1)
                v = [((b2[c] - b1[c]) / dk) * (k - k2) for c in range(4)]
                preds[ti] = [b2[c] + v[c] for c in range(4)]
            else:
                preds[ti] = fs[-1][1]

        cands = []
        for ti, t in enumerate(active):
            for di, b in enumerate(dets):
                c = _assoc_cost(t["last"], b, preds.get(ti))
                if c <= cost_max:
                    cands.append((c, ti, di))
        cands.sort(key=lambda x: x[0])

        used_t, matched_d = set(), set()
        for c, ti, di in cands:
            if ti in used_t or di in matched_d:
                continue
            active[ti]["frames"].append((k, dets[di]))
            active[ti]["last"] = dets[di]
            active[ti]["miss"] = 0
            used_t.add(ti)
            matched_d.add(di)
        for ti, t in enumerate(active):
            if ti not in used_t:
                t["miss"] += 1
        for di, b in enumerate(dets):
            if di not in matched_d:
                active.append({"frames": [(k, b)], "last": b, "miss": 0,
                               "born": k})

        still = []
        for t in active:
            if t["miss"] > max_miss:
                t["closed_at"] = k
                closed.append(t)
            else:
                still.append(t)
        active = still
    return active + closed


def box_quality(b, W, H, thr_h, edge_px=None):
    """
    判定一个检测框能不能当「完整人体」候选。

    2026-11-24 踩过的坑：SAM2 从一个 [0,1455,9,1630] 的种子起跟 ——
    只有 9 px 宽、贴在画面左边缘，是拍摄者身体露出来的一小条；
    之后一路滑到拍摄者的影子上，516 帧里 447 帧框底边贴住画面底部。
    完整、互换、撕裂三项照样全过，只有「目标锁错」能看出来。

    通用判据（不含任何视频编号）：
      · 太窄 / 太矮的框不是完整人体（用框高门限 thr_h 和长宽比共同约束）
      · 框边贴住画面边界 = 人体被裁掉，只露一部分
    返回 (ok, reasons)
    """
    if edge_px is None:
        edge_px = _LOCK.get("edge_px", 2.0)
    min_aspect = _LOCK.get("min_aspect", 0.12)   # 宽/高，正常人 >= 0.15
    max_aspect = _LOCK.get("max_aspect", 3.0)

    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    reasons = []
    if bh < thr_h:
        reasons.append("too_short")
    if bh > 0:
        ar = bw / bh
        if ar < min_aspect:
            reasons.append("too_narrow")
        if ar > max_aspect:
            reasons.append("too_wide")
    # 贴边：框的某条边压在画面边界上（留 edge_px 容差）
    if x1 <= edge_px:
        reasons.append("clip_left")
    if x2 >= W - 1 - edge_px:
        reasons.append("clip_right")
    if y2 >= H - 1 - edge_px:
        reasons.append("clip_bottom")
    if y1 <= edge_px:
        reasons.append("clip_top")
    return (len(reasons) == 0), reasons


def score_track(track, n_frames, W, H, thr_h):
    """
    运动员得分（v3：以「做动作」为主，但对劣质框和贴边轨迹做惩罚）

    v2 的主特征（合计 0.75）：
      motion_score   —— 沿主方向的滑行速度 + 位移跨度
      air_score      —— 竖直方向的加速度峰值（腾空/跳跃的特征）
      region_score   —— 轨迹覆盖的画面纵向范围（助滑→腾空→落地）
    v2 的次特征（合计 0.25）：
      usable_ratio   —— 框够大的帧占比
      dur_score      —— 出现时长

    v3 新增惩罚（2026-11-24 锁到拍摄者影子）：
      影子轨迹只有 5 帧检测，但框剧烈跳动把 motion/air/region 全拉满 1.0，
      得分 0.753 压过了有 122 帧（111 帧可用）的真运动员轨迹 0.690。
      要压住这种情况，用三个「劣质轨迹」的通用特征：
        edge_ratio    —— 框贴画面边缘的帧占比（拍摄者/影子贴边是典型特征）
        short_ratio   —— 框高不足 thr_h 的帧占比
        track_len     —— 轨迹长度（帧数少 = 证据不足，不能当主目标）
    """
    frames = track["frames"]
    if len(frames) < 5:
        return -1.0, {}

    ks = np.array([k for k, _ in frames], dtype=float)
    boxes = np.array([b for _, b in frames], dtype=float)
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    cys = (boxes[:, 1] + boxes[:, 3]) / 2
    hs = boxes[:, 3] - boxes[:, 1]

    if len(ks) < 5:
        return -1.0, {}

    # 时间间隔（帧）
    dt = np.diff(ks)
    dt[dt == 0] = 1

    # ---- motion_score：滑行速度 + 位移跨度 ----
    vx = np.diff(cxs) / dt
    vy = np.diff(cys) / dt
    speed = np.hypot(vx, vy)
    # 速度（像素/帧）归一化到画面长边，取 P90
    v_p90 = float(np.percentile(speed, 90)) / max(H, W)
    # 位移跨度
    disp = np.linalg.norm(np.stack([cxs, cys], 1) - [cxs[0], cys[0]], axis=1)
    span = float(disp.max()) / max(W, H)
    motion_score = min(v_p90 * 8, 1.0) * 0.6 + min(span * 3, 1.0) * 0.4

    # ---- air_score：竖直加速度峰值（腾空特征） ----
    if len(vy) >= 3:
        ay = np.diff(vy) / dt[:-1]
        ay_peak = float(np.percentile(np.abs(ay), 95))
        air_score = min(ay_peak / max(H, 1) * 300, 1.0)
    else:
        air_score = 0.0

    # ---- region_score：轨迹纵向覆盖范围（助滑→落地） ----
    y_span = float(cys.max() - cys.min()) / max(H, 1)
    region_score = min(y_span * 2, 1.0)

    # ---- 跳变惩罚：框尺寸/位置突变 = 轨迹碎裂，不是一个人连续的运动 ----
    # 2026-11-24：这条素材的一条短轨迹 只有 5 帧，框高在 70 ↔ 1273 之间跳了 18 倍，
    # 靠这种跳变把 motion/air/region 全拉满。这类「跳变」的轨迹不该得高分。
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.maximum(hs[1:], 1) / np.maximum(hs[:-1], 1)
        jump = np.maximum(r, 1.0 / np.maximum(r, 1e-6))
    jump_ratio = float(np.percentile(jump, 90)) if len(jump) else 1.0
    jump_ratio = min(jump_ratio, 50.0)
    jump_factor = 1.0 / (1.0 + max(0.0, jump_ratio - 1.5)
                         * _LOCK.get("penalty_jump", 0.30))

    # ---- 次特征 ----
    n_usable = int(np.sum(hs >= thr_h))
    usable_ratio = n_usable / max(n_frames, 1)
    dur_score = len(frames) / max(n_frames, 1)

    # ---- v3 惩罚项：识别「拍摄者 / 影子 / 露一半的身体」这类劣质轨迹 ----
    # 逐帧判框是否贴边或不合格（和 box_quality 同一套判据）
    edge_px = _LOCK.get("edge_px", 2.0)
    n_edge = 0
    n_short = 0
    for b in boxes:
        _, why = box_quality(b, W, H, thr_h, edge_px)
        if any(r.startswith("clip_") for r in why):
            n_edge += 1
        if "too_short" in why or "too_narrow" in why or "too_wide" in why:
            n_short += 1
    edge_ratio = n_edge / max(len(boxes), 1)
    short_ratio = n_short / max(len(boxes), 1)

    # 轨迹长度门槛：证据太少（帧数不到要求）的轨迹不能当主目标。
    # 某条素材里拍摄者影子的轨迹只有 5 帧检测，真运动员轨迹 211 帧。
    # 门槛设成 211 的一半左右 —— 用「全片可用帧」的固定比例更通用，
    # 但固定帧数对短片不公平，所以两者取小。
    min_len = min(_LOCK.get("min_track_frames", 20),
                  max(5, int(n_frames * _LOCK.get("min_track_ratio", 0.10))))
    len_factor = min(len(frames) / max(min_len, 1), 1.0)

    # 时长占比单独作为一项特征（权重 w_dur）：连续跟踪时间长 = 更可能是主体。
    # 2026-11-24 实测：短片段（29 帧）靠 motion/region 就能冲到第一，
    # 把这一项权重从 0.10 提到 0.20 后，211 帧的真运动员轨迹才排到第一，
    # 且其余 20 条锁定的轨迹索引完全不变（见 网格搜索验证 的网格搜索）。
    w_dur = _LOCK.get("weight_duration", 0.20)

    # 其余四项按 (1 - w_dur) 等比缩放，保证总分仍在 0~1
    rest = 1.0 - w_dur
    score = (0.30 * rest * motion_score
             + 0.28 * rest * air_score
             + 0.17 * rest * region_score
             + 0.15 * rest * min(usable_ratio * 1.6, 1.0)
             + w_dur * dur_score)
    # 惩罚：贴边、框不合格、证据不足、跳变
    w_edge = _LOCK.get("penalty_edge", 0.35)
    w_short = _LOCK.get("penalty_short", 0.30)
    score = score * len_factor * jump_factor \
        * (1.0 - w_edge * edge_ratio) * (1.0 - w_short * short_ratio)

    return score, {
        "frames": len(frames),
        "n_usable": n_usable,
        "usable_ratio": round(usable_ratio, 3),
        "motion_score": round(motion_score, 3),
        "air_score": round(air_score, 3),
        "region_score": round(region_score, 3),
        "edge_ratio": round(edge_ratio, 3),
        "short_ratio": round(short_ratio, 3),
        "len_factor": round(len_factor, 3),
        "jump_ratio": round(jump_ratio, 2),
        "jump_factor": round(jump_factor, 3),
        "v_p90_px": round(float(np.percentile(speed, 90)), 2),
        "ay_peak_px": round(float(np.percentile(np.abs(np.diff(vy) / dt[:-1]), 95))
                            if len(vy) >= 3 else 0.0, 3),
        "y_span_px": round(float(cys.max() - cys.min()), 1),
        "median_h": round(float(np.median(hs)), 1),
        "k_range": [int(ks.min()), int(ks.max())],
    }


def match_hint(tracks, hint):
    """人工提示优先"""
    frame, x, y = hint["frame"], hint["x"], hint["y"]
    best, best_d = None, 1e18
    for ti, t in enumerate(tracks):
        for k, b in t["frames"]:
            if abs(k - frame) > 2:
                continue
            inside = (b[0] <= x <= b[2] and b[1] <= y <= b[3])
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            d = 0.0 if inside else np.hypot(cx-x, cy-y)
            if d < best_d:
                best_d, best = d, ti
    return best if best is not None and best_d < 200 else None


def lock_target(all_dets, W, H, fps=30.0, thr_h=80.0, target_hint=None,
                verbose=False):
    n = len(all_dets)
    tracks = build_tracks(all_dets)
    if verbose:
        print(f"    候选轨迹: {len(tracks)} 条")

    if not tracks:
        return [None] * n, {"method": "none", "n_tracks": 0}

    if target_hint:
        ti = match_hint(tracks, target_hint)
        if ti is not None:
            t = tracks[ti]
            boxes = [None] * n
            for k, b in t["frames"]:
                boxes[k] = b
            sc, det = score_track(t, n, W, H, thr_h)
            if verbose:
                print(f"    人工提示锁定 #{ti} ({len(t['frames'])} 帧)")
            return boxes, {"method": "target_hint", "track_idx": ti,
                           "score": round(sc, 3), "n_tracks": len(tracks),
                           "detail": det}

    scored = []
    for ti, t in enumerate(tracks):
        sc, det = score_track(t, n, W, H, thr_h)
        scored.append((sc, ti, det))
    scored.sort(reverse=True, key=lambda x: x[0])

    sc, ti, det = scored[0]
    t = tracks[ti]
    boxes = [None] * n
    for k, b in t["frames"]:
        boxes[k] = b

    # 说明：试过「主轨迹断掉后按时间/尺寸/位置连续性把新轨迹接回来」，
    # 实测 7 条视频接了 0 段（断点处的轨迹往往不满足连续性条件），
    # 无法证明有效，因此已移除，不留未验证的改动。
    if verbose:
        print(f"    自动锁定 #{ti} ({len(t['frames'])} 帧, 得分 {sc:.3f})")
        for s, i, d in scored[:3]:
            if not d:      # 短轨迹 detail 为空
                print(f"      #{i}: {s:.3f} (轨迹太短)")
                continue
            print(f"      #{i}: {s:.3f} k={d.get('k_range','?')} "
                  f"motion={d.get('motion_score','?')} "
                  f"air={d.get('air_score','?')} "
                  f"region={d.get('region_score','?')} "
                  f"edge={d.get('edge_ratio','?')} "
                  f"short={d.get('short_ratio','?')} "
                  f"medh={d.get('median_h','?')}")

    return boxes, {"method": "auto_score", "track_idx": ti,
                   "score": round(sc, 3), "n_tracks": len(tracks),
                   "detail": det,
                   "candidates": [{"idx": i, "score": round(s, 3), "detail": d}
                                  for s, i, d in scored[:5]]}


def usable_detections(all_dets, W, H, thr_h, edge_px=None):
    """
    把每帧检测框里「能当起点」的那些挑出来（完整人体、不贴边、够大）。

    2026-11-24：这条素材的 SAM2 种子是 [0,1455,9,1630] —— 9 px 宽、贴左边缘，
    是拍摄者身体露出来的一小条。起点不合格，后面无论如何都会跟错。
    """
    out = []
    for dets in all_dets:
        keep = []
        for b in dets:
            ok, _ = box_quality(b, W, H, thr_h, edge_px)
            if ok:
                keep.append(b)
        out.append(keep)
    return out
