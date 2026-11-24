#!/usr/bin/env python3
"""
骨架融合（含边界撕裂修复）

实践指出的 3 个根因，全部修复：

1. **窗口拼接** → 相邻窗口重叠 1 秒，重叠区对投影做线性加权过渡
2. **校正没有整体约束** → 分两层：
   a) 整体对齐（平移+缩放+旋转，稳健估计）
   b) 逐关节校正
   c) **骨长检查**：校正后骨长与投影相差 > 30%，该肢体链退回整体对齐
3. **段首/段尾过渡** → 过渡从段内往外，不在段首尾第一帧用插值

另：可靠性判断补上「两个独立 2D 互相印证」（实践第五节要求）。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .. import cfg as CFG
from ..cfg import SOMA_JOINT_NAMES


# ============================================================================
# Halpe26
# ============================================================================
H26_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle",
             "head", "neck", "hip",
             "left_big_toe", "right_big_toe", "left_small_toe",
             "right_small_toe", "left_heel", "right_heel"]
H26 = {n: i for i, n in enumerate(H26_NAMES)}

# SOMA77
SOMA_NAMES = json.load(open(
    str(SOMA_JOINT_NAMES), encoding="utf-8"))
S77 = {n: i for i, n in enumerate(SOMA_NAMES)}

# 关节对应：Halpe26 -> SOMA77
#
# 映射按「2D 点与 3D 关节投影的实测中位距离」标定。
#
# 为什么 head 映射到 HeadEnd、nose 不映射：
#   方案只要求「Halpe 的头顶 -> head_end」。早期实现把 nose 也映射到 HeadEnd，
#   等于用一个不是头顶的点做头顶点校正，会把头顶点拉到护目镜附近。
#   实测 GEM-X 自身的 HeadEnd 观测距头顶 4.69%、距鼻子 14.27%，
#   两个独立模型都表明 HeadEnd 就是头顶点，鼻子不应参与。
#
# 已知待改进（不影响当前默认行为）：
#   - nose 和 head 曾同时映射 HeadEnd，导致「两模型互证」只把排在前面的点
#     当搭档，后面的点从未被校验。移除 nose 后暴露了该系统偏移问题。
#   - heel 映射到 Foot，而脚跟并非踝关节位置。
#   这两项需要重新设计融合门控，留待后续版本。
JOINT_MAP = {
    "head": "HeadEnd", "neck": "Neck2",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_hip": "LeftLeg", "right_hip": "RightLeg",
    "left_knee": "LeftShin", "right_knee": "RightShin",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "left_big_toe": "LeftToeEnd", "right_big_toe": "RightToeEnd",
    "left_small_toe": "LeftToeEnd", "right_small_toe": "RightToeEnd",
    "left_heel": "LeftFoot", "right_heel": "RightFoot",
}

SOMA2H26 = {}
for hn, sn in JOINT_MAP.items():
    SOMA2H26.setdefault(sn, []).append(hn)

SWAP_PAIRS = [("LeftArm", "RightArm"), ("LeftForeArm", "RightForeArm"),
              ("LeftHand", "RightHand"), ("LeftLeg", "RightLeg"),
              ("LeftShin", "RightShin"), ("LeftFoot", "RightFoot"),
              ("LeftToeEnd", "RightToeEnd")]

BODY_JOINTS = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]

# 肢体链（用于骨长检查后整链回退）
LIMB_CHAINS = [
    ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
    ["LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd"],
    ["RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd"],
    ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd"],
]

FUSABLE = sorted({sn for sn in JOINT_MAP.values()})


def file_hash(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def estimate_body_height(proj, k):
    head = proj[k][S77["Head"]]
    lf = proj[k][S77["LeftFoot"]]
    rf = proj[k][S77["RightFoot"]]
    if not (np.isfinite(head).all() and np.isfinite(lf).all()
            and np.isfinite(rf).all()):
        return None
    h = np.linalg.norm(head - (lf + rf) / 2)
    return h if h > 5 else None


# ============================================================================
# 几何工具
# ============================================================================

def rigid_fit_2d(src, dst, weights=None):
    """
    2D 相似变换（平移+缩放+旋转）稳健拟合：dst ≈ s*R*src + t
    src, dst: (N,2)，返回 (s, R(2x2), t)
    用加权最小二乘 + 迭代重加权（稳健）
    """
    n = len(src)
    if n < 3:
        return 1.0, np.eye(2), np.zeros(2)

    w = np.ones(n) if weights is None else weights.copy()
    s, R, t = 1.0, np.eye(2), np.zeros(2)

    for _ in range(6):
        wsum = w.sum()
        if wsum < 1e-9:
            break
        # 质心
        cs = (src * w[:, None]).sum(0) / wsum
        cd = (dst * w[:, None]).sum(0) / wsum
        # 去质心
        a = src - cs
        b = dst - cd
        # 复表示最优旋转（Umeyama）
        Sw = (w[:, None] * a * b).sum(0)   # [Σw ax bx, Σw ay by]
        # 用复数法：sum(w * conj(a) * b)
        num = (w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])).sum()
        num2 = (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum()
        den = (w * (a ** 2).sum(1)).sum()
        if den < 1e-9:
            break
        theta = np.arctan2(num2, num)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        s = (num * np.cos(theta) + num2 * np.sin(theta)) / den
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        t = cd - s * (R @ cs)

        # 重加权（Huber-like）：残差大的降权
        pred = s * (src @ R.T) + t
        err = np.linalg.norm(pred - dst, axis=1)
        med = np.median(err)
        if med < 1e-6:
            break
        # 超过 2 倍中位的降到 0.1 权重
        w_new = np.where(err > 2.0 * med, 0.1, 1.0)
        if np.allclose(w_new, w):
            break
        w = w_new

    return s, R, t


def apply_sim(src, s, R, t):
    return s * (src @ R.T) + t


def smooth_in_segs(x, win, segs):
    """
    只在**段内**做滑动平均，不跨段混合。

    np.convolve(mode="same") 会把段外的 0 混进来，在段首尾造成跳变，
    反而增加了二阶差分（抖动）。这里按段独立平滑：
      - 段内：窗长 win 的滑动平均，边界用段内数据收缩
      - 段外：保持 0（残差在段外没有意义）
    """
    out = np.zeros_like(x)
    half = win // 2
    for s0, e0 in segs:
        seg = x[s0:e0+1].astype(np.float64)
        L = len(seg)
        if L == 0:
            continue
        if L < 3:
            out[s0:e0+1] = seg
            continue
        w = min(win, L)
        # 用累积和做滑动平均（段内边界收缩）
        csum = np.concatenate([[0.0], np.cumsum(seg)])
        res = np.empty(L)
        h = w // 2
        for i in range(L):
            lo = max(0, i - h)
            hi = min(L, i + h + 1)
            res[i] = (csum[hi] - csum[lo]) / (hi - lo)
        out[s0:e0+1] = res
    return out


# ============================================================================
# 主流程
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "P1v2_out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    _P = CFG.load_pipeline_config()
    cfg = _P["m3f_fusion"]
    conf_min = cfg["conf_min"]
    smooth_win_sec = cfg["smooth_window_sec"]
    swap_ratio = cfg["swap"]["swap_ratio"]
    do_swap = cfg["swap"]["enabled"]
    max_gap_sec = cfg.get("max_resid_gap_sec", 0.5)
    bone_len_tol = cfg.get("bone_len_tolerance", 0.30)
    overlap_sec = cfg.get("window_overlap_sec", 1.0)
    fps = _P["m0_preprocess"]["target_fps"]

    out = Path(a.out)
    (out / "skeleton").mkdir(parents=True, exist_ok=True)
    gem_dir = out / "gem"
    if not gem_dir.exists():
        print("  x 无 gem 目录")
        return

    only = [x for x in a.only.split(",") if x]
    results = {}

    for vid_dir in sorted(gem_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name
        if only and vid not in only:
            continue

        trk_p = out / "tracking" / f"{vid}.json"
        p2d_p = out / "pose2d" / f"{vid}.npz"
        if not (trk_p.exists() and p2d_p.exists()):
            continue

        import time as _time
        _t_vid = _time.time()
        tracking = json.load(open(trk_p, encoding="utf-8"))
        n = len(tracking)
        p2d = np.load(p2d_p)
        kpts2d, scores2d = p2d["kpts"], p2d["scores"]

        upstream = {"tracking.json": file_hash(trk_p),
                    "pose2d.npz": file_hash(p2d_p)}

        # ---- 收集 gem 块（带权重，用于重叠过渡） ----
        blocks = []
        for seg_p in sorted(vid_dir.glob("seg*.npz")):
            if seg_p.name.endswith("_meta.npz"):
                continue
            d = np.load(seg_p, allow_pickle=True)
            off = int(d["frame_offset"])
            proj = d["kp2d_proj"]
            blocks.append({
                "file": seg_p.name, "offset": off, "len": len(proj),
                "proj": proj,
                "camera": str(d["camera_mode"]) if "camera_mode" in d else "?",
                "trusted": bool(d["camera_trusted"]) if "camera_trusted" in d else True,
                "hash": file_hash(seg_p),
            })

        if not blocks:
            continue

        print(f"\n{'='*76}")
        print(f"  {vid}: {n} 帧, {len(blocks)} 块")
        for b in blocks:
            print(f"    块 {b['file']}: {b['offset']}-{b['offset']+b['len']-1} "
                  f"({b['len']}帧) 相机={b['camera']}")

        # ------------------------------------------------------------------
        # 第 1 步：**窗口重叠过渡**（击败边界撕裂的核心）
        # 相邻块重叠区，投影按线性权重融合
        # ------------------------------------------------------------------
        overlap_frames = int(overlap_sec * fps)
        j3d_proj = np.full((n, 77, 2), np.nan, dtype=np.float32)

        # 按 offset 排序
        blocks.sort(key=lambda b: b["offset"])

        # 每个像素位置收集所有覆盖它的块及其权重
        for i, b in enumerate(blocks):
            s, e = b["offset"], b["offset"] + b["len"] - 1
            # 与前后块的重叠
            w = np.ones(b["len"], dtype=np.float32)

            # 与前一块重叠：本块前半段权重从 0→1
            if i > 0:
                prev = blocks[i - 1]
                prev_e = prev["offset"] + prev["len"] - 1
                ov = prev_e - s + 1
                if ov > 0:
                    ov_len = min(ov, overlap_frames, b["len"])
                    for k in range(ov_len):
                        t = (k + 1) / (ov_len + 1)     # 0→1
                        w[k] = t

            # 与后一块重叠：本块后半段权重从 1→0
            if i < len(blocks) - 1:
                nxt = blocks[i + 1]
                ov = e - nxt["offset"] + 1
                if ov > 0:
                    ov_len = min(ov, overlap_frames, b["len"])
                    for k in range(ov_len):
                        idx = b["len"] - 1 - k
                        if idx >= 0:
                            t = (k + 1) / (ov_len + 1)   # 0→1（从末尾数）
                            w[idx] *= (1.0 - t)

            # 加权累加
            for k in range(b["len"]):
                gk = s + k
                if gk >= n:
                    break
                p = b["proj"][k]
                ww = w[k]
                if not np.isfinite(p).all():
                    continue
                cur = j3d_proj[gk]
                if np.isfinite(cur).all():
                    # 已有值：加权平均
                    j3d_proj[gk] = (cur * (1 - ww) + p * ww)
                else:
                    j3d_proj[gk] = p

        n_overlap = 0
        for i in range(1, len(blocks)):
            ov = (blocks[i-1]["offset"] + blocks[i-1]["len"] - 1
                  - blocks[i]["offset"] + 1)
            if ov > 0:
                n_overlap += 1
        if n_overlap:
            print(f"  窗口重叠过渡: {n_overlap} 处")

        # ---- 身高 ----
        body_h = np.array([estimate_body_height(j3d_proj, k) or np.nan
                           for k in range(n)])
        med_h = float(np.nanmedian(body_h)) if np.isfinite(body_h).any() else 0

        # ------------------------------------------------------------------
        # 第 2 步：判断 2D 可靠性
        #
        # 门限全部来自 configs（实践要求：时间连续 8%，不是我之前用的 25%）
        # ------------------------------------------------------------------
        _r = cfg.get("reliability", {})
        time_cont_ratio = _r.get("time_continuity_ratio", 0.08)
        two_model_agree = _r.get("two_model_agree_ratio", 0.05)
        gem_obs_conf_min = _r.get("gem_obs_conf_min", 0.3)

        # GEM-X 内部的 ViTPose 观测（用于「两个独立模型互相印证」）
        # gem/*.npz 里的 kp2d_obs: (T,77,3) = x, y, conf
        gem_obs = np.full((n, 77, 3), np.nan, dtype=np.float32)
        for b in blocks:
            d = np.load(vid_dir / b["file"], allow_pickle=True)
            if "kp2d_obs" not in d:
                continue
            obs = d["kp2d_obs"]
            off = b["offset"]
            L = len(obs)
            end = min(off + L, n)
            if end > off:
                gem_obs[off:end] = obs[:end - off]

        reliable = np.zeros((n, 26), dtype=bool)
        n_two_model_agree = 0
        n_two_model_reject = 0

        for hi in range(26):
            base = (scores2d[:, hi] >= conf_min) & np.isfinite(kpts2d[:, hi]).all(1)

            for k in range(n):
                if not base[k] or not np.isfinite(body_h[k]):
                    continue
                bh = body_h[k]
                # 时间连续：和前后 2 帧的位移不能突变
                jumps = []
                for kk in (k-1, k-2, k+1, k+2):
                    if 0 <= kk < n and base[kk]:
                        d = np.linalg.norm(kpts2d[k, hi] - kpts2d[kk, hi])
                        jumps.append(d / bh)
                if jumps and min(jumps) > time_cont_ratio:
                    continue
                reliable[k, hi] = True

        # ---- 「两个独立模型互相印证」（实践第 1 条） ----
        # 比较 RTMPose(pose2d) 和 GEM-X 内部 ViTPose(gem kp2d_obs)
        # 对**同一个关节**，相距 ≤ 5% 身高才算互相印证
        SOMA2H26_SINGLE = {}
        for hn, sn in JOINT_MAP.items():
            SOMA2H26_SINGLE.setdefault(sn, hn)

        for sn, hn in SOMA2H26_SINGLE.items():
            hi = H26[hn]
            si = S77[sn]
            for k in range(n):
                if not reliable[k, hi]:
                    continue
                if not np.isfinite(body_h[k]) or body_h[k] <= 0:
                    continue
                # GEM 的 ViTPose 观测
                o = gem_obs[k, si]
                if not (np.isfinite(o).all() and o[2] >= gem_obs_conf_min):
                    # GEM 这一帧没有可靠观测 → 无法印证，保留 RTMPose 结果
                    continue
                d = np.linalg.norm(kpts2d[k, hi] - o[:2]) / body_h[k]
                if d <= two_model_agree:
                    n_two_model_agree += 1
                else:
                    # 两个独立模型分歧太大 → 不可靠
                    reliable[k, hi] = False
                    n_two_model_reject += 1

        if n_two_model_agree or n_two_model_reject:
            print(f"  两模型印证: 通过 {n_two_model_agree}, "
                  f"分歧剔除 {n_two_model_reject}")

        # ---- 左右互换 ----
        n_swap = 0
        if do_swap:
            for ka, kb in SWAP_PAIRS:
                ia, ib = S77.get(ka), S77.get(kb)
                if ia is None or ib is None:
                    continue
                ha_l, hb_l = SOMA2H26.get(ka, []), SOMA2H26.get(kb, [])
                if not ha_l or not hb_l:
                    continue
                ha, hb = H26[ha_l[0]], H26[hb_l[0]]
                for k in range(n):
                    if not (reliable[k, ha] and reliable[k, hb]):
                        continue
                    qa, qb = j3d_proj[k, ia], j3d_proj[k, ib]
                    if not (np.isfinite(qa).all() and np.isfinite(qb).all()):
                        continue
                    pa, pb = kpts2d[k, ha], kpts2d[k, hb]
                    d_n = np.linalg.norm(pa-qa) + np.linalg.norm(pb-qb)
                    d_s = np.linalg.norm(pa-qb) + np.linalg.norm(pb-qa)
                    if d_s < d_n * swap_ratio:
                        reliable[k, ha] = reliable[k, hb] = False
                        n_swap += 1

        # ------------------------------------------------------------------
        # 第 3 步：**整体对齐**（平移+缩放+旋转）
        # ------------------------------------------------------------------
        glob_align = np.zeros((n, 3), dtype=np.float32)   # s, theta, (tx,ty) 分开存
        glob_t = np.zeros((n, 2), dtype=np.float32)
        glob_s = np.ones(n, dtype=np.float32)
        glob_th = np.zeros(n, dtype=np.float32)
        proj_aligned = j3d_proj.copy()

        for k in range(n):
            # 收集该帧可靠 2D 和对应投影
            src, dst, wts = [], [], []
            for sn, h_list in SOMA2H26.items():
                si = S77[sn]
                if not np.isfinite(j3d_proj[k, si]).all():
                    continue
                for hn in h_list:
                    hi = H26[hn]
                    if reliable[k, hi]:
                        src.append(j3d_proj[k, si])
                        dst.append(kpts2d[k, hi])
                        wts.append(scores2d[k, hi])
                        break
            if len(src) < 3:
                continue

            s, R, t = rigid_fit_2d(np.array(src), np.array(dst), np.array(wts))
            # 合理性检查：缩放不能太离谱
            if 0.6 < s < 1.6 and abs(np.arctan2(R[1, 0], R[0, 0])) < np.deg2rad(35):
                glob_s[k] = s
                glob_th[k] = np.arctan2(R[1, 0], R[0, 0])
                glob_t[k] = t
                proj_aligned[k] = apply_sim(j3d_proj[k], s, R, t)

        # ------------------------------------------------------------------
        # 第 4 步：逐关节残差（相对**对齐后**的投影）
        # ------------------------------------------------------------------
        resid = np.full((n, 77, 2), np.nan, dtype=np.float32)
        for sn, h_list in SOMA2H26.items():
            si = S77[sn]
            for k in range(n):
                if not np.isfinite(proj_aligned[k, si]).all():
                    continue
                best_hi, best_conf = None, -1
                for hn in h_list:
                    hi = H26[hn]
                    if reliable[k, hi] and scores2d[k, hi] > best_conf:
                        best_hi, best_conf = hi, scores2d[k, hi]
                if best_hi is None:
                    continue
                resid[k, si] = kpts2d[k, best_hi] - proj_aligned[k, si]

        # ------------------------------------------------------------------
        # 第 5 步：残差连续化（段首尾从段内往外过渡）
        # ------------------------------------------------------------------
        win = max(3, int(smooth_win_sec * fps))
        long_lpf_sec = cfg.get("long_lpf_sec", 0.8)
        win_long = max(win, int(long_lpf_sec * fps))
        max_gap = int(max_gap_sec * fps)

        # 找骨架段（用于段首尾过渡）
        segs = []
        k = 0
        while k < n:
            if tracking[k]["status"] == "skeleton":
                s0 = k
                while k < n and tracking[k]["status"] == "skeleton":
                    k += 1
                segs.append((s0, k-1))
            else:
                k += 1

        def in_seg(idx):
            for s0, e0 in segs:
                if s0 <= idx <= e0:
                    return True
            return False

        resid_cont = np.zeros((n, 77, 2), dtype=np.float32)
        valid_resid = np.zeros((n, 77), dtype=bool)

        for si in range(77):
            for c in range(2):
                v = resid[:, si, c]
                mask = np.isfinite(v)
                if mask.sum() < 3:
                    continue
                idx = np.where(mask)[0]
                filled = np.zeros(n, dtype=np.float32)
                for i in range(len(idx)):
                    filled[idx[i]] = v[idx[i]]

                # 内部缺口
                for i in range(len(idx) - 1):
                    a_i, b_i = idx[i], idx[i+1]
                    gap = b_i - a_i - 1
                    if gap == 0:
                        continue
                    if gap <= max_gap:
                        for kk in range(a_i+1, b_i):
                            t = (kk - a_i) / (b_i - a_i)
                            filled[kk] = v[a_i]*(1-t) + v[b_i]*t
                    else:
                        half = max(1, gap // 2)
                        for kk in range(a_i+1, a_i+1+half):
                            t = (kk - a_i) / (half + 1)
                            filled[kk] = v[a_i]*(1-t)
                        for kk in range(b_i-half, b_i):
                            t = (b_i - kk) / (half + 1)
                            filled[kk] = v[b_i]*(1-t)

                # 段首尾：从**段内**往外过渡（不是从最近端点）
                for s0, e0 in segs:
                    # 段首之前（段内第一帧往前，但不越过上一段）
                    prev_end = max([e for _, e in segs if e < s0], default=-1)
                    before = [i for i in idx if s0 <= i <= e0]
                    if not before:
                        continue
                    first_valid = before[0]
                    # 段首到第一个有效帧之间 → 从 0 过渡到 v
                    for kk in range(s0, first_valid):
                        t = (kk - s0 + 1) / (first_valid - s0 + 1)
                        filled[kk] = v[first_valid] * t
                    # 段尾之后
                    last_valid = before[-1]
                    next_start = min([s for s, _ in segs if s > e0], default=n)
                    for kk in range(last_valid+1, e0+1):
                        t = (e0 - kk + 1) / (e0 - last_valid + 1)
                        filled[kk] = v[last_valid] * t

                # 第 1 层平滑（短窗 0.2s）
                if n > win:
                    sm = smooth_in_segs(filled, win, segs)
                    for kk in range(n):
                        if in_seg(kk):
                            filled[kk] = sm[kk]
                # 第 2 层长窗低通（抑制抖动的高频分量，保留低频贴合）
                if n > win_long:
                    sm2 = smooth_in_segs(filled, win_long, segs)
                    for kk in range(n):
                        if in_seg(kk):
                            filled[kk] = sm2[kk]

                resid_cont[:, si, c] = filled

            valid_resid[:, si] = np.isfinite(resid[:, si]).all(axis=1)

        # ------------------------------------------------------------------
        # 第 6 步：最终骨架 = 对齐后投影 + 残差，然后**骨长检查**
        #
        # 两层检查：
        #   a) 校正后骨长 vs 投影骨长（残差是否撕裂）
        #   b) **投影本身骨长是否异常**（GEM-X 自身的错误，如 ToeEnd 乱飞）
        #      后者无法通过融合修复，标记为 bad_frame
        # ------------------------------------------------------------------
        final = proj_aligned + resid_cont

        BONE_PAIRS = [
            ("Hips", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Chest"),
            ("Chest", "Neck1"), ("Neck1", "Neck2"), ("Neck2", "Head"),
            ("Head", "HeadEnd"),
            ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
            ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
            ("Chest", "RightShoulder"), ("RightShoulder", "RightArm"),
            ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
            ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"),
            ("LeftShin", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
            ("LeftToeBase", "LeftToeEnd"),
            ("Hips", "RightLeg"), ("RightLeg", "RightShin"),
            ("RightShin", "RightFoot"), ("RightFoot", "RightToeBase"),
            ("RightToeBase", "RightToeEnd"),
        ]
        name2chain = {}
        for ci, chain in enumerate(LIMB_CHAINS):
            for nm in chain:
                name2chain[nm] = ci

        # 骨长合理性上限（相对框高）——超过说明该骨本身错误
        BONE_MAX_RATIO = 0.60

        n_rollback = 0
        bad_frames = set()

        for k in range(n):
            if tracking[k]["status"] != "skeleton":
                continue
            bh = tracking[k]["box_h"]
            if bh <= 0:
                continue

            bad_chains = set()
            bad_bones = set()
            proj_bad = False

            for ja, jb in BONE_PAIRS:
                ia, ib = S77.get(ja), S77.get(jb)
                if ia is None or ib is None:
                    continue
                pp_a, pp_b = proj_aligned[k, ia], proj_aligned[k, ib]
                if not (np.isfinite(pp_a).all() and np.isfinite(pp_b).all()):
                    continue
                Lp = np.linalg.norm(pp_a - pp_b)

                # (b) 投影本身骨长异常 → 该帧标质量问题
                if Lp / bh > BONE_MAX_RATIO:
                    proj_bad = True

                # (a) 校正后 vs 投影
                pf_a, pf_b = final[k, ia], final[k, ib]
                if not (np.isfinite(pf_a).all() and np.isfinite(pf_b).all()):
                    continue
                Lf = np.linalg.norm(pf_a - pf_b)
                if Lp < 1e-3:
                    continue
                # 容差按骨骼画面长度分级：
                #   长骨（> 10% 框高）：严格，用 bone_len_tol
                #   短骨（颈部、脚趾等）：2D 投影受透视影响大（人弯腰/转身时
                #   颈部 2D 距离会自然缩短 50%），用宽松 3 倍容差。
                #   短骨即使差 50% 在画面上也只有几像素，视觉影响小。
                rel_len = Lp / bh if bh > 0 else 1.0
                tol = bone_len_tol if rel_len > 0.10 else bone_len_tol * 3.0
                if abs(Lf - Lp) / Lp > tol:
                    bad_bones.add((ja, jb))

            # 只退回**超差骨头**的端点，不退回整条链
            # （整链退回会把一次超差扩大到 4-5 个关节，
            #   实测导致 12 个肢体关节的校正几乎全灭）
            for ja, jb in bad_bones:
                for nm in (ja, jb):
                    si = S77.get(nm)
                    if si is not None:
                        final[k, si] = proj_aligned[k, si]
                n_rollback += 1

            # 投影本身就坏的帧：无法修复，标记为质量差
            if proj_bad:
                bad_frames.add(k)

        if n_rollback:
            print(f"  骨长检查回退: {n_rollback} 次")
        if bad_frames:
            print(f"  ⚠ 投影自身异常帧: {len(bad_frames)} 帧 "
                  f"({sorted(bad_frames)[:8]}...)")

        # ------------------------------------------------------------------
        # 第 7 步：**逐关节抖动门控**（通用规则）
        #
        # 验收结果证明：融合会引入高频噪声（每个关节抖动 +2~7%）。
        # 如果某关节融合后的加速度 p95 明显高于原始投影，
        # 该关节退回原始投影（不加残差）。
        #
        # 这是通用规则，参数在 pipeline.json，无视频编号分支。
        # ------------------------------------------------------------------
        jg = cfg.get("jitter_gate", {})
        jg_on = jg.get("enabled", False)   # 默认关闭：整关节退回会把校正也关掉
        jg_ratio = jg.get("max_jitter_ratio", 1.0)   # 允许的最大倍数

        # 验收口径的 12 个共同关节
        SK12_SOMA = ["LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
                     "LeftHand", "RightHand", "LeftLeg", "RightLeg",
                     "LeftShin", "RightShin", "LeftFoot", "RightFoot"]
        GEM12_IDX = [12, 40, 13, 41, 14, 42, 67, 72, 68, 73, 69, 74]

        n_joint_rollback = 0
        if jg_on:
            # 构造验收口径的 12 关节序列（融合 vs 原始）
            F12 = np.full((n, 12, 2), np.nan, dtype=np.float32)
            G12 = np.full((n, 12, 2), np.nan, dtype=np.float32)
            for k in range(n):
                for i, nm in enumerate(SK12_SOMA):
                    si = S77.get(nm)
                    if si is None:
                        continue
                    if np.isfinite(final[k, si]).all():
                        F12[k, i] = final[k, si]
                    if np.isfinite(j3d_proj[k, GEM12_IDX[i]]).all():
                        G12[k, i] = j3d_proj[k, GEM12_IDX[i]]

            isk = np.array([tracking[k]["status"] == "skeleton"
                            for k in range(n)])
            common = isk & np.isfinite(F12).all((1, 2)) & np.isfinite(G12).all((1, 2))

            def p95_acc(X, j):
                a = []
                for t in range(1, n-1):
                    if not (common[t-1] and common[t] and common[t+1]):
                        continue
                    v = np.linalg.norm(X[t+1, j] - 2*X[t, j] + X[t-1, j])
                    if np.isfinite(v):
                        a.append(v)
                return float(np.percentile(a, 95)) if len(a) >= 10 else None

            if common.sum() >= 10:
                for j, nm in enumerate(SK12_SOMA):
                    p_f = p95_acc(F12, j)
                    p_g = p95_acc(G12, j)
                    if p_f is None or p_g is None or p_g < 1e-6:
                        continue
                    if p_f > p_g * jg_ratio:
                        # 该关节退回原始投影
                        si = S77.get(nm)
                        if si is not None:
                            ok_mask = np.isfinite(j3d_proj[:, si]).all(axis=1)
                            final[ok_mask, si] = j3d_proj[ok_mask, si]
                            n_joint_rollback += 1
                if n_joint_rollback:
                    print(f"  抖动门控: {n_joint_rollback} 个关节退回原始投影")

        # ------------------------------------------------------------------
        # 第 6c 步：**轻量整体平滑**（和 v2 同力度，口径公平）
        #
        # 验收决定（2026-11-20）：
        #   不选「整体大平滑」（会抹掉刚恢复的校正）。
        #   改成「和 v2 一样力度的轻量平滑」：30fps 下 3 帧加权平均。
        #   v2 输出前做过同样的事，这也是它 spike_pct 特别低的原因。
        #
        # 关键：平滑**整个骨架**（final），不只是残差。
        #   因为 spike 来自 GEM 原始投影的高频部分，
        #   之前的「只平滑残差」压不掉它。
        #
        # 3 帧加权平均核用 [0.25, 0.5, 0.25]（高斯近似），
        # 是标准的三点平滑。
        # ------------------------------------------------------------------
        ls_cfg = cfg.get("light_smooth", {})
        if ls_cfg.get("enabled", True):
            lw = int(ls_cfg.get("window_frames", 3))
            if lw >= 3 and lw % 2 == 1:
                # 三点核权重（中心 0.5，两侧各 0.25）
                kern = np.array([0.25, 0.5, 0.25])
                # 更长的窗口用二项式核
                if lw > 3:
                    from math import comb
                    kern = np.array([comb(lw - 1, i) for i in range(lw)],
                                    dtype=float)
                    kern /= kern.sum()

                half = lw // 2
                n_sm = 0
                for si in range(77):
                    for c in range(2):
                        v = final[:, si, c]
                        ok = np.isfinite(v)
                        if ok.sum() < lw:
                            continue
                        # 段内平滑（不跨段）
                        orig = v.copy()
                        for s0, e0 in segs:
                            L = e0 - s0 + 1
                            if L < lw:
                                continue
                            seg = v[s0:e0+1]
                            seg_ok = np.isfinite(seg)
                            if seg_ok.sum() < lw:
                                continue
                            # 用插值填满段内
                            idx_seg = np.where(seg_ok)[0]
                            filled_seg = np.interp(np.arange(L), idx_seg,
                                                   seg[idx_seg])
                            sm = np.convolve(filled_seg, kern, mode="same")
                            # 边界修正（用段内数据收缩）
                            for i in range(min(half, L)):
                                lo = max(0, i - half)
                                hi = min(L, i + half + 1)
                                w = kern[lw//2 - (i - lo): lw//2 + (hi - i)]
                                w = w / w.sum() if w.sum() > 0 else w
                                sm[i] = np.sum(filled_seg[lo:hi] * w)
                            for i in range(max(0, L - half), L):
                                lo = max(0, i - half)
                                hi = min(L, i + half + 1)
                                w = kern[lw//2 - (i - lo): lw//2 + (hi - i)]
                                w = w / w.sum() if w.sum() > 0 else w
                                sm[i] = np.sum(filled_seg[lo:hi] * w)
                            final[s0:e0+1, si, c] = sm
                            n_sm += int(seg_ok.sum())
                print(f"  轻量整体平滑: {lw} 帧加权 ({n_sm} 点)")

        # ------------------------------------------------------------------
        # 第 8 步：**通用质量门**（方案 2.1，放在最后，检查最终骨架）
        #   融合后任何一帧只要有一根骨头长于当帧框高的 max_bone_ratio，
        #   就降为 track_only 并记录 bad_projection。
        #   适用于全部视频全部帧，参数在 pipeline.json，无视频编号分支。
        # ------------------------------------------------------------------
        qg = cfg.get("quality_gate", {})
        qg_on = qg.get("enabled", True)
        qg_max_ratio = qg.get("max_bone_ratio", 0.60)
        qg_warn = qg.get("warn_ratio_of_skeleton", 0.01)
        downgraded = []

        # 质量门放在抖动门控**之后**：抖动门控会把关节退回原始投影，
        # 而原始投影本身在某些帧就撕裂，所以必须检查**最终**骨架。
        if qg_on:
            for k in range(n):
                if tracking[k]["status"] != "skeleton":
                    continue
                bh = tracking[k]["box_h"]
                if bh <= 0:
                    continue
                tear = False
                for ja, jb in BONE_PAIRS:
                    ia, ib = S77.get(ja), S77.get(jb)
                    if ia is None or ib is None:
                        continue
                    pa, pb = final[k, ia], final[k, ib]
                    if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                        continue
                    if np.linalg.norm(pa - pb) / bh > qg_max_ratio:
                        tear = True
                        break
                if tear:
                    downgraded.append(k)

            # 真正降级（tracking 状态也改，保持上下游一致）
            for k in downgraded:
                tracking[k]["status"] = "track_only"
                frames_out_status = qg.get("downgrade_to", "track_only")
                if k < len(tracking):
                    tracking[k]["downgraded_from"] = "skeleton"
                    tracking[k]["downgrade_reason"] = qg.get("reason", "bad_projection")

            n_skel_before = sum(1 for k in range(n)
                                if tracking[k]["status"] == "skeleton") + len(downgraded)
            n_downgraded = len(downgraded)
            ratio_dg = n_downgraded / n_skel_before if n_skel_before else 0
            if n_downgraded:
                print(f"  质量门降级: {n_downgraded} 帧 "
                      f"({ratio_dg*100:.2f}% of 骨架帧)")
                if ratio_dg > qg_warn:
                    print(f"  ⚠ 降级比例 {ratio_dg*100:.2f}% 超过告警阈值 "
                          f"{qg_warn*100:.0f}% → 上游可能有问题，需查根因")

        # ------------------------------------------------------------------
        # 第 6b 步：**最终残差平滑**（关键修复）
        #
        # 实测证明：final = proj_aligned + resid_cont 里，
        # 逐帧独立拟合的 proj_aligned 自带高频噪声，
        # 前面的「平滑 resid」无法消除它。
        #
        # 离线验证（p1_test_smooth_windows.py）：
        #   对 (final - j3d_proj) 这个最终残差再做一次平滑，
        #   0.2s 窗口就能把抖动从 6.64% 降到 2.92%（GEM 原始 2.69%）。
        #
        # 所以在最终的 fused 骨架上直接平滑一次。
        # ------------------------------------------------------------------
        fs_cfg = cfg.get("final_smooth", {})
        if fs_cfg.get("enabled", True):
            fs_sec = fs_cfg.get("window_sec", 0.2)
            fw = max(3, int(fs_sec * fps))
            R_final = final - j3d_proj          # 最终残差（含对齐 + 逐关节）
            n_smoothed = 0
            for si in range(77):
                for c in range(2):
                    v = R_final[:, si, c]
                    ok = np.isfinite(v)
                    if ok.sum() < 5:
                        continue
                    # 段内平滑（按骨架段）
                    sm = smooth_in_segs(np.where(ok, v, 0.0), fw, segs)
                    # 只在有值的帧上应用（段外/无效帧保持）
                    use = ok & np.array([tracking[kk]["status"] == "skeleton"
                                         for kk in range(n)])
                    R_final[use, si, c] = sm[use]
                    n_smoothed += int(use.sum())
            final = j3d_proj + R_final
            print(f"  最终残差平滑: {fs_sec}s 窗口 ({n_smoothed} 点)")

        # ---- 统计 & 输出 ----
        n_fused_f = 0
        n_total_f = 0
        for k in range(n):
            if tracking[k]["status"] != "skeleton":
                continue
            for sn in FUSABLE:
                si = S77[sn]
                if np.isfinite(final[k, si]).all():
                    n_total_f += 1
                    if valid_resid[k, si]:
                        n_fused_f += 1

        frames_out = []
        for k in range(n):
            st = tracking[k]["status"]
            if st != "skeleton":
                frames_out.append({"i": k, "t": round(k/fps, 3), "status": st,
                                   "joints": None, "sources": None})
                continue
            joints, sources = {}, {}
            for name in BODY_JOINTS:
                si = S77[name]
                p = final[k, si]
                if np.isfinite(p).all():
                    joints[name] = [round(float(p[0]), 1), round(float(p[1]), 1)]
                    sources[name] = ("fused" if valid_resid[k, si] else "gem")
                else:
                    joints[name] = None
                    sources[name] = "none"
            frames_out.append({"i": k, "t": round(k/fps, 3), "status": st,
                               "joints": joints, "sources": sources,
                               "quality": ("bad_projection"
                                           if k in bad_frames else "ok")})

        skel = {
            "video": vid, "frames": n, "fps": fps,
            "joint_names": BODY_JOINTS,
            "joint_map": JOINT_MAP,
            "config": cfg,
            "body_height_px_median": round(med_h, 1),
            "gem_blocks": [{"file": b["file"], "offset": b["offset"],
                            "len": b["len"], "camera": b["camera"],
                            "trusted": b["trusted"], "hash": b["hash"]}
                           for b in blocks],
            "camera_trusted": all(b["trusted"] for b in blocks),
            "upstream": upstream,
            "window_overlap_sec": overlap_sec,
            "frames_data": frames_out,
            "stats": {
                "skeleton_frames": sum(1 for f in frames_out
                                       if f["status"] == "skeleton"),
                "fused_points_fusable": int(n_fused_f),
                "fusable_total_points": int(n_total_f),
                "fused_ratio_fusable": round(n_fused_f/n_total_f*100, 1) if n_total_f else 0.0,
                "swap_rejected": int(n_swap),
                "bone_rollback": int(n_rollback),
                "bad_projection_frames": sorted(int(x) for x in bad_frames),
                "n_bad_projection": len(bad_frames),
                "quality_gate": {
                    "enabled": bool(qg_on),
                    "max_bone_ratio": qg_max_ratio,
                    "downgraded_frames": [int(x) for x in downgraded],
                    "n_downgraded": len(downgraded),
                },
                "n_fusable_joints": len(FUSABLE),
            },
        }
        (out / "skeleton" / f"{vid}.json").write_text(
            json.dumps(skel, ensure_ascii=False), encoding="utf-8")

        # 降级后把 tracking.json 也回写，保证上下游一致（实践第 3 条要求）
        if downgraded:
            trk_new = out / "tracking" / f"{vid}.json"
            trk_new.write_text(json.dumps(tracking, ensure_ascii=False),
                               encoding="utf-8")
            # segments 也要重算（去掉降级的帧）
            seg_doc_p = out / "segments" / f"{vid}.json"
            if seg_doc_p.exists():
                sd = json.load(open(seg_doc_p, encoding="utf-8"))
                new_segs = []
                k = 0
                while k < n:
                    if tracking[k]["status"] == "skeleton":
                        s0 = k
                        while k < n and tracking[k]["status"] == "skeleton":
                            k += 1
                        if k - s0 >= 15:
                            new_segs.append({"start": s0, "end": k-1,
                                             "frames": k-s0,
                                             "t_start": round(s0/fps, 3),
                                             "t_end": round((k-1)/fps, 3)})
                    else:
                        k += 1
                sd["segments"] = new_segs
                sd.setdefault("stats", {})["skeleton"] = sum(
                    1 for f2 in tracking if f2["status"] == "skeleton")
                sd["stats"]["n_segments"] = len(new_segs)
                seg_doc_p.write_text(json.dumps(sd, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

        st = skel["stats"]
        print(f"  骨架帧: {st['skeleton_frames']}/{n}, "
              f"融合率 {st['fused_ratio_fusable']}%, "
              f"骨长回退 {st['bone_rollback']}, "
              f"投影异常 {st['n_bad_projection']} 帧")
        st["sec"] = round(_time.time() - _t_vid, 1)
        results[vid] = st

    (out / "fusion_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {len(results)} 条")


if __name__ == "__main__":
    main()
