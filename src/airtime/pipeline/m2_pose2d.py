#!/usr/bin/env python3
"""
P1 T1.3 逐帧 2D（RTMPose-X Halpe26）

要求：
  - 只在 skeleton 段上跑
  - 用 T1.2 的框
  - 含倒立时旋转画面重新识别（复用 A 版本 v2 的思路）
  - **不做平滑、不补点**（留给融合）

输出：pose2d.npz
  kpts   [T,26,2]  整段帧序，无结果填 NaN
  scores [T,26]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .. import det_deterministic as DET

HALPE26_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
               "onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-"
               "7fb6e239_20230606.zip")

# Halpe26 关节名（与 A 版本一致）
NAMES26 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle",
           "head", "neck", "hip",
           "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
           "left_heel", "right_heel"]

# 倒立重识别参数（A 版本 v2）
CONF_MIN = 0.55
TILT_DEG = 50.0


def torso_tilt_deg(kpts):
    """躯干偏离竖直角度（用 hip 中点和 neck）"""
    try:
        hip = kpts[19]   # hip (骨盆中点)
        neck = kpts[18]  # neck
        if not (np.isfinite(hip).all() and np.isfinite(neck).all()):
            return None
        v = neck - hip
        if np.linalg.norm(v) < 1e-3:
            return None
        # 与竖直(向上)的夹角
        ang = np.degrees(np.arccos(np.clip(-v[1] / np.linalg.norm(v), -1, 1)))
        return float(ang)
    except Exception:
        return None


def rotate_image_and_box(im, box, angle):
    """旋转图像和框（绕框中心）"""
    h, w = im.shape[:2]
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(114, 114, 114))

    # 框旋转后取外接矩形
    pts = np.array([[box[0], box[1]], [box[2], box[1]],
                    [box[2], box[3]], [box[0], box[3]]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    new_pts = (M @ pts_h.T).T
    nb = [new_pts[:, 0].min(), new_pts[:, 1].min(),
          new_pts[:, 0].max(), new_pts[:, 1].max()]
    nb = [max(0, nb[0]), max(0, nb[1]),
          min(w-1, nb[2]), min(h-1, nb[3])]
    return rot, nb, M


def inverse_points(kpts, M_inv):
    """把旋转后识别的点映射回原图"""
    out = kpts.copy()
    ones = np.ones((len(kpts), 1), dtype=np.float32)
    pts_h = np.hstack([kpts.astype(np.float32), ones])
    new_pts = (M_inv @ pts_h.T).T
    out[:, 0] = new_pts[:, 0]
    out[:, 1] = new_pts[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="")
    ap.add_argument("--track", default="")
    ap.add_argument("--out", default=str(Path.home() / "P1v2_out"))
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if not a.pkg:
        a.pkg = a.out
    if not a.track:
        a.track = a.out

    from rtmlib import RTMPose

    # 确定性：必须在建 RTMPose 之前打补丁（理由同 M1，见 p1_t12_track.py）
    # DET 已在模块顶部导入
    DET.patch_rtmlib(verbose=False)

    print("加载 RTMPose-X Halpe26...", flush=True)
    pose = RTMPose(HALPE26_URL, model_input_size=(288, 384),
                   backend="onnxruntime", device=a.device)
    print("  ✓ 加载完成", flush=True)

    pkg = Path(a.pkg)
    trk_dir = Path(a.track) / "tracking"
    out = Path(a.out)
    (out / "pose2d").mkdir(parents=True, exist_ok=True)

    index = json.load(open(pkg / "index.json", encoding="utf-8"))
    only = [x for x in a.only.split(",") if x]

    results = {}

    for vid in sorted(index):
        if only and vid not in only:
            continue

        video = pkg / "clips" / f"{vid}.mp4"
        trk_p = trk_dir / f"{vid}.json"
        if not video.exists() or not trk_p.exists():
            print(f"  x {vid}: 缺视频或跟踪", flush=True)
            continue

        tracking = json.load(open(trk_p, encoding="utf-8"))
        n = len(tracking)

        print(f"\n{'='*70}", flush=True)
        print(f"  {vid}: {n} 帧", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        cap = cv2.VideoCapture(str(video))

        kpts_all = np.full((n, 26, 2), np.nan, dtype=np.float32)
        scores_all = np.full((n, 26), np.nan, dtype=np.float32)

        n_skel = 0
        n_rot_retry = 0
        prev_tilt = None

        for k in range(n):
            ok, im = cap.read()
            if not ok:
                break

            fr = tracking[k]
            if fr["status"] != "skeleton" or fr["bbox"] is None:
                continue

            box = fr["bbox"]
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue

            # 第一次识别
            kp, sc = pose(im, bboxes=[box])
            kp, sc = kp[0], sc[0]

            # 倒立/大幅倾斜时旋转重识别
            tilt = torso_tilt_deg(kp)
            need_retry = False
            if tilt is not None and tilt > TILT_DEG:
                need_retry = True
            elif np.nanmean(sc[:17]) < CONF_MIN:
                need_retry = True
            elif prev_tilt is not None and prev_tilt > TILT_DEG:
                need_retry = True

            if need_retry:
                best_kp, best_sc, best_conf = kp, sc, np.nanmean(sc[:17])

                # 尝试几种角度
                angles = []
                if tilt is not None:
                    # 把躯干转正的角度
                    angles.append(-(90 - tilt) if tilt < 180 else 0)
                angles.extend([90, 180, 270])

                for ang in angles:
                    if abs(ang) < 5:
                        continue
                    try:
                        rot_im, rot_box, M = rotate_image_and_box(im, box, ang)
                        kp2, sc2 = pose(rot_im, bboxes=[rot_box])
                        kp2, sc2 = kp2[0], sc2[0]
                        conf2 = np.nanmean(sc2[:17])
                        if conf2 > best_conf + 0.03:
                            M_inv = cv2.invertAffineTransform(M)
                            kp2 = inverse_points(kp2, M_inv)
                            best_kp, best_sc, best_conf = kp2, sc2, conf2
                    except Exception:
                        continue

                if best_conf > np.nanmean(sc[:17]):
                    kp, sc = best_kp, best_sc
                    n_rot_retry += 1

            kpts_all[k] = kp
            scores_all[k] = sc
            n_skel += 1
            prev_tilt = tilt if tilt is not None else prev_tilt

        cap.release()

        np.savez_compressed(out / "pose2d" / f"{vid}.npz",
                            kpts=kpts_all, scores=scores_all,
                            names=np.array(NAMES26))

        print(f"  识别 {n_skel} 帧, 旋转重识别 {n_rot_retry} 次 "
              f"({time.time()-t0:.0f}s)", flush=True)

        # 统计置信度
        valid = np.isfinite(scores_all[:, :17]).all(axis=1)
        mean_conf = float(np.nanmean(scores_all[valid, :17])) if valid.any() else 0
        print(f"  平均置信度: {mean_conf:.3f} ({valid.sum()} 帧有效)", flush=True)

        results[vid] = {
            "frames": n, "skeleton_frames": n_skel,
            "rot_retry": n_rot_retry,
            "mean_conf": round(mean_conf, 4),
            "sec": round(time.time()-t0, 1),
        }

    (out / "pose2d_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  完成 {len(results)} 条")
    print(f"{'='*70}")
    print(f"\n{'视频':<9} {'帧':>4} {'骨架帧':>7} {'重识别':>7} {'置信度':>7}", flush=True)
    print("-" * 40, flush=True)
    for vid, s in results.items():
        print(f"{vid:<9} {s['frames']:>4} {s['skeleton_frames']:>7} "
              f"{s['rot_retry']:>7} {s['mean_conf']:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
