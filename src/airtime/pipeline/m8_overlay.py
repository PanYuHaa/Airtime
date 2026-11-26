#!/usr/bin/env python3
"""
骨架叠加视频（从 skeleton.json 画，不画 kp2d_proj / kp2d_obs）

样式按 configs/style.json：
  - 左白 #FFFFFF、右冷灰 #A7B0BC
  - 38% 黑色描边
  - gem 来源的关节 60% 透明度
  - track_only 帧只画框中心小点
"""
import argparse
import json
import time
from pathlib import Path

import cv2

# SOMA 骨骼连线（26 关节，**大写名，和 skeleton.json 的 joint_names 一致**）
# 2026-11-20 修复：原版用小写名（hips/spine1/l_arm…），
# 但 skeleton.json 输出的是 SOMA 大写名（Hips/Spine1/LeftArm…），
# 导致 joints.get() 全部返回 None，一条线都没画出来（参考验收发现）。
BONES = [
    ("Hips", "Spine1", "center"), ("Spine1", "Spine2", "center"),
    ("Spine2", "Chest", "center"), ("Chest", "Neck1", "center"),
    ("Neck1", "Neck2", "center"), ("Neck2", "Head", "center"),
    ("Head", "HeadEnd", "center"),
    ("Chest", "LeftShoulder", "center"),
    ("Chest", "RightShoulder", "center"),
    ("LeftShoulder", "LeftArm", "left"), ("LeftArm", "LeftForeArm", "left"),
    ("LeftForeArm", "LeftHand", "left"),
    ("RightShoulder", "RightArm", "right"),
    ("RightArm", "RightForeArm", "right"),
    ("RightForeArm", "RightHand", "right"),
    ("Hips", "LeftLeg", "left"), ("Hips", "RightLeg", "right"),
    ("LeftLeg", "LeftShin", "left"), ("LeftShin", "LeftFoot", "left"),
    ("LeftFoot", "LeftToeBase", "left"),
    ("LeftToeBase", "LeftToeEnd", "left"),
    ("RightLeg", "RightShin", "right"), ("RightShin", "RightFoot", "right"),
    ("RightFoot", "RightToeBase", "right"),
    ("RightToeBase", "RightToeEnd", "right"),
]

DRAW_JOINTS = set()
for a, b, _ in BONES:
    DRAW_JOINTS.add(a)
    DRAW_JOINTS.add(b)

# style.json（BGR）
C_LEFT = (255, 255, 255)       # #FFFFFF
C_RIGHT = (188, 176, 167)      # #A7B0BC
C_CENTER = (255, 255, 255)
C_HALO = (0, 0, 0)
C_TRACK = (10, 214, 255)       # #FFD60A
GEM_ALPHA = 0.60

# 画线用的抗锯齿缩放基准（style.json 的 720p 基准）
STYLE_REF_H = 720.0


def blend_color(base, alpha, bg):
    return tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(base, bg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="")
    ap.add_argument("--out", default=str(Path.home() / "output_out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not a.pkg:
        a.pkg = a.out

    pkg = Path(a.pkg)
    out = Path(a.out)
    (out / "render").mkdir(parents=True, exist_ok=True)

    index = json.load(open(pkg / "index.json", encoding="utf-8"))
    only = [x for x in a.only.split(",") if x]

    results = {}

    for vid in sorted(index):
        if only and vid not in only:
            continue

        _t_vid = time.time()
        skel_p = out / "skeleton" / f"{vid}.json"
        trk_p = out / "tracking" / f"{vid}.json"
        video = pkg / "clips" / f"{vid}.mp4"

        if not (skel_p.exists() and trk_p.exists() and video.exists()):
            print(f"  x {vid}: 缺输入")
            continue

        skel = json.load(open(skel_p, encoding="utf-8"))
        tracking = json.load(open(trk_p, encoding="utf-8"))
        frames_data = skel["frames_data"]
        n = len(frames_data)

        cap = cv2.VideoCapture(str(video))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        lw = max(2, int(round(2.4 * H / 720)))
        lw_halo = lw + max(2, int(round(2.0 * H / 720)))
        jr = max(2, int(round(3.0 * H / 720)))

        dst = out / "render" / vid
        dst.mkdir(parents=True, exist_ok=True)
        tmp = dst / "_tmp.mp4"
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

        n_skel_drawn = 0
        n_lines_drawn = 0
        for k in range(n):
            ok, img = cap.read()
            if not ok:
                break

            fd = frames_data[k]
            st = fd["status"]

            if st == "skeleton" and fd["joints"]:
                joints = fd["joints"]
                sources = fd["sources"] or {}

                # ---- 先画描边（在原帧上，不叠半透明）----
                for ja, jb, side in BONES:
                    pa = joints.get(ja)
                    pb = joints.get(jb)
                    if pa is None or pb is None:
                        continue
                    cv2.line(img, (int(pa[0]), int(pa[1])),
                             (int(pb[0]), int(pb[1])), C_HALO, lw_halo,
                             cv2.LINE_AA)

                # ---- 再画彩色 ----
                # 修复：原版对 gem 来源用 overlay+addWeighted 混到整帧上，
                # 会把已画好的描边和线一起冲淡。改成只在**局部 ROI** 上混合。
                for ja, jb, side in BONES:
                    pa = joints.get(ja)
                    pb = joints.get(jb)
                    if pa is None or pb is None:
                        continue
                    both_gem = (sources.get(ja) == "gem"
                                and sources.get(jb) == "gem")
                    base = C_RIGHT if side == "right" else C_LEFT
                    x1, y1 = int(pa[0]), int(pa[1])
                    x2, y2 = int(pb[0]), int(pb[1])
                    if both_gem:
                        # 局部 ROI 混合，避免影响其他线
                        pad = lw + 4
                        rx0 = max(0, min(x1, x2) - pad)
                        ry0 = max(0, min(y1, y2) - pad)
                        rx1 = min(img.shape[1], max(x1, x2) + pad)
                        ry1 = min(img.shape[0], max(y1, y2) + pad)
                        if rx1 <= rx0 or ry1 <= ry0:
                            continue
                        roi = img[ry0:ry1, rx0:rx1]
                        tmp_roi = roi.copy()
                        cv2.line(tmp_roi, (x1 - rx0, y1 - ry0),
                                 (x2 - rx0, y2 - ry0), base, lw,
                                 cv2.LINE_AA)
                        cv2.addWeighted(tmp_roi, GEM_ALPHA, roi,
                                        1 - GEM_ALPHA, 0, roi)
                    else:
                        cv2.line(img, (x1, y1), (x2, y2), base, lw,
                                 cv2.LINE_AA)
                    n_lines_drawn += 1

                # 关节点
                for jn in DRAW_JOINTS:
                    p = joints.get(jn)
                    if p is None:
                        continue
                    src = sources.get(jn, "gem")
                    col = (C_LEFT if src == "fused"
                           else blend_color(C_LEFT, GEM_ALPHA, (100, 100, 100)))
                    cv2.circle(img, (int(p[0]), int(p[1])), jr,
                               C_HALO, -1, cv2.LINE_AA)
                    cv2.circle(img, (int(p[0]), int(p[1])),
                               max(1, jr - 1), col, -1, cv2.LINE_AA)

                n_skel_drawn += 1

            elif st == "track_only":
                b = tracking[k]["bbox"]
                if b:
                    cx = int((b[0] + b[2]) / 2)
                    cy = int((b[1] + b[3]) / 2)
                    cv2.circle(img, (cx, cy), max(3, jr), C_TRACK, -1,
                               cv2.LINE_AA)

            # HUD
            st_name = {"skeleton": "SKELETON", "track_only": "TRACK",
                       "none": "NONE"}.get(st, "?")
            hud = f"F{k}  {k/fps:.2f}s  {st_name}"
            cv2.putText(img, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 1, cv2.LINE_AA)

            vw.write(img)

        cap.release()
        vw.release()

        import subprocess
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
             str(dst / "overlay.mp4")],
            capture_output=True, text=True)
        tmp.unlink(missing_ok=True)

        if r.returncode != 0:
            print(f"  x {vid}: ffmpeg 失败 {r.stderr[-200:]}")
            continue

        sz = (dst / "overlay.mp4").stat().st_size / 1024**2
        print(f"  ✓ {vid}: {n_skel_drawn}/{n} 帧骨架, "
              f"{n_lines_drawn} 条线  {sz:.1f}MB", flush=True)
        results[vid] = {"frames": n, "skeleton_drawn": n_skel_drawn,
                        "lines_drawn": n_lines_drawn,
                        "mb": round(sz, 1),
                        "sec": round(time.time() - _t_vid, 1)}

    (out / "render_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成 {len(results)} 条")


if __name__ == "__main__":
    main()
