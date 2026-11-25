#!/usr/bin/env python3
"""SAM2 单种子视频分割补帧子进程脚本（需在含 torch + CUDA 的环境中运行）。

与 m3_run_gemx.py 同构：主流程环境只需轻量 onnxruntime，SAM2 通过本脚本跨进程执行。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed-k", type=int, required=True)
    ap.add_argument("--seed-box", required=True, help="JSON 数组 [x1,y1,x2,y2]")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--sam2-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.sam2_dir).resolve()))
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    seed_box = [float(x) for x in json.loads(a.seed_box)]
    W, H = int(a.width), int(a.height)
    scale = min(1.0, 720.0 / min(W, H))
    tw, th = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2

    tmpdir = Path(tempfile.mkdtemp(prefix="sam2t_"))
    out: dict[str, list[float]] = {}
    try:
        cap = cv2.VideoCapture(str(a.video))
        n = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            cv2.imwrite(
                str(tmpdir / f"{n:05d}.jpg"),
                cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            n += 1
        cap.release()

        device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
        pred = build_sam2_video_predictor(a.cfg, str(a.ckpt), device=device)
        sb = [seed_box[0] * scale, seed_box[1] * scale, seed_box[2] * scale, seed_box[3] * scale]

        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else torch.inference_mode()
        )
        with torch.inference_mode(), ctx:
            st = pred.init_state(str(tmpdir), offload_video_to_cpu=True)
            pred.add_new_points_or_box(
                st, frame_idx=a.seed_k, obj_id=1, box=np.array(sb, dtype=np.float32)
            )
            for rev in (False, True):
                for fidx, _, ml in pred.propagate_in_video(
                    st, start_frame_idx=a.seed_k, reverse=rev
                ):
                    m = (ml[0, 0] > 0).cpu().numpy()
                    if m.any():
                        ys, xs = np.where(m)
                        out[str(int(fidx))] = [
                            float(xs.min() / scale),
                            float(ys.min() / scale),
                            float(xs.max() / scale),
                            float(ys.max() / scale),
                        ]
        pred.reset_state(st)
        del pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
