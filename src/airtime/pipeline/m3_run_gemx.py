#!/usr/bin/env python3
"""GEM-X 单段推理（必须在 sp-gem 环境里运行，即 torch 可用）。

由 m3_gemx.py 通过 subprocess 调用，不直接 import 到主流程，
因为主流程环境（.venv-p1）没有 torch。

用法：
    python m3_run_gemx.py --clip clip.mp4 --bbx bbx.npy --out result.pt \
        --gem /path/to/GEM-X --camera background
"""
import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--bbx", required=True, help="npy 文件，N×4 xyxy")
    ap.add_argument("--out", required=True, help="输出的 .pt 路径")
    ap.add_argument("--gem", required=True, help="GEM-X 代码目录")
    ap.add_argument("--camera", default="background",
                    choices=["background", "static"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    gem = Path(a.gem).resolve()
    sys.path.insert(0, str(gem))

    import torch  # noqa: E402
    import gem.inference.body_video as BV  # noqa: E402
    from gem.inference.camera_motion import camera_angular_velocity  # noqa: E402
    from gem.inference.motion import load_predictor, run_motion  # noqa: E402
    from gem.inference.serialization import save_predictions, tensor_tree  # noqa: E402
    from gem.inference.video import VideoFrames, normalize_video  # noqa: E402
    from gem.common.detectors.tracks import (  # noqa: E402
        TrackResult, smooth_boxes, square_crops)

    # 确定性开关
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    bbx = np.load(a.bbx)
    clip = Path(a.clip)

    cfg0 = BV.load_body_video_config(
        gem / "docs/configs/body_video_native77.yaml")
    predictor = load_predictor(cfg0["motion"]["checkpoint"], cfg0["motion"],
                               stats_path=cfg0["motion"].get("stats"),
                               device=a.device)

    def make_tracks(n_frames, H, W, bbx_in):
        B = np.asarray(bbx_in, dtype=np.float32)
        if len(B) != n_frames:
            src = np.linspace(0, 1, len(B))
            dst = np.linspace(0, 1, n_frames)
            B = np.stack([np.interp(dst, src, B[:, c]) for c in range(4)],
                         1).astype(np.float32)
        B[:, [0, 2]] = np.clip(B[:, [0, 2]], 0, W - 1)
        B[:, [1, 3]] = np.clip(B[:, [1, 3]], 0, H - 1)
        empty = [np.zeros((0, 4), np.float32) for _ in range(n_frames)]
        return TrackResult(
            np.arange(n_frames), B, square_crops(smooth_boxes(B), 1.2),
            np.zeros(n_frames, np.int32), np.ones(n_frames, bool),
            (0, n_frames - 1), [b[None] for b in B], empty,
            [b[None] for b in B],
            [np.ones(1, np.float32) for _ in range(n_frames)],
            ["provided_by_ski_professor"] * n_frames)

    cfg = copy.deepcopy(cfg0)
    cfg.setdefault("camera", {})["mode"] = a.camera

    # M1 已提供逐帧 bbox，跳过 GEM-X 内部无用的 YOLOXONNX 初始化与检测器加载
    BV.YOLOXONNX = lambda *a, **kw: None
    BV.detect_track = lambda frames, det, **kw: make_tracks(
        len(frames), *frames[0].shape[:2], bbx)

    with tempfile.TemporaryDirectory() as tmp:
        meta = normalize_video(clip, Path(tmp) / "in.mp4")
        with VideoFrames(Path(tmp) / "in.mp4") as frames:
            seq, tracks, camera = BV.build_body_conditions(
                frames, cfg, device=a.device, progress=lambda m: None)
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
        res = run_motion(predictor, s, device=a.device)
        res = tensor_tree({**res, "format": "gem_body_video_predictions_v1",
                           "video": meta, "release": cfg, "conditions": s,
                           "tracking": {k: v for k, v in tracks.__dict__.items()},
                           "camera": cam})
        res["video"]["source"]["path"] = str(clip.resolve())
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_predictions(res, out)

    del predictor
    torch.cuda.empty_cache()
    print(json.dumps({"ok": True, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
