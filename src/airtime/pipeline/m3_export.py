#!/usr/bin/env python3
"""
P1 M3：从 GEM-X .pt 导出为 npz（直接读文件，不依赖目录结构）

复用 P0 的 export_gemx.py 逻辑，但接受明确的 .pt 路径。
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def export_one(res, geo, path, postprocess=False, soma=None, device="cuda"):
    P = {}
    for coord in ("incam", "global"):
        joints = torch.cat(
            [ch["joints"] for _, ch in geo.iter_chunks(res[f"pred_soma_params_{coord}"])],
            1)[0].numpy()
        P[coord] = joints

    cond = res["conditions"]
    K = to_np(cond["K_fullimg"])[0]
    J = P["incam"]
    uvw = np.einsum("tij,tkj->tki", K, J)
    kp = uvw[..., :2] / np.clip(uvw[..., 2:3], 1e-6, None)

    obs = cond.get("kp2d_native77")
    obs = to_np(obs)[0] if obs is not None else np.zeros((len(J), 77, 3))

    g = res["pred_soma_params_global"]

    np.savez_compressed(
        path,
        j3d_cam=J.astype(np.float32),
        j3d_world=P["global"].astype(np.float32),
        kp2d_proj=kp.astype(np.float32),
        kp2d_obs=obs.astype(np.float32),
        contact=torch.sigmoid(res["static_conf_logits"][0]).numpy().astype(np.float32),
        R_w2c=to_np(res["camera"]["R_w2c"]).astype(np.float32),
        K=K.astype(np.float32),
        global_orient=to_np(g["global_orient"])[0].astype(np.float32),
        transl=to_np(g["transl"])[0].astype(np.float32),
        bbx_xys=to_np(cond["bbx_xys"])[0].astype(np.float32),
        frame_times=to_np(cond["frame_times"])[0],
        img_size=np.array([res["video"]["normalized"]["width"],
                           res["video"]["normalized"]["height"]]))

    if postprocess:
        try:
            from gem.inference.postprocess import postprocess_motion_result
            pp = postprocess_motion_result(res, soma_assets=soma, device=device)
            export_one(pp, geo, path.with_name(path.stem + "_pp.npz"),
                       postprocess=False, soma=soma, device=device)
        except Exception as e:
            print(f"    后处理失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help=".pt 文件或目录")
    ap.add_argument("--out", required=True, help="输出 npz 路径（单文件）或目录（多文件）")
    ap.add_argument("--soma", required=True)
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from gem.motion.geometry import MotionGeometry
    geo = MotionGeometry(a.soma, device=a.device, chunk_size=64)

    pt = Path(a.pt)
    out = Path(a.out)

    if pt.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted(pt.glob("*.pt")):
            dst = out / f"{p.stem}.npz"
            if dst.exists():
                continue
            res = torch.load(p, map_location="cpu", weights_only=False)
            export_one(res, geo, dst, a.postprocess, a.soma, a.device)
            print(f"  ✓ {dst.name}", flush=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        res = torch.load(pt, map_location="cpu", weights_only=False)
        export_one(res, geo, out, a.postprocess, a.soma, a.device)
        print(f"  ✓ {out}", flush=True)


if __name__ == "__main__":
    main()
