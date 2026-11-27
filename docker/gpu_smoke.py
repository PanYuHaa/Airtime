#!/usr/bin/env python3
"""镜像内 GPU 冒烟：确认 ORT 能真正在 CUDA EP 上跑一次推理。

用法（在拿得到 GPU 的容器里）：
    docker run --rm --gpus all -v "$PWD/models:/models:ro" airtime:lean \
        python /app/docker/gpu_smoke.py [模型路径]

默认取 cfg.RTMPOSE_ONNX（即 ${AIRTIME_MODELS}/rtmpose-...onnx）。
脚本只用随机张量做一次前向，不需要真实视频；找不到模型时以退出码 2 结束并提示。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from airtime import cfg

LIBS = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12",
        "libcufft.so.11", "libcurand.so.10", "libcudnn.so.9")


def main() -> int:
    for lib in LIBS:
        ctypes.CDLL(lib)
        print(f"OK {lib}")

    print("available:", ort.get_available_providers())
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.RTMPOSE_ONNX
    if not model.is_file():
        print(f"模型不存在，跳过推理: {model}", file=sys.stderr)
        return 2

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(model), sess_options=so,
        providers=[("CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"})],
    )
    print("session:", sess.get_providers())

    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
        print("input", inp.name, shape)

    outs = sess.run(None, feeds)
    print("outputs:", [o.shape for o in outs])
    print("GPU smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
