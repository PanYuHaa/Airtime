# ADR-001: Cross-Process Isolation for PyTorch Workloads (GEM-X & SAM2)

- **Status**: Accepted
- **Date**: 2026-11-24

## Context

The Airtime main pipeline (`M0`, `M1` detection, `M2` 2D pose, `M3F` fusion, `M8` rendering) runs on lightweight `onnxruntime-gpu` + `opencv-python` + `numpy`. In contrast, 3D human reconstruction (`M3` GEM-X) and video object segmentation (`M1` SAM2) require heavy `torch` + CUDA stacks with strict version constraints that frequently conflict with `onnxruntime-gpu` / Python 3.12 environments.

Previously, importing `torch` directly inside the main process caused `M3` to fail in environments without PyTorch installed.

## Decision

Isolate all PyTorch-dependent execution into dedicated CLI worker scripts invoked via `subprocess`:

- `src/airtime/pipeline/m3_gemx.py` dispatches `m3_run_gemx.py` using `AIRTIME_GEMX_PYTHON`.
- `src/airtime/pipeline/m1_track.py` dispatches `m1_run_sam2.py` using `AIRTIME_SAM2_PYTHON` (defaulting to `AIRTIME_GEMX_PYTHON`).
- Data crosses process boundaries exclusively through explicit temporary files (`.npy`, `.json`, `.pt`, `.mp4`).

## Consequences

- **Pros**: Clean dependency isolation, graceful degradation when SAM2/torch is absent, easy upgrade of PyTorch or ONNX Runtime independently.
- **Cons**: Small subprocess startup overhead per segment (negligible compared to 15–55 s GPU inference).
