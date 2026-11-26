# ADR-002: Deterministic ONNX Runtime CUDA Execution

- **Status**: Accepted
- **Date**: 2026-11-23

## Context

Running `M1` (YOLOX) and `M2` (RTMPose-x) in two separate OS processes on the exact same video produced non-deterministic outputs (bounding box drift up to `0.03 px`, keypoint drift up to `0.5–13 px`), causing downstream skeleton frames to flip across quality thresholds.

Root cause: `onnxruntime`'s `CUDAExecutionProvider` defaults to `cudnn_conv_algo_search=EXHAUSTIVE`, which benchmarks convolution algorithms dynamically based on instantaneous GPU clock/memory state and selects different floating-point reduction orders.

## Decision

Patch `onnxruntime.InferenceSession` via `airtime.det_deterministic.patch_rtmlib()` prior to instantiating `rtmlib` models, setting **only**:

```python
CUDA_OPTS = {"cudnn_conv_algo_search": "HEURISTIC"}
```

## Alternatives Rejected

Ablation experiments showed that forcing single-threading, disabling TF32, or disabling memory pattern optimization was unnecessary for cross-process determinism and regressed skeleton jitter metrics (`p95`) on benchmark clips.

## Consequences

- **Pros**: Cross-process bit-identical outputs with minimal perturbation to numerical accuracy and runtime speed.
- **Cons**: Must be invoked before `rtmlib` constructs any `InferenceSession`.
