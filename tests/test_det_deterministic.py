"""Unit tests for deterministic ONNX Runtime patching and M1 box quality rules."""
from __future__ import annotations

import sys
import types

from airtime import det_deterministic, m1_lock


def test_patch_rtmlib_injects_heuristic_cuda_option(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeSession:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    det_deterministic.patch_rtmlib(verbose=False)
    fake_ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    fake_ort.InferenceSession(
        "model.onnx",
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )

    assert captured[0]["providers"][0] == (
        "CUDAExecutionProvider",
        {"cudnn_conv_algo_search": "HEURISTIC"},
    )
    assert captured[0]["providers"][1] == "CPUExecutionProvider"
    assert captured[1]["providers"][0] == (
        "CUDAExecutionProvider",
        {"cudnn_conv_algo_search": "HEURISTIC", "device_id": 0},
    )


def test_box_quality_rejects_edge_and_narrow_boxes() -> None:
    ok_good, reasons_good = m1_lock.box_quality([300, 300, 420, 600], W=1080, H=1920, thr_h=80)
    assert ok_good is True
    assert reasons_good == []

    ok_bad, reasons_bad = m1_lock.box_quality([0, 1455, 9, 1630], W=1080, H=1920, thr_h=80)
    assert ok_bad is False
    assert "clip_left" in reasons_bad
    assert "too_narrow" in reasons_bad
