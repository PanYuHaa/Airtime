#!/usr/bin/env python3
"""Validate Airtime JSON configuration files for structural and semantic integrity.

Runs with only the Python standard library so it can execute in minimal CI jobs
and pre-commit checks without GPU or OpenCV dependencies.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pipeline_config(doc: object) -> list[str]:
    """Validate configs/pipeline.json required sections and core invariants."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["pipeline.json root must be a JSON object"]

    required_sections = (
        "m0_preprocess",
        "m1_track",
        "m2_pose2d",
        "m3_gemx",
        "m3f_fusion",
        "m8_render",
    )
    for section in required_sections:
        if section not in doc or not isinstance(doc[section], dict):
            errors.append(f"pipeline.json missing object section: {section}")

    if errors:
        return errors

    m0 = doc["m0_preprocess"]
    if m0.get("target_fps", 0) <= 0:
        errors.append("m0_preprocess.target_fps must be > 0")
    if m0.get("short_side", 0) <= 0:
        errors.append("m0_preprocess.short_side must be > 0")

    m1 = doc["m1_track"]
    if m1.get("min_person_height_px", 0) <= 0:
        errors.append("m1_track.min_person_height_px must be > 0")
    lock = m1.get("lock", {})
    w_dur = lock.get("weight_duration", -1)
    if not (0.0 <= w_dur <= 1.0):
        errors.append(f"m1_track.lock.weight_duration out of [0, 1]: {w_dur}")

    m3f = doc["m3f_fusion"]
    if not (0.0 < m3f.get("bone_len_tolerance", 0) < 2.0):
        errors.append("m3f_fusion.bone_len_tolerance must be in (0, 2)")
    ra = m3f.get("rigid_align", {})
    if ra.get("scale_min", 1) >= ra.get("scale_max", 0):
        errors.append("m3f_fusion.rigid_align.scale_min must be < scale_max")

    return errors


def validate_joint_names(doc: object) -> list[str]:
    """Validate configs/joint_names.json matches SOMA 77 contract."""
    errors: list[str] = []
    if not isinstance(doc, list):
        return ["joint_names.json must be a JSON array"]
    if len(doc) != 77:
        errors.append(f"joint_names.json must contain 77 SOMA joints, got {len(doc)}")
    if len(set(doc)) != len(doc):
        errors.append("joint_names.json contains duplicate joint names")
    for required in ("Hips", "Head", "HeadEnd", "LeftFoot", "RightFoot"):
        if required not in doc:
            errors.append(f"joint_names.json missing required joint: {required}")
    return errors


def validate_style_config(doc: object) -> list[str]:
    """Validate configs/style.json color tokens and required sections."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["style.json root must be a JSON object"]
    for section in ("skeleton", "accent", "hud", "scene3d", "chart"):
        if section not in doc or not isinstance(doc[section], dict):
            errors.append(f"style.json missing object section: {section}")
    for section_name, section in doc.items():
        if not isinstance(section, dict):
            continue
        for k, v in section.items():
            if isinstance(v, str) and v.startswith("#") and not HEX_COLOR_RE.match(v):
                errors.append(f"style.json invalid hex color at {section_name}.{k}: {v}")
    return errors


def validate_videos_example(doc: object) -> list[str]:
    """Validate configs/videos.example.json structure."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["videos.example.json root must be a JSON object"]
    videos = doc.get("videos")
    if not isinstance(videos, list) or not videos:
        return ["videos.example.json must contain a non-empty 'videos' list"]
    for idx, item in enumerate(videos):
        if not isinstance(item, dict) or not item.get("id") or not item.get("file"):
            errors.append(f"videos.example.json videos[{idx}] must define 'id' and 'file'")
    return errors


def main() -> int:
    checks = [
        ("pipeline.json", validate_pipeline_config),
        ("joint_names.json", validate_joint_names),
        ("style.json", validate_style_config),
        ("videos.example.json", validate_videos_example),
    ]
    all_errors: list[str] = []
    for filename, fn in checks:
        path = CONFIGS_DIR / filename
        try:
            doc = _load_json(path)
        except Exception as exc:
            all_errors.append(f"{filename}: {exc}")
            continue
        errs = fn(doc)
        if errs:
            all_errors.extend(errs)
        else:
            print(f"OK   configs/{filename}")

    if all_errors:
        print("\nConfiguration validation failed:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nAll Airtime configurations are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
