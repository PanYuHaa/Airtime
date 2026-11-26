"""Unit tests for Airtime configuration files and schema validator."""
from __future__ import annotations

import importlib.util

from airtime import cfg

SCRIPT_PATH = cfg.REPO_ROOT / "scripts" / "validate_configs.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_configs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_configs_script_passes() -> None:
    validator = _load_validator()
    assert validator.main() == 0


def test_cfg_check_setup_non_strict_passes() -> None:
    assert cfg.check_setup(strict=False) == []


def test_style_config_has_valid_sections() -> None:
    style = cfg.load_config("style.json")
    validator = _load_validator()
    assert validator.validate_style_config(style) == []


def test_validate_pipeline_config_rejects_bad_scale_range() -> None:
    validator = _load_validator()
    bad = cfg.load_pipeline_config()
    bad["m3f_fusion"]["rigid_align"]["scale_min"] = 2.0
    bad["m3f_fusion"]["rigid_align"]["scale_max"] = 1.0
    errs = validator.validate_pipeline_config(bad)
    assert any("scale_min" in e for e in errs)
