# Changelog

All notable changes to **Airtime** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Engineering Standards & Governance**:
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`) running config validation, `ruff` lint, and cross-version Python `pytest` checks.
  - Engineering documentation suite under `docs/`: architecture overview, data schema contract, development guide, and Architecture Decision Records (`docs/adr/001`–`003`).
  - Repository governance files: `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.editorconfig`, `Makefile`, and GitHub Issue/PR templates.
  - Automated configuration schema validator (`scripts/validate_configs.py`) and additional unit tests (`tests/test_configs.py`, `tests/test_det_deterministic.py`).
  - CLI entrypoint `airtime-eval` registered in `pyproject.toml`.
- **Benchmark Evaluation & Time Contract** (PR #2):
  - `airtime.eval`: Quantitative evaluation, regression detection, and visual comparison sheet generator.
  - `airtime.time_contract`: Frame-accurate PTS mapping, VFR detection, and explicit slow-motion physical time conversion.
  - `airtime.pipeline.m1_run_sam2`: Cross-process SAM2 video segmentation runner for torch-isolated environments.

### Fixed
- **CLI Entrypoint & Path Defaults** (PR #1): Fixed relative imports when running stages via `airtime`, default camera fallback, and GEM-X default path resolution.
- Removed unused self-import in `airtime.det_deterministic`.

---

## [1.0.0] - 2026-11-24

### Added
- Complete end-to-end 6-stage monocular freestyle ski 3D skeleton reconstruction pipeline (`M0` → `M1` → `M2` → `M3` → `M3F` → `M8`).
- Activity-driven target locking (`m1_lock.py`) with edge/aspect/jump penalties to avoid locking onto bystanders or shadows.
- Deterministic ONNX Runtime CUDA execution patch (`det_deterministic.py`) using `cudnn_conv_algo_search=HEURISTIC`.
- Cross-process GEM-X 3D inference (`m3_gemx.py` + `m3_run_gemx.py`) isolating heavy PyTorch/CUDA dependencies from the lightweight main pipeline.
- Two-model 2D/3D fusion (`m3f_fuse.py`) with rigid alignment, bone-length rollback protection, and `head -> HeadEnd` anatomical mapping.
- HDR (HLG/PQ) and SDR color-managed overlay rendering (`m8_overlay.py`).
- Apache-2.0 open-source license and zero-hardcoded-path configuration (`configs/pipeline.json`).
