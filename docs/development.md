# Development Guide

This guide covers local setup, coding standards, testing, and benchmarking for Airtime contributors.

---

## 1. Local Environment Setup

```bash
git clone https://github.com/PanYuHaa/Airtime.git
cd Airtime

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Configure external data and weight directories via environment variables rather than modifying code:

```bash
export AIRTIME_MODELS=/path/to/models
export AIRTIME_DATA=/path/to/data
export AIRTIME_OUTPUT=/path/to/runs
export AIRTIME_GEMX=/path/to/GEM-X
export AIRTIME_GEMX_PYTHON=/path/to/sp-gem/bin/python
# Optional: defaults to AIRTIME_GEMX_PYTHON if unset
export AIRTIME_SAM2_PYTHON=/path/to/sp-gem/bin/python
```

Verify configuration health:

```bash
airtime-check
airtime-check --strict   # also verifies model weight paths
```

---

## 2. Standard Engineering Checks

Use the repository `Makefile`:

```bash
make validate-configs   # Validate configs/*.json invariants (stdlib only)
make lint               # Run ruff static analysis
make test               # Run pytest unit test suite
make check              # Run all three checks above
```

---

## 3. Coding Standards (Google Python Style)

1. **Formatting & Linting**: Enforced via `ruff` (`line-length = 100`, `target-version = "py310"`).
2. **Docstrings**: Use Google-style docstrings describing behavior, inputs, outputs, and invariants.
3. **No Magic Thresholds**: Put tunable parameters in `configs/pipeline.json` with a clear explanation of calibration rationale.
4. **Determinism**: Any ONNX Runtime / PyTorch inference change must preserve bit-identical outputs across separate processes.
5. **Minimal Coupling**: Keep `src/airtime/` free of heavy torch imports at module top-level so unit tests and CPU tooling remain fast and portable.

---

## 4. Quantitative Evaluation Workflow

Before submitting any PR that modifies tracking, pose estimation, or 2D/3D fusion:

```bash
# Run baseline vs experiment
airtime-eval --before runs/baseline --after runs/experiment \
  --sheet-dir runs/compare_sheets \
  --md runs/compare.md
```

Include the generated Markdown table in your Pull Request description.
