## Summary

<!-- Describe what this change does and why. Link related issues or ADRs. -->

Closes #

## Changes

-
-

## Verification & Testing

- [ ] `python scripts/validate_configs.py` passes
- [ ] `pytest -v` passes locally
- [ ] `ruff check src tests scripts` passes with no new warnings
- [ ] Determinism verified (if touching M1 / M2 / M3 / M3F inference or fusion)

### Benchmark Comparison (`airtime-eval`, required for algorithm/pipeline PRs)

<!-- Paste output of `python -m airtime.eval --before <base> --after <exp>` if applicable. -->

| Metric | Before | After | Delta |
|---|---|---|---|
| Skeleton coverage (%) | | | |
| Jitter p95 (% box_h) | | | |
| Bad projection frames | | | |

## Engineering Standards Checklist

- [ ] No hardcoded local paths (`F:\`, `/mnt/`, `/home/<user>`) in repository code
- [ ] No model weights, raw videos, API keys, or tokens committed
- [ ] Configurable thresholds live in `configs/pipeline.json`, not hardcoded in source
- [ ] Documentation (`README.md`, `docs/`, `CHANGELOG.md`) updated if behavior or schema changed
