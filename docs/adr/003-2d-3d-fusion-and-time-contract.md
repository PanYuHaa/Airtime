# ADR-003: 2D/3D Skeleton Fusion & Explicit Physical Time Contract

- **Status**: Accepted
- **Date**: 2026-11-25

## Context

1. **Monocular 3D vs 2D trade-off**: 3D reconstruction (`GEM-X`) provides depth priors during spins and aerials but can drift in image space under occlusion or motion blur. 2D keypoints (`RTMPose-x`) are accurate in screen space but lack depth. Naive per-joint snapping can tear limb proportions.
2. **Anatomical head mapping**: Mapping Halpe26 `nose` to SOMA `HeadEnd` pulled the top-of-head vertex down toward the athlete's goggles (~14.3% body-height offset vs ~4.7% for `head`).
3. **Slow-motion & VFR footage**: Normalizing videos to 30 fps CFR for inference loses source PTS and risks computing bogus physical airtime or angular velocity on slow-motion clips.

## Decision

1. **Two-stage fusion with rollback (`m3f_fuse.py`)**: Perform robust rigid alignment first, apply reliability-gated per-joint 2D corrections, and roll back any limb chain whose resulting bone length deviates beyond `bone_len_tolerance` (`0.3`).
2. **Remove `nose -> HeadEnd` mapping**: Map only `head -> HeadEnd` in `JOINT_MAP`.
3. **Explicit Time Contract (`time_contract.py`)**: Record source PTS and VFR status before transcoding, map every normalized frame back to source PTS, and emit `real_time_s` only when `slowmo_factor` is explicitly provided.

## Consequences

- **Pros**: Prevents skeleton tearing, removes systematic head offset, and guarantees downstream physics modules never silently mistake slow-motion playback time for real physical time.
- **Cons**: Requires callers needing physical metrics to supply `slowmo_factor` explicitly.
