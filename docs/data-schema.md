# Data Schema & Artifact Contracts

This document specifies the directory layout and JSON/NPZ contracts produced by an Airtime run (`runs/<run_name>/`).

---

## 1. Run Directory Layout

```text
runs/<run_name>/
├── clips/<vid>.mp4             # M0 normalized 1080p 30fps CFR clip
├── meta/<vid>.json             # M0 video metadata (resolution, fps, HDR transfer, time contract)
├── tracking/<vid>.json         # M1 per-frame bounding boxes, source, and status
├── segments/<vid>.json         # M1 contiguous skeleton segments & GEM-X inference windows
├── pose2d/<vid>.npz            # M2 Halpe26 2D keypoints [T, 26, 2] & scores [T, 26]
├── gem/<vid>/                  # M3 GEM-X raw per-segment 3D outputs (.npz)
├── skeleton/<vid>.json         # ★ M3F final fused skeleton (primary output)
├── render/<vid>/overlay.mp4    # M8 visual overlay (plus overlay_sdr.mp4 for HDR sources)
└── *_stats.json / index.json   # Stage timing and summary metrics consumed by airtime.eval
```

---

## 2. Primary Artifact: `skeleton/<vid>.json`

```json
{
  "video": "my_ski",
  "fps": 30,
  "frames": 180,
  "joint_names": ["Hips", "Spine1", "Spine2", "..."],
  "frames_data": [
    {
      "i": 0,
      "t": 0.0,
      "status": "skeleton",
      "joints": {
        "Hips": [812.3, 645.1],
        "HeadEnd": [815.0, 520.4]
      }
    },
    {
      "i": 1,
      "t": 0.033,
      "status": "track_only",
      "joints": null
    }
  ]
}
```

### Frame `status` Enumeration

| Value | Meaning |
|---|---|
| `skeleton` | Valid fused skeleton available for this frame |
| `track_only` | Athlete tracked, but skeleton suppressed (too small, occluded, or quality-gated) |
| `none` | Target not detected/tracked in this frame |

---

## 3. Time Contract (`airtime.time_contract.v1`)

Stored inside `meta/<vid>.json` (or generated directly by `airtime.time_contract.build_time_contract`):

- `schema_version`: `"airtime.time_contract.v1"`
- `summary`:
  - `source_frames`, `source_nominal_fps`, `source_avg_fps`, `is_vfr`
  - `clip_frames`, `clip_fps`
  - `slowmo_factor`: positive float or `null`
  - `real_time_confirmed`: `true` iff `slowmo_factor` is explicitly supplied
  - `effective_capture_fps`: `clip_fps * slowmo_factor` (or `null`)
- `frames[]`: per-clip-frame record containing `k`, `playback_time_s`, `source_frame_idx`, `source_pts_s`, and `real_time_s` (`null` when unconfirmed).

---

## 4. 2D Keypoints: `pose2d/<vid>.npz`

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `kpts` | `(T, 26, 2)` | `float32` | Halpe26 `(x, y)` pixel coordinates; `NaN` on frames outside skeleton segments |
| `scores` | `(T, 26)` | `float32` | Per-joint confidence scores `[0, 1]` |

---

## 5. Configuration Contracts (`configs/`)

- `configs/pipeline.json`: Master pipeline thresholds (`m0_preprocess`, `m1_track`, `m2_pose2d`, `m3_gemx`, `m3f_fusion`, `m8_render`). Validated by `scripts/validate_configs.py`.
- `configs/joint_names.json`: Ordered 77-element SOMA joint list.
- `configs/style.json`: Shared visual tokens (`#RRGGBB` colors, stroke widths, HUD styling).
- `configs/videos.example.json`: Template video manifest (`id`, `file`, `camera`, `target_hint`).
