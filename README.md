# tongflow-modal-sapiens2

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Human-centric vision suite built on **Sapiens2** (Meta, ICLR 2026, [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)) — six capabilities from the five 1B task checkpoints, running on one GPU app via [Modal](https://modal.com).

## Capabilities

- **Pose detection** (`image-pose`) — 308-keypoint whole-body pose (body, hands, 274 face points) for every detected person, rendered as a skeleton overlay PNG.
- **Body-part segmentation** (`image-body-seg`) — 29-class per-pixel body-part segmentation, color-coded overlay PNG.
- **Surface normals** (`image-normal`) — per-pixel surface normal map, background masked via body-part segmentation.
- **Human matting** (`image-matting`) — human foreground extraction as a straight-alpha transparent PNG.
- **Image → 3D** (`image-gen-model`) — Sapiens2 pointmap turned into a colored, human-only 3D point cloud GLB.
- **Video motion capture** (`video-gen-model`) — monocular video → animated 3D human GLB, pure Sapiens2: 2D pose is lifted to 3D by sampling the pointmap at each keypoint, One-Euro smoothed, solved into per-joint rotations (body + fingers + jaw, with rest-pose hold for unseen body parts), and retargeted onto **MHR** (Meta's Momentum Human Rig, Apache-2.0). Single subject (largest person, IoU-tracked). Plays directly in the TongFlow model node; imports as a skinned armature in Blender. Set `MOCAP_STYLE=skeleton` for the raw bone-puppet output; run `modal run extract_mhr.py::extract` once to (re)build the MHR bundle. For the SAM-3D-Body-based capture (learned prior, hands + experimental face), use the [tongflow-modal-sam-3d-body](https://github.com/tong-io/tongflow-modal-sam-3d-body) plugin on the same node slot.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

Checkpoints are public (no HF token needed). On first use the plugin deploys to your Modal account automatically; weights (~31 GB: five 1B checkpoints + the DETR person detector) are cached on a shared Modal volume.

## Tuning (env, optional)

| Env | Default | Notes |
| --- | --- | --- |
| `SAPIENS2_MODEL_SIZE` | `1b` | Checkpoint size to load (`0.4b`, `0.8b`, `1b` — re-run the weight download after changing). |
| `SAPIENS2_GPU` | `L40S` | Modal GPU type (`A100`, `H100`, ... — applied on the next deploy). |
| `SAPIENS2_DTYPE` | `bf16` | Inference precision; set `fp32` to disable bf16. |
| `SAPIENS2_GPU_LRU` | `5` (bf16) / `3` (fp32) | Task models kept GPU-resident before evicting to CPU. |
| `MOCAP_FPS` | `24` | Mocap sampling framerate. |
| `MOCAP_BATCH` | `4` | Frames per batched pose/pointmap/detector forward. |
| `MOCAP_MAX_SECONDS` | `60` | Mocap duration cap. |
| `MOCAP_POSE_CONF_THR` | `0.3` | Keypoint confidence below this is temporally interpolated. |
| `MOCAP_ONE_EURO_MIN_CUTOFF` / `MOCAP_ONE_EURO_BETA` | `1.5` / `0.3` | Smoothing: lower cutoff = smoother, higher beta = snappier. |
| `TONGFLOW_MODAL_CALL_TIMEOUT_S` | `3600` | Max seconds to wait for a Modal call. |

## Hardware

One **L40S** container serves all six slots. Models load lazily with a 3-model GPU LRU; the mocap slot keeps pose + pointmap resident and runs ~2–4 frames/s end to end.

## License

Plugin code: AGPL-3.0 (TongFlow). Sapiens2 models and code are under the [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md) (Meta) — review it before commercial use.
