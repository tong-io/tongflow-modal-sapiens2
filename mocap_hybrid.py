"""Hybrid video mocap: SAM 3D Body drives the skeleton, Sapiens2 the face.

Per frame, SAM 3D Body regresses the full MHR state (joint global rotations,
camera translation, 72 face-expression coefficients) — a learned prior that
handles occlusion and limb twist far better than geometric lifting. Sapiens2's
274 facial landmarks refine the jaw hinge, and its keypoint pipeline remains
available for future gaze/expression refinement.

Output: the same skinned MHR GLB as mocap_retarget, plus sparse morph targets
driven by the smoothed expression coefficients.
"""

from __future__ import annotations

import math
import os

import numpy as np

import sapiens2_runtime as rt
from gltf_writer import skinned_character_glb
from mocap_pipeline import (
    MIN_VISIBILITY,
    POSE_CONF_THR,
    _quat_conj,
    _quat_from_matrix,
    _quat_mul,
    _quat_rotate,
    extract_frames,
    one_euro,
)
from mocap_retarget import JAW_ENABLED, JAW_MAX_RAD, MHR_SCALE, load_character

MOCAP_BATCH = int(os.environ.get("MOCAP_BATCH", "4"))
EXPR_ENABLED = os.environ.get("MOCAP_EXPR", "1") != "0"
# Cam->glTF axis flip (y-down/z-forward -> y-up), as a quaternion conjugation.
_FLIP = np.diag([1.0, -1.0, -1.0])


def _mat_to_quat_flipped(mats: np.ndarray) -> np.ndarray:
    """(J, 3, 3) camera-frame rotations -> (J, 4) world-frame quats."""
    out = np.zeros((len(mats), 4))
    for j, m in enumerate(mats):
        out[j] = _quat_from_matrix(_FLIP @ m @ _FLIP)
    return out


def _fk_positions(char, glob_rot: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
    """Joint positions implied by global rotations (one frame)."""
    parents = char["parents"]
    rest_pos = char["rest_pos"]
    pos = np.zeros_like(rest_pos)
    for j in range(len(parents)):
        p = parents[j]
        if p < 0:
            pos[j] = root_pos
        else:
            offset = rest_pos[j] - rest_pos[p]
            pos[j] = pos[p] + _quat_rotate(glob_rot[p], offset)
    return pos


def _resolve_convention(char, sample: dict) -> bool:
    """True if pred_global_rots are absolute orientations (bind included).

    Verified against the model's own pred_joint_coords: FK the skeleton under
    both interpretations and keep whichever reproduces the coordinates better.
    """
    coords = sample["joint_coords"] @ _FLIP.T
    root = coords[char["by_name"]["root"]]
    quats_abs = _mat_to_quat_flipped(sample["global_rots"])
    quats_delta = np.array(
        [_quat_mul(q, r) for q, r in zip(quats_abs, char["rest_rot"])]
    )

    def err(glob):
        fk = _fk_positions(char, glob, root)
        return float(np.linalg.norm(fk - coords, axis=1).mean())

    e_abs, e_delta = err(quats_abs), err(quats_delta)
    print(f"sam3dbody rotation convention: absolute={e_abs:.4f}m delta={e_delta:.4f}m")
    return e_abs <= e_delta


def capture_hybrid(hub: rt.ModelHub, video_bytes: bytes, progress=None) -> bytes:
    def report(msg: str) -> None:
        if progress is not None:
            progress(msg)

    char = load_character()
    by_name = char["by_name"]
    parents = char["parents"]
    rest_rot = char["rest_rot"]
    n_joints = len(char["names"])

    report("mocap: extracting frames")
    frames, fps = extract_frames(video_bytes)
    n_frames = len(frames)

    engine = rt.sam3d_engine()
    report(f"mocap: SAM 3D Body across {n_frames} frames")
    quats = np.zeros((n_frames, n_joints, 4))
    cam_t = np.zeros((n_frames, 3))
    expr = np.zeros((n_frames, 72))
    valid = np.zeros(n_frames, dtype=bool)
    convention_absolute: bool | None = None
    prev_bbox = None
    for f, frame in enumerate(frames):
        if f % 24 == 0:
            report(f"mocap: body {f}/{n_frames}")
        person = engine.infer_frame(frame, prev_bbox)
        if person is None:
            continue
        prev_bbox = person["bbox"]
        if person["global_rots"].shape[0] != n_joints:
            raise RuntimeError(
                f"SAM 3D Body joint count {person['global_rots'].shape[0]} != "
                f"MHR bundle {n_joints}"
            )
        if convention_absolute is None:
            convention_absolute = _resolve_convention(char, person)
        q = _mat_to_quat_flipped(person["global_rots"])
        if not convention_absolute:
            q = np.array([_quat_mul(qq, r) for qq, r in zip(q, rest_rot)])
        quats[f] = q
        cam_t[f] = _FLIP @ person["cam_t"]
        e = person["expr"]
        expr[f, : min(72, len(e))] = e[:72]
        valid[f] = True

    if not valid.any():
        raise RuntimeError("no person detected in the video")

    # Hold the last valid state through detection gaps, then smooth.
    last = None
    for f in range(n_frames):
        if valid[f]:
            last = f
        elif last is not None:
            quats[f], cam_t[f], expr[f] = quats[last], cam_t[last], expr[last]
    first = int(np.argmax(valid))
    quats[:first], cam_t[:first], expr[:first] = (
        quats[first],
        cam_t[first],
        expr[first],
    )

    for f in range(1, n_frames):
        dots = (quats[f] * quats[f - 1]).sum(axis=1)
        quats[f, dots < 0] *= -1
    quats = one_euro(quats, fps)
    quats /= np.linalg.norm(quats, axis=2, keepdims=True).clip(min=1e-9)
    cam_t = one_euro(cam_t, fps)
    expr = np.clip(one_euro(expr, fps), 0.0, 1.5)

    # Sapiens2 face refinement: jaw hinge from the chin-to-nose distance.
    jaw_theta = None
    if JAW_ENABLED:
        report("mocap: Sapiens2 face refinement")
        jaw_theta = _jaw_track(hub, frames, fps)

    report("mocap: exporting GLB")
    glob = quats
    jaw = by_name.get("c_jaw")
    if jaw_theta is not None and jaw is not None:
        head = by_name["c_head"]
        for f in range(n_frames):
            # Hinge about the head's local left axis (ear line).
            ax = _quat_rotate(glob[f, head], np.array([1.0, 0.0, 0.0]))
            half = jaw_theta[f] / 2.0
            q = np.array([*(ax * math.sin(half)), math.cos(half)])
            glob[f, jaw] = _quat_mul(q, glob[f, jaw])

    local_anim: dict[int, np.ndarray] = {}
    for j in range(n_joints):
        p = parents[j]
        local = np.zeros((n_frames, 4))
        for f in range(n_frames):
            local[f] = (
                glob[f, j]
                if p < 0
                else _quat_mul(_quat_conj(glob[f, p]), glob[f, j])
            )
        for f in range(1, n_frames):
            if np.dot(local[f], local[f - 1]) < 0:
                local[f] = -local[f]
        local_anim[j] = local.astype(np.float32)

    root = by_name["root"]
    root_parent = parents[root]
    rest_pos = char["rest_pos"]
    motion = cam_t - cam_t[0]
    root_trans = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        base = rest_pos[root] + motion[f]
        if root_parent >= 0:
            base = _quat_rotate(
                _quat_conj(rest_rot[root_parent]), base - rest_pos[root_parent]
            )
        root_trans[f] = base

    from mocap_retarget import _top4_skin

    w4, i4 = _top4_skin(char["skin_weights"], char["skin_indices"])
    times = np.arange(n_frames, dtype=np.float32) / fps

    morph = None
    weights = None
    if EXPR_ENABLED and char.get("face_shapes") is not None:
        morph = char["face_shapes"]
        weights = expr.astype(np.float32)

    return skinned_character_glb(
        vertices=char["vertices"].astype(np.float32),
        faces=char["faces"].astype(np.uint32),
        joints_weights=w4,
        joints_indices=i4,
        joint_names=char["names"],
        parents=parents,
        rest_local_pos=_rest_local(char)[0],
        rest_local_rot=_rest_local(char)[1],
        rest_global_pos=rest_pos.astype(np.float32),
        rest_global_rot=rest_rot.astype(np.float32),
        rotation_channels=local_anim,
        translation_channels={root: root_trans},
        times=times,
        morph_targets=morph,
        morph_weights=weights,
    )


def _rest_local(char) -> tuple[np.ndarray, np.ndarray]:
    parents = char["parents"]
    rest_pos, rest_rot = char["rest_pos"], char["rest_rot"]
    n = len(parents)
    lp = np.zeros((n, 3), dtype=np.float32)
    lr = np.zeros((n, 4), dtype=np.float32)
    for j in range(n):
        p = parents[j]
        if p < 0:
            lp[j], lr[j] = rest_pos[j], rest_rot[j]
        else:
            inv = _quat_conj(rest_rot[p])
            lp[j] = _quat_rotate(inv, rest_pos[j] - rest_pos[p])
            lr[j] = _quat_mul(inv, rest_rot[j])
    return lp, lr


def _jaw_track(hub: rt.ModelHub, frames, fps: float) -> np.ndarray | None:
    """Jaw-open angle per frame from Sapiens2's facial landmarks."""
    meta = hub.pose_meta
    name2id = meta["keypoint_name2id"]
    need = ("tip_of_chin", "nose")
    if any(n not in name2id for n in need):
        return None

    n_frames = len(frames)
    boxes = None
    try:
        from mocap_pipeline import track_person

        boxes = track_person(hub, frames)
    except Exception:
        return None

    chin, nose, conf_ok = np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames)
    open_d = np.zeros(n_frames)
    for start in range(0, n_frames, MOCAP_BATCH):
        end = min(start + MOCAP_BATCH, n_frames)
        kpts, conf = rt.pose_infer_frames(hub, frames[start:end], boxes[start:end])
        for f in range(start, end):
            c = kpts[f - start, name2id["tip_of_chin"]]
            n = kpts[f - start, name2id["nose"]]
            open_d[f] = np.linalg.norm(c - n)
            conf_ok[f] = min(
                conf[f - start, name2id["tip_of_chin"]],
                conf[f - start, name2id["nose"]],
            )
    _ = chin, nose

    seen = conf_ok >= POSE_CONF_THR
    if seen.mean() < MIN_VISIBILITY:
        return None
    ref = float(np.percentile(open_d[seen], 10))
    theta = np.clip((open_d / max(ref, 1e-6) - 1.0) * 2.0, 0.0, 1.0) * JAW_MAX_RAD
    theta[~seen] = 0.0
    return one_euro(theta[:, None], fps)[:, 0]
