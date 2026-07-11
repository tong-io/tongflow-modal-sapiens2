"""Video motion capture: video -> skeletal-animation GLB.

Pipeline: frame extraction -> DETR person tracking (single subject, IoU
continuity) -> Sapiens2-pose 308 2D keypoints -> 3D lift by sampling the
Sapiens2-pointmap at each keypoint pixel -> One-Euro temporal smoothing ->
FK skeleton solve (body + finger rotations, face/jaw translation channels)
-> skinned bone-viz GLB via gltf_writer.

Body and hands become rotation channels on a canonical hierarchy built from
Goliath keypoint names; facial capture is expressed as translation-animated
leaf bones under the head (jaw, brows, eyelids, lips), which imports as
plain animated bones in DCC tools.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile

import cv2
import numpy as np

import sapiens2_runtime as rt
from gltf_writer import Joint, skeleton_animation_glb

MOCAP_FPS = float(os.environ.get("MOCAP_FPS", "24"))
MOCAP_MAX_SECONDS = float(os.environ.get("MOCAP_MAX_SECONDS", "60"))
MOCAP_BATCH = int(os.environ.get("MOCAP_BATCH", "4"))
MOCAP_MAX_DIM = int(os.environ.get("MOCAP_MAX_DIM", "1920"))
POSE_CONF_THR = float(os.environ.get("MOCAP_POSE_CONF_THR", "0.3"))
# A keypoint must be confidently seen in at least this fraction of frames to
# drive its bones; below that the bone holds its rest pose.
MIN_VISIBILITY = float(os.environ.get("MOCAP_MIN_VISIBILITY", "0.3"))
# One-Euro parameters (position smoothing, per joint channel)
ONE_EURO_MIN_CUTOFF = float(os.environ.get("MOCAP_ONE_EURO_MIN_CUTOFF", "1.5"))
ONE_EURO_BETA = float(os.environ.get("MOCAP_ONE_EURO_BETA", "0.3"))

# ------------------------------------------------------------------ skeleton

_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")

# (bone name, goliath keypoint name or virtual spec, parent bone name)
# Virtual markers: pelvis/neck/spine/head are derived from raw keypoints.
_BODY = [
    ("pelvis", None, None),
    ("spine", None, "pelvis"),
    ("neck", None, "spine"),
    ("head", None, "neck"),
    ("left_hip", "left_hip", "pelvis"),
    ("left_knee", "left_knee", "left_hip"),
    ("left_ankle", "left_ankle", "left_knee"),
    ("left_foot", "left_big_toe", "left_ankle"),
    ("left_heel", "left_heel", "left_ankle"),
    ("right_hip", "right_hip", "pelvis"),
    ("right_knee", "right_knee", "right_hip"),
    ("right_ankle", "right_ankle", "right_knee"),
    ("right_foot", "right_big_toe", "right_ankle"),
    ("right_heel", "right_heel", "right_ankle"),
    ("left_shoulder", "left_shoulder", "neck"),
    ("left_elbow", "left_elbow", "left_shoulder"),
    ("left_wrist", "left_wrist", "left_elbow"),
    ("right_shoulder", "right_shoulder", "neck"),
    ("right_elbow", "right_elbow", "right_shoulder"),
    ("right_wrist", "right_wrist", "right_elbow"),
]

# Face capture channels: leaf bones under the head, animated by translation.
# Values are Goliath keypoint names; missing ones are skipped defensively.
_FACE = {
    "jaw": "tip_of_chin",
    "nose_tip": "tip_of_nose",
    "brow_left": "upper_midpoint_2_of_l_eyebrow",
    "brow_right": "upper_midpoint_2_of_r_eyebrow",
    "eyelid_upper_left": "l_centerpoint_of_upper_eyelid_line",
    "eyelid_lower_left": "l_centerpoint_of_lower_eyelid_line",
    "eyelid_upper_right": "r_centerpoint_of_upper_eyelid_line",
    "eyelid_lower_right": "r_centerpoint_of_lower_eyelid_line",
    "mouth_left": "l_outer_corner_of_mouth",
    "mouth_right": "r_outer_corner_of_mouth",
    # No center_of_upper_outer_lip in Goliath; use its middle midpoint.
    "lip_upper": "midpoint_1_of_upper_outer_lip",
    "lip_lower": "center_of_lower_outer_lip",
}


def _hand_chains(side: str) -> list[tuple[str, str, str]]:
    out = []
    for finger in _FINGERS:
        prev = f"{side}_wrist"
        for seg, suffix in (
            ("base", "_third_joint"),
            ("mid", "2"),
            ("dist", "3"),
            ("tip", "4"),
        ):
            name = f"{side}_{finger}_{seg}"
            out.append((name, f"{side}_{finger}{suffix}", prev))
            prev = name
    return out


# --------------------------------------------------------------- quaternions
# Quaternions are (x, y, z, w), matching glTF.


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[:3]


def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking direction a to direction b."""
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -1.0 + 1e-9:  # opposite: rotate 180° about any perpendicular
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + d])
    return q / np.linalg.norm(q)


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


def _triad(
    p_rest: np.ndarray, s_rest: np.ndarray, p_cur: np.ndarray, s_cur: np.ndarray
) -> np.ndarray:
    """Rotation aligning a (primary, secondary) rest frame to the current one."""

    def frame(p, s):
        x = p / max(np.linalg.norm(p), 1e-9)
        y = s - np.dot(s, x) * x
        n = np.linalg.norm(y)
        if n < 1e-6:
            y = np.array([0.0, 1.0, 0.0]) - x[1] * x
            n = max(np.linalg.norm(y), 1e-9)
        y /= n
        z = np.cross(x, y)
        return np.stack([x, y, z], axis=1)

    return _quat_from_matrix(frame(p_cur, s_cur) @ frame(p_rest, s_rest).T)


# ------------------------------------------------------------------ filters


def one_euro(x: np.ndarray, fps: float) -> np.ndarray:
    """One-Euro filter over axis 0 of (F, ...) data."""
    min_cutoff, beta, d_cutoff = ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, 1.0

    def alpha(cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    out = np.empty_like(x)
    out[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    a_d = alpha(d_cutoff)
    for i in range(1, len(x)):
        dx = (x[i] - out[i - 1]) * fps
        dx_prev = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_prev)
        a = 1.0 / (1.0 + (fps / (2 * math.pi * cutoff)))
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along axis 0; hold edges."""
    out = x.copy()
    idx = np.arange(len(x))
    flat = out.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = np.isnan(col)
        if bad.all():
            flat[:, c] = 0.0
        elif bad.any():
            flat[bad, c] = np.interp(idx[bad], idx[~bad], col[~bad])
    return out


# ---------------------------------------------------------------- extraction


def extract_frames(video_bytes: bytes) -> tuple[list[np.ndarray], float]:
    """Decode to BGR frames at MOCAP_FPS, capped at MOCAP_MAX_SECONDS."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("could not decode input video")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = min(MOCAP_FPS, src_fps)
        step = src_fps / fps
        max_frames = int(MOCAP_MAX_SECONDS * fps)
        frames: list[np.ndarray] = []
        next_pick = 0.0
        i = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= next_pick:
                h, w = frame.shape[:2]
                scale = MOCAP_MAX_DIM / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(
                        frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA
                    )
                frames.append(frame)
                next_pick += step
            i += 1
        cap.release()
        if len(frames) < 2:
            raise RuntimeError("video too short for motion capture (need >= 2 frames)")
        return frames, fps
    finally:
        os.unlink(path)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(area, 1e-9)


def track_person(hub: rt.ModelHub, frames: list[np.ndarray]) -> list[np.ndarray]:
    """One bbox per frame: largest person, then IoU continuity."""
    boxes: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for start in range(0, len(frames), MOCAP_BATCH):
        chunk = frames[start : start + MOCAP_BATCH]
        for dets in rt.detect_persons_batch(hub, chunk):
            if prev is None:
                areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                pick = dets[int(np.argmax(areas * (dets[:, 4] + 0.1)))]
            else:
                ious = np.array([_iou(prev, d) for d in dets])
                pick = dets[int(np.argmax(ious))] if ious.max() > 0.1 else prev
            boxes.append(pick[:4])
            prev = pick[:4]
    return boxes


# ------------------------------------------------------------------- capture


def capture(hub: rt.ModelHub, video_bytes: bytes, progress=None) -> bytes:
    """Full pipeline: video bytes -> animated GLB bytes."""

    def report(msg: str) -> None:
        if progress is not None:
            progress(msg)

    report("mocap: extracting frames")
    frames, fps = extract_frames(video_bytes)
    n_frames = len(frames)
    h, w = frames[0].shape[:2]

    report(f"mocap: tracking person across {n_frames} frames")
    boxes = track_person(hub, frames)

    meta = hub.pose_meta
    name2id = meta["keypoint_name2id"]
    n_kpts = meta["num_keypoints"]

    kpts_2d = np.zeros((n_frames, n_kpts, 2), dtype=np.float64)
    conf = np.zeros((n_frames, n_kpts), dtype=np.float64)
    points_3d = np.full((n_frames, n_kpts, 3), np.nan)

    for start in range(0, n_frames, MOCAP_BATCH):
        report(f"mocap: pose+depth {start}/{n_frames}")
        end = min(start + MOCAP_BATCH, n_frames)
        chunk = frames[start:end]
        kpts_2d[start:end], conf[start:end] = rt.pose_infer_frames(
            hub, chunk, boxes[start:end]
        )
        pms = rt.pointmap_metric_frames(hub, chunk)  # (B, H, W, 3) camera coords
        for f in range(start, end):
            pm = pms[f - start]
            u = np.clip(kpts_2d[f, :, 0], 0, w - 1)
            v = np.clip(kpts_2d[f, :, 1], 0, h - 1)
            u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
            u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
            fu, fv = (u - u0)[:, None], (v - v0)[:, None]
            sample = (
                pm[v0, u0] * (1 - fu) * (1 - fv)
                + pm[v0, u1] * fu * (1 - fv)
                + pm[v1, u0] * (1 - fu) * fv
                + pm[v1, u1] * fu * fv
            )
            ok = conf[f] >= POSE_CONF_THR
            points_3d[f, ok] = sample[ok]

    # Pointmap depth on small finger keypoints is noisy (background bleed):
    # clamp each hand's depth to its wrist depth +- a hand-sized band.
    for side in ("left", "right"):
        wrist = name2id.get(f"{side}_wrist")
        hand_ids = [
            i
            for n, i in name2id.items()
            if n.startswith(f"{side}_") and ("finger" in n or "thumb" in n or "hand" in n)
        ]
        if wrist is None or not hand_ids:
            continue
        wz = points_3d[:, wrist, 2][:, None]
        points_3d[:, hand_ids, 2] = np.clip(
            points_3d[:, hand_ids, 2], wz - 0.2, wz + 0.2
        )

    # Camera coords are y-down / z-forward; glTF wants y-up.
    points_3d *= np.array([1.0, -1.0, -1.0])

    # Fraction of frames each keypoint was confidently observed. Keypoints
    # that stay out of frame (e.g. legs in a head-and-shoulders video) must
    # not drive bones — their tracks are extrapolation garbage.
    visibility = (conf >= POSE_CONF_THR).mean(axis=0)

    report("mocap: smoothing and solving skeleton")
    points_3d = _interp_nan(points_3d)
    points_3d = one_euro(points_3d, fps)

    style = os.environ.get("MOCAP_STYLE", "mhr")
    if style == "mhr":
        import mocap_retarget

        report("mocap: retargeting onto MHR")
        glb = mocap_retarget.export_mhr_glb(points_3d, name2id, fps, visibility)
    else:  # "skeleton": the plain bone-puppet exporter
        glb = _solve_and_export(points_3d, name2id, fps, visibility)
    report("mocap: done")
    return glb


# ----------------------------------------------------------------- solve/export


def _marker_positions(
    points: np.ndarray, name2id: dict[str, int]
) -> dict[str, np.ndarray]:
    """(F, K, 3) raw keypoints -> named marker tracks incl. virtual joints."""

    def kp(name: str) -> np.ndarray:
        return points[:, name2id[name]]

    m: dict[str, np.ndarray] = {}
    for _, kp_name, _ in _BODY + _hand_chains("left") + _hand_chains("right"):
        if kp_name is not None and kp_name in name2id:
            m[kp_name] = kp(kp_name)
    m["pelvis"] = (kp("left_hip") + kp("right_hip")) / 2
    m["neck"] = (kp("left_shoulder") + kp("right_shoulder")) / 2
    m["spine"] = m["pelvis"] * 0.5 + m["neck"] * 0.5
    m["nose"] = kp("nose")
    ears_ok = "left_ear" in name2id and "right_ear" in name2id
    if ears_ok:
        m["left_ear"], m["right_ear"] = kp("left_ear"), kp("right_ear")
        m["head"] = (m["left_ear"] + m["right_ear"]) / 2
    else:
        m["head"] = m["nose"]
    for kp_name in _FACE.values():
        if kp_name in name2id:
            m[kp_name] = kp(kp_name)
    return m


class SolvedSkeleton:
    """Result of the marker -> rotation solve, shared by both exporters."""

    def __init__(self, bone_defs, index, tracks, rest, global_rot, markers, driven):
        self.bone_defs = bone_defs  # [(bone, marker, parent-bone | None)]
        self.index = index  # bone name -> row
        self.tracks = tracks  # list of (F, 3) marker positions
        self.rest = rest  # (B, 3) rest positions
        self.global_rot = global_rot  # (F, B, 4) global quats vs rest
        self.markers = markers  # marker name -> (F, 3)
        self.driven = driven  # (B,) bool: solved from observed markers


def _marker_visibility(
    visibility: np.ndarray | None, name2id: dict[str, int]
) -> dict[str, float]:
    """Per-marker observed fraction, incl. the virtual body markers."""
    if visibility is None:
        return {}
    vis = {n: float(visibility[i]) for n, i in name2id.items()}

    def lo(*names: str) -> float:
        return min(vis.get(n, 0.0) for n in names)

    vis["pelvis"] = lo("left_hip", "right_hip")
    vis["neck"] = lo("left_shoulder", "right_shoulder")
    vis["spine"] = min(vis["pelvis"], vis["neck"])
    vis["head"] = (
        lo("left_ear", "right_ear") if "left_ear" in name2id else vis.get("nose", 0.0)
    )
    return vis


def solve_skeleton(
    points: np.ndarray,
    name2id: dict[str, int],
    rest_positions: dict[str, np.ndarray] | None = None,
    head_rest_frame: tuple[np.ndarray, np.ndarray] | None = None,
    visibility: np.ndarray | None = None,
) -> SolvedSkeleton:
    """Solve per-frame global joint rotations from 3D keypoints.

    ``rest_positions`` overrides the rest pose per bone (e.g. with a target
    character's bind pose so the rotations retarget directly); by default the
    rest pose is derived from frame 0 with median bone lengths.
    ``head_rest_frame`` supplies (nose-direction, ear-line) rest vectors for
    the head triad when the rest pose is not frame-0-derived.
    """
    n_frames = len(points)
    markers = _marker_positions(points, name2id)

    # Assemble the joint list (topological order).
    bone_defs: list[tuple[str, str, str | None]] = []  # (bone, marker, parent bone)
    for bone, kp_name, parent in _BODY:
        marker = kp_name if kp_name is not None else bone
        bone_defs.append((bone, marker, parent))
    body_names = {b for b, _, _ in bone_defs}
    for side in ("left", "right"):
        chain_names = set(body_names)
        for bone, kp_name, parent in _hand_chains(side):
            # Skip a segment whose keypoint OR parent segment is unavailable —
            # keeps every appended bone's parent resolvable.
            if kp_name in markers and parent in chain_names:
                bone_defs.append((bone, kp_name, parent))
                chain_names.add(bone)

    index: dict[str, int] = {b: i for i, (b, _, _) in enumerate(bone_defs)}
    tracks = [markers[m] for _, m, _ in bone_defs]  # list of (F, 3)

    rest = np.zeros((len(bone_defs), 3))
    if rest_positions is not None:
        for i, (bone, _, _) in enumerate(bone_defs):
            rest[i] = rest_positions[bone]
    else:
        # Rest pose: median-length bones along frame-0 directions keep
        # proportions stable; positions come from the first frame.
        rest[0] = tracks[0][0]
        for i, (_, _, parent) in enumerate(bone_defs):
            if parent is None:
                continue
            p = index[parent]
            offsets = tracks[i] - tracks[p]
            length = float(np.median(np.linalg.norm(offsets, axis=1)))
            d0 = offsets[0]
            d0 /= max(np.linalg.norm(d0), 1e-9)
            rest[i] = rest[p] + d0 * length

    # Secondary-axis pairs for stable twist at multi-child joints.
    triad_secondary = {
        "pelvis": ("left_hip", "right_hip"),
        "spine": ("left_shoulder", "right_shoulder"),
        "neck": ("left_shoulder", "right_shoulder"),
        "left_wrist": ("left_forefinger_base", "left_pinky_finger_base"),
        "right_wrist": ("right_forefinger_base", "right_pinky_finger_base"),
    }
    primary_child = {
        "pelvis": "spine",
        "spine": "neck",
        "neck": "head",
        "left_wrist": "left_middle_finger_base",
        "right_wrist": "right_middle_finger_base",
    }

    children: dict[int, list[int]] = {i: [] for i in range(len(bone_defs))}
    for i, (_, _, parent) in enumerate(bone_defs):
        if parent is not None:
            children[index[parent]].append(i)

    ident = np.array([0.0, 0.0, 0.0, 1.0])
    global_rot = np.tile(ident, (n_frames, len(bone_defs), 1))

    # A bone only gets a solved rotation when its own marker and its primary
    # child's marker were actually observed; otherwise it inherits the parent
    # (== holds its rest pose relative to it).
    mvis = _marker_visibility(visibility, name2id)

    def seen(marker: str) -> bool:
        return visibility is None or mvis.get(marker, 0.0) >= MIN_VISIBILITY

    driven = np.zeros(len(bone_defs), dtype=bool)

    head_markers = (
        ("nose" in markers, "left_ear" in markers and "right_ear" in markers)
    )
    for f in range(n_frames):
        for i, (bone, marker, parent) in enumerate(bone_defs):
            kids = children[i]
            if bone == "head" and head_markers[0] and head_markers[1]:
                if not (seen("nose") and seen("left_ear") and seen("right_ear")):
                    global_rot[f, i] = (
                        global_rot[f, index[parent]] if parent is not None else ident
                    )
                    continue
                driven[i] = True
                # Skull orientation from the nose direction + ear line — the
                # head bone is a leaf, so child bones can't orient it.
                if head_rest_frame is not None:
                    p_rest, s_rest = head_rest_frame
                else:
                    p_rest = markers["nose"][0] - rest[i]
                    s_rest = markers["left_ear"][0] - markers["right_ear"][0]
                p_cur = markers["nose"][f] - tracks[i][f]
                s_cur = markers["left_ear"][f] - markers["right_ear"][f]
                global_rot[f, i] = _triad(p_rest, s_rest, p_cur, s_cur)
                continue
            if not kids:
                # Leaf: inherit parent's global rotation.
                global_rot[f, i] = (
                    global_rot[f, index[parent]] if parent is not None else ident
                )
                continue
            if bone in primary_child and index.get(primary_child[bone]) in kids:
                prim = index[primary_child[bone]]
            else:
                prim = kids[0]
            if not (seen(marker) and seen(bone_defs[prim][1])):
                global_rot[f, i] = (
                    global_rot[f, index[parent]] if parent is not None else ident
                )
                continue
            driven[i] = True
            p_rest = rest[prim] - rest[i]
            p_cur = tracks[prim][f] - tracks[i][f]
            if bone in triad_secondary:
                a, c = triad_secondary[bone]
                if (
                    a in index
                    and c in index
                    and seen(bone_defs[index[a]][1])
                    and seen(bone_defs[index[c]][1])
                ):
                    s_rest = rest[index[a]] - rest[index[c]]
                    s_cur = tracks[index[a]][f] - tracks[index[c]][f]
                    global_rot[f, i] = _triad(p_rest, s_rest, p_cur, s_cur)
                    continue
            global_rot[f, i] = _quat_from_two_vectors(p_rest, p_cur)

    # A leaf is only as reliable as the joint it inherits from.
    for i, (_, _, parent) in enumerate(bone_defs):
        if not children[i] and parent is not None:
            driven[i] = driven[index[parent]]

    return SolvedSkeleton(bone_defs, index, tracks, rest, global_rot, markers, driven)


def _solve_and_export(
    points: np.ndarray,
    name2id: dict[str, int],
    fps: float,
    visibility: np.ndarray | None = None,
) -> bytes:
    n_frames = len(points)
    solved = solve_skeleton(points, name2id, visibility=visibility)
    bone_defs = solved.bone_defs
    index = solved.index
    tracks = solved.tracks
    rest = solved.rest
    global_rot = solved.global_rot
    markers = solved.markers

    mvis = _marker_visibility(visibility, name2id)
    face_bones: list[tuple[str, str]] = [
        (bone, kp_name)
        for bone, kp_name in _FACE.items()
        if kp_name in markers
        and (visibility is None or mvis.get(kp_name, 0.0) >= MIN_VISIBILITY)
    ]

    # Local rotations; keep quaternion sign continuity for clean LERP.
    joints: list[Joint] = []
    for i, (bone, _, parent) in enumerate(bone_defs):
        local = np.zeros((n_frames, 4))
        for f in range(n_frames):
            if parent is None:
                local[f] = global_rot[f, i]
            else:
                local[f] = _quat_mul(
                    _quat_conj(global_rot[f, index[parent]]), global_rot[f, i]
                )
        for f in range(1, n_frames):
            if np.dot(local[f], local[f - 1]) < 0:
                local[f] = -local[f]
        j = Joint(
            name=bone,
            parent=index[parent] if parent is not None else -1,
            rest=rest[i],
            rotations=local,
        )
        if parent is None:
            # Root motion anchor: the pelvis when observed, else the neck
            # (e.g. head-and-shoulders videos never see the hips).
            anchor = (
                tracks[i]
                if solved.driven[i]
                else rest[i] + (markers["neck"] - markers["neck"][0])
            )
            j.translations = anchor.astype(np.float32)
        joints.append(j)

    # Face channels: translation-animated leaves under the head, expressed in
    # the head's FK frame so they follow the skull and add expression motion.
    head_idx = index["head"]
    head_fk_pos = _fk_positions(joints, n_frames)[:, head_idx]
    for bone, kp_name in face_bones:
        track = markers[kp_name]
        local = np.zeros((n_frames, 3))
        for f in range(n_frames):
            local[f] = _quat_rotate(
                _quat_conj(global_rot[f, head_idx]), track[f] - head_fk_pos[f]
            )
        rest_pos = rest[head_idx] + local[0]
        joints.append(
            Joint(
                name=f"face_{bone}",
                parent=head_idx,
                rest=rest_pos,
                translations=local.astype(np.float32),
            )
        )

    times = np.arange(n_frames, dtype=np.float32) / fps
    return skeleton_animation_glb(joints, times)


def _fk_positions(joints: list[Joint], n_frames: int) -> np.ndarray:
    """Global joint positions implied by the exported channels (F, J, 3)."""
    n = len(joints)
    pos = np.zeros((n_frames, n, 3))
    rot = np.zeros((n_frames, n, 4))
    for i, j in enumerate(joints):
        rest_local = j.rest - (joints[j.parent].rest if j.parent >= 0 else 0.0)
        for f in range(n_frames):
            r = j.rotations[f] if j.rotations is not None else np.array([0, 0, 0, 1.0])
            t = j.translations[f] if j.translations is not None else rest_local
            if j.parent < 0:
                pos[f, i] = t
                rot[f, i] = r
            else:
                p = j.parent
                pos[f, i] = pos[f, p] + _quat_rotate(rot[f, p], t)
                rot[f, i] = _quat_mul(rot[f, p], r)
    return pos
