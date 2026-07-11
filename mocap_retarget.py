"""Retarget solved mocap rotations onto Meta's MHR character (Apache-2.0).

The character bundle (rest mesh + skin weights + skeleton with bind-pose
global transforms) is extracted once by ``extract_mhr.py`` into
``$SAPIENS2_WEIGHTS/mhr/mhr_lod{N}.npz`` — this module is pure numpy.

Retarget principle: the solver is run with MHR's own bind pose as the rest
skeleton, so the solved per-frame global rotations are *deltas from MHR's
bind pose* and can be applied verbatim: ``R_joint(t) = delta(t) ∘ R_bind``.
Unmapped joints (twists, clavicles, metacarpals) keep their bind-pose local
rotation and simply follow their parent.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from gltf_writer import skinned_character_glb
from mocap_pipeline import (
    MIN_VISIBILITY,
    _marker_visibility,
    _quat_conj,
    _quat_mul,
    _quat_rotate,
    solve_skeleton,
)

MHR_LOD = int(os.environ.get("MHR_LOD", "3"))
MHR_NPZ = os.environ.get("MHR_NPZ", f"/models/mhr/mhr_lod{MHR_LOD}.npz")
# Legacy location used before the bundle became plugin-neutral.
MHR_NPZ_FALLBACK = (
    f"{os.environ.get('SAPIENS2_WEIGHTS', '/models/sapiens2')}/mhr/mhr_lod{MHR_LOD}.npz"
)
# Momentum works in centimeters; export in meters.
MHR_SCALE = float(os.environ.get("MHR_SCALE", "0.01"))
JAW_ENABLED = os.environ.get("MOCAP_JAW", "1") != "0"
JAW_MAX_RAD = 0.45

# Our solver bone -> MHR joint driven by its rotation. The same map provides
# the rest positions the solver uses (so deltas apply directly).
BONE_TO_MHR = {
    "pelvis": "root",
    "spine": "c_spine1",
    "neck": "c_neck",
    "head": "c_head",
    "left_hip": "l_upleg",
    "left_knee": "l_lowleg",
    "left_ankle": "l_foot",
    "left_foot": "l_ball",
    "right_hip": "r_upleg",
    "right_knee": "r_lowleg",
    "right_ankle": "r_foot",
    "right_foot": "r_ball",
    "left_shoulder": "l_uparm",
    "left_elbow": "l_lowarm",
    "left_wrist": "l_wrist",
    "right_shoulder": "r_uparm",
    "right_elbow": "r_lowarm",
    "right_wrist": "r_wrist",
}
_FINGER_TO_MHR = {"thumb": "thumb", "forefinger": "index", "middle_finger": "middle", "ring_finger": "ring", "pinky_finger": "pinky"}
for side, m_side in (("left", "l"), ("right", "r")):
    for ours, mhr in _FINGER_TO_MHR.items():
        for seg, num in (("base", 1), ("mid", 2), ("dist", 3)):
            BONE_TO_MHR[f"{side}_{ours}_{seg}"] = f"{m_side}_{mhr}{num}"
        BONE_TO_MHR[f"{side}_{ours}_tip"] = f"{m_side}_{mhr}_null"

# Bones whose marker has no MHR counterpart: rest position approximated from
# an MHR joint (heels sit below/behind the ankle; use the foot joint).
_REST_FALLBACK = {"left_heel": "l_foot", "right_heel": "r_foot"}


@lru_cache(maxsize=1)
def load_character() -> dict:
    path = MHR_NPZ if os.path.exists(MHR_NPZ) else MHR_NPZ_FALLBACK
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["joint_names"]]
    char = {
        "names": names,
        "by_name": {n: i for i, n in enumerate(names)},
        "parents": d["parents"].astype(int),
        "rest_pos": d["rest_positions"] * MHR_SCALE,
        "rest_rot": d["rest_rotations"],
        "vertices": d["vertices"] * MHR_SCALE,
        "faces": d["faces"],
        "skin_weights": d["skin_weights"],
        "skin_indices": d["skin_indices"],
        # 72 face-expression blendshape deltas (bundle v3+); None on old bundles.
        "face_shapes": (
            d["face_shape_vectors"] * MHR_SCALE
            if "face_shape_vectors" in d.files
            else None
        ),
    }
    return char


def _top4_skin(weights: np.ndarray, indices: np.ndarray):
    """Reduce (V, K) influences to glTF's 4, renormalized."""
    order = np.argsort(-weights, axis=1)[:, :4]
    rows = np.arange(len(weights))[:, None]
    w4 = np.take_along_axis(weights, order, axis=1)
    i4 = np.take_along_axis(indices, order, axis=1)
    s = w4.sum(axis=1, keepdims=True)
    w4 = np.where(s > 1e-8, w4 / np.maximum(s, 1e-8), np.array([1.0, 0, 0, 0]))
    _ = rows
    return w4.astype(np.float32), i4.astype(np.uint16)


def _body_axes(char: dict) -> tuple[np.ndarray, np.ndarray]:
    """(forward, left) unit vectors of the bind-pose character."""
    left = char["rest_pos"][char["by_name"]["l_uparm"]] - char["rest_pos"][
        char["by_name"]["r_uparm"]
    ]
    left = left / max(np.linalg.norm(left), 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    # y-up right-handed: a character facing +z has its left side at +x, so
    # forward = left x up. Verified against the MHR bind pose (eyes/teeth/
    # toes all sit at positive z while l_* joints sit at positive x).
    fwd = np.cross(left, up)
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    return fwd, left


def export_mhr_glb(
    points: np.ndarray,
    name2id: dict[str, int],
    fps: float,
    visibility: np.ndarray | None = None,
) -> bytes:
    char = load_character()
    by_name = char["by_name"]
    parents = char["parents"]
    rest_pos = char["rest_pos"]
    rest_rot = char["rest_rot"]
    n_joints = len(char["names"])
    n_frames = len(points)

    fwd, left = _body_axes(char)
    head_pos = rest_pos[by_name["c_head"]]
    rest_bones = {}
    for bone, mhr in BONE_TO_MHR.items():
        rest_bones[bone] = rest_pos[by_name[mhr]]
    for bone, mhr in _REST_FALLBACK.items():
        rest_bones[bone] = rest_pos[by_name[mhr]] - fwd * 0.07

    solved = solve_skeleton(
        points,
        name2id,
        rest_positions=rest_bones,
        # Head triad rest: nose points forward, ear line points left.
        head_rest_frame=(fwd * 0.1, left * 0.15),
        visibility=visibility,
    )
    _ = head_pos

    # Per-frame global rotations for every MHR joint: mapped joints get the
    # solved delta applied to their bind rotation; unmapped joints keep their
    # bind local rotation under the animated parent. Bones the solver never
    # observed (out-of-frame body parts) are left unmapped so the character
    # holds its rest pose there.
    mhr_from_bone = {}
    for bone, mhr in BONE_TO_MHR.items():
        i = solved.index.get(bone)
        if i is not None and solved.driven[i]:
            mhr_from_bone[by_name[mhr]] = i

    rest_local_rot = np.zeros((n_joints, 4))
    rest_local_pos = np.zeros((n_joints, 3))
    for j in range(n_joints):
        p = parents[j]
        if p < 0:
            rest_local_rot[j] = rest_rot[j]
            rest_local_pos[j] = rest_pos[j]
        else:
            inv_p = _quat_conj(rest_rot[p])
            rest_local_rot[j] = _quat_mul(inv_p, rest_rot[j])
            rest_local_pos[j] = _quat_rotate(inv_p, rest_pos[j] - rest_pos[p])

    ident = np.array([0.0, 0.0, 0.0, 1.0])
    glob = np.tile(ident, (n_frames, n_joints, 1))
    for f in range(n_frames):
        for j in range(n_joints):
            p = parents[j]
            if j in mhr_from_bone:
                delta = solved.global_rot[f, mhr_from_bone[j]]
                glob[f, j] = _quat_mul(delta, rest_rot[j])
            elif p < 0:
                glob[f, j] = rest_rot[j]
            else:
                glob[f, j] = _quat_mul(glob[f, p], rest_local_rot[j])

    # Jaw open from the chin-to-nose distance, hinged on the current ear axis.
    mvis = _marker_visibility(visibility, name2id)
    face_seen = visibility is None or (
        mvis.get("tip_of_chin", 0.0) >= MIN_VISIBILITY
        and mvis.get("nose", 0.0) >= MIN_VISIBILITY
    )
    jaw = by_name.get("c_jaw")
    if JAW_ENABLED and face_seen and jaw is not None:
        m = solved.markers
        if "tip_of_chin" in m and "nose" in m and "left_ear" in m:
            open_d = np.linalg.norm(m["tip_of_chin"] - m["nose"], axis=1)
            ref = float(np.percentile(open_d, 10))  # near-closed baseline
            theta = np.clip((open_d / max(ref, 1e-6) - 1.0) * 2.0, 0.0, 1.0)
            theta *= JAW_MAX_RAD
            ear_axis = m["left_ear"] - m["right_ear"]
            for f in range(n_frames):
                ax = ear_axis[f] / max(np.linalg.norm(ear_axis[f]), 1e-9)
                half = theta[f] / 2.0
                q = np.array([*(ax * np.sin(half)), np.cos(half)])
                glob[f, jaw] = _quat_mul(q, glob[f, jaw])

    # Locals + root translation channel.
    animated = sorted(set(mhr_from_bone) | ({jaw} if JAW_ENABLED and jaw else set()))
    local_anim = {}
    for j in animated:
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
    # Root motion anchor: pelvis when observed, else the neck (e.g. upper-body
    # videos never see the hips).
    if solved.driven[solved.index["pelvis"]]:
        anchor_track = solved.tracks[solved.index["pelvis"]]
    else:
        anchor_track = solved.markers["neck"]
    root_parent = parents[root]
    root_trans = np.zeros((n_frames, 3), dtype=np.float32)
    motion = anchor_track - anchor_track[0]  # relative subject motion, meters
    for f in range(n_frames):
        base = rest_pos[root] + motion[f]
        if root_parent >= 0:
            base = _quat_rotate(
                _quat_conj(rest_rot[root_parent]), base - rest_pos[root_parent]
            )
        root_trans[f] = base

    w4, i4 = _top4_skin(char["skin_weights"], char["skin_indices"])
    times = np.arange(n_frames, dtype=np.float32) / fps

    return skinned_character_glb(
        vertices=char["vertices"].astype(np.float32),
        faces=char["faces"].astype(np.uint32),
        joints_weights=w4,
        joints_indices=i4,
        joint_names=char["names"],
        parents=parents,
        rest_local_pos=rest_local_pos.astype(np.float32),
        rest_local_rot=rest_local_rot.astype(np.float32),
        rest_global_pos=rest_pos.astype(np.float32),
        rest_global_rot=rest_rot.astype(np.float32),
        rotation_channels=local_anim,
        translation_channels={root: root_trans},
        times=times,
    )
