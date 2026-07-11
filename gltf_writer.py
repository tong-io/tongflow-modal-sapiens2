"""Minimal binary glTF (.glb) writer — no dependencies beyond numpy.

Two entry points:
- ``skeleton_animation_glb``: a skinned bone-visualization mesh driven by a
  joint hierarchy with per-frame rotation/translation animation. Imports as a
  proper armature in Blender and auto-plays in three.js AnimationMixer.
- ``point_cloud_glb``: a colored POINTS primitive.

Conventions: y-up right-handed (glTF standard); quaternions are (x, y, z, w).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

_COMP_F32 = 5126
_COMP_U16 = 5123
_COMP_U8 = 5121
_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942


class _Builder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def _pad(self, align: int = 4) -> None:
        while len(self.blob) % align:
            self.blob.append(0)

    def add_view(self, data: bytes, target: int | None = None) -> int:
        self._pad()
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.blob.extend(data)
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        array: np.ndarray,
        component_type: int,
        type_str: str,
        *,
        target: int | None = None,
        normalized: bool = False,
        minmax: bool = False,
    ) -> int:
        view = self.add_view(array.tobytes(), target)
        acc: dict = {
            "bufferView": view,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": type_str,
        }
        if normalized:
            acc["normalized"] = True
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def finish(self, gltf: dict) -> bytes:
        self._pad()
        gltf["buffers"] = [{"byteLength": len(self.blob)}]
        gltf["bufferViews"] = self.buffer_views
        gltf["accessors"] = self.accessors
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.blob)
        out = struct.pack("<III", _GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(js), _CHUNK_JSON) + js
        out += struct.pack("<II", len(self.blob), _CHUNK_BIN) + bytes(self.blob)
        return out


@dataclass
class Joint:
    """One joint of the export skeleton.

    ``rest`` is the rest-pose GLOBAL position (y-up). Channels are optional
    per-frame animation data: ``rotations`` (F, 4) local quaternions xyzw,
    ``translations`` (F, 3) local translations.
    """

    name: str
    parent: int  # -1 for root
    rest: np.ndarray
    rotations: np.ndarray | None = None
    translations: np.ndarray | None = None
    children: list[int] = field(default_factory=list)


def _octahedron(a: np.ndarray, b: np.ndarray, width: float) -> np.ndarray:
    """Bone-viz octahedron between global points a->b: (24, 3) triangle soup."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-8:
        d = np.array([0.0, 1e-3, 0.0])
        length = 1e-3
    axis = d / length
    # Any perpendicular frame
    up = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, up)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ring_c = a + d * 0.15
    w = width
    ring = [ring_c + u * w, ring_c + v * w, ring_c - u * w, ring_c - v * w]
    tris: list[np.ndarray] = []
    for i in range(4):
        r0, r1 = ring[i], ring[(i + 1) % 4]
        tris += [a, r1, r0]  # cap toward head
        tris += [r0, r1, b]  # long face toward tail
    return np.asarray(tris, dtype=np.float32)


def skeleton_animation_glb(
    joints: list[Joint],
    times: np.ndarray,
    *,
    bone_width_ratio: float = 0.08,
    color: tuple[float, float, float, float] = (0.65, 0.74, 0.86, 1.0),
) -> bytes:
    """Build a skinned, animated bone-visualization GLB from a joint list.

    Joints must be topologically ordered (parent before child).
    """
    n = len(joints)
    for i, j in enumerate(joints):
        if j.parent >= 0:
            joints[j.parent].children.append(i)

    # Rest local translations (rest global rotations are identity by design,
    # so the inverse bind matrix is a pure translation).
    rest_local = [
        j.rest - (joints[j.parent].rest if j.parent >= 0 else 0.0) for j in joints
    ]

    b = _Builder()

    # --- skinned bone mesh (triangle soup, rigid weight to the parent joint)
    positions: list[np.ndarray] = []
    joint_ids: list[int] = []
    lengths = [
        np.linalg.norm(j.rest - joints[j.parent].rest)
        for j in joints
        if j.parent >= 0
    ]
    median_len = float(np.median(lengths)) if lengths else 0.1
    for i, j in enumerate(joints):
        for c in j.children:
            tri = _octahedron(
                j.rest,
                joints[c].rest,
                max(
                    float(np.linalg.norm(joints[c].rest - j.rest))
                    * bone_width_ratio,
                    median_len * 0.02,
                ),
            )
            positions.append(tri)
            joint_ids += [i] * len(tri)
        if not j.children:  # leaf marker (face channel bones, fingertips)
            s = median_len * 0.06
            tri = _octahedron(
                j.rest - np.array([0.0, s, 0.0]),
                j.rest + np.array([0.0, s, 0.0]),
                s,
            )
            positions.append(tri)
            joint_ids += [i] * len(tri)

    pos = np.concatenate(positions, axis=0).astype(np.float32)
    vcount = len(pos)
    jnt = np.zeros((vcount, 4), dtype=np.uint16)
    jnt[:, 0] = np.asarray(joint_ids, dtype=np.uint16)
    wgt = np.zeros((vcount, 4), dtype=np.float32)
    wgt[:, 0] = 1.0

    pos_acc = b.add_accessor(pos, _COMP_F32, "VEC3", target=34962, minmax=True)
    jnt_acc = b.add_accessor(jnt, _COMP_U16, "VEC4", target=34962)
    wgt_acc = b.add_accessor(wgt, _COMP_F32, "VEC4", target=34962)

    ibm = np.zeros((n, 4, 4), dtype=np.float32)
    for i, j in enumerate(joints):
        m = np.eye(4, dtype=np.float32)
        m[3, :3] = -j.rest  # column-major storage: translation in last row
        ibm[i] = m
    ibm_acc = b.add_accessor(ibm.reshape(n, 16), _COMP_F32, "MAT4")

    # --- nodes: joints first (index == joint index), then the mesh node
    nodes: list[dict] = []
    for i, j in enumerate(joints):
        node: dict = {"name": j.name, "translation": [float(v) for v in rest_local[i]]}
        if j.children:
            node["children"] = list(j.children)
        nodes.append(node)
    mesh_node = {"name": "bones", "mesh": 0, "skin": 0}
    nodes.append(mesh_node)
    root_idx = next(i for i, j in enumerate(joints) if j.parent < 0)

    # --- animation
    t_acc = b.add_accessor(times.astype(np.float32).reshape(-1, 1), _COMP_F32, "SCALAR", minmax=True)
    samplers: list[dict] = []
    channels: list[dict] = []
    for i, j in enumerate(joints):
        if j.rotations is not None:
            out = b.add_accessor(j.rotations.astype(np.float32), _COMP_F32, "VEC4")
            samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": out})
            channels.append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": i, "path": "rotation"},
                }
            )
        if j.translations is not None:
            out = b.add_accessor(j.translations.astype(np.float32), _COMP_F32, "VEC3")
            samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": out})
            channels.append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": i, "path": "translation"},
                }
            )

    gltf = {
        "asset": {"version": "2.0", "generator": "tongflow-modal-sapiens2"},
        "scene": 0,
        "scenes": [{"nodes": [root_idx, len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos_acc,
                            "JOINTS_0": jnt_acc,
                            "WEIGHTS_0": wgt_acc,
                        },
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(color),
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.8,
                },
                "doubleSided": True,
            }
        ],
        "skins": [
            {
                "inverseBindMatrices": ibm_acc,
                "joints": list(range(n)),
                "skeleton": root_idx,
            }
        ],
        "animations": [
            {"name": "mocap", "samplers": samplers, "channels": channels}
        ],
    }
    return b.finish(gltf)


def _quat_to_mat3(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def skinned_character_glb(
    *,
    vertices: np.ndarray,  # (V, 3) model-space bind pose
    faces: np.ndarray,  # (T, 3)
    joints_weights: np.ndarray,  # (V, 4) float
    joints_indices: np.ndarray,  # (V, 4) uint16
    joint_names: list[str],
    parents: np.ndarray,  # (J,) int, -1 for root
    rest_local_pos: np.ndarray,  # (J, 3)
    rest_local_rot: np.ndarray,  # (J, 4) xyzw
    rest_global_pos: np.ndarray,  # (J, 3)
    rest_global_rot: np.ndarray,  # (J, 4) xyzw
    rotation_channels: dict[int, np.ndarray],  # joint -> (F, 4)
    translation_channels: dict[int, np.ndarray],  # joint -> (F, 3)
    times: np.ndarray,
    color: tuple[float, float, float, float] = (0.62, 0.71, 0.83, 1.0),
) -> bytes:
    """Skinned character with a full bind pose (rotations included) + clip."""
    n = len(joint_names)
    b = _Builder()

    pos_acc = b.add_accessor(
        vertices.astype(np.float32), _COMP_F32, "VEC3", target=34962, minmax=True
    )
    idx_acc = b.add_accessor(
        faces.astype(np.uint32).reshape(-1, 1), 5125, "SCALAR", target=34963
    )
    jnt_acc = b.add_accessor(
        joints_indices.astype(np.uint16), _COMP_U16, "VEC4", target=34962
    )
    wgt_acc = b.add_accessor(
        joints_weights.astype(np.float32), _COMP_F32, "VEC4", target=34962
    )

    ibm = np.zeros((n, 16), dtype=np.float32)
    for j in range(n):
        rot = _quat_to_mat3(rest_global_rot[j])
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = rest_global_pos[j]
        inv = np.eye(4)
        inv[:3, :3] = rot.T
        inv[:3, 3] = -rot.T @ rest_global_pos[j]
        # glTF stores matrices column-major; C-order flatten of the transpose.
        ibm[j] = inv.T.reshape(16)
    ibm_acc = b.add_accessor(ibm, _COMP_F32, "MAT4")

    nodes: list[dict] = []
    children: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        if p >= 0:
            children.setdefault(int(p), []).append(j)
    roots = [j for j, p in enumerate(parents) if p < 0]
    for j in range(n):
        node: dict = {
            "name": joint_names[j],
            "translation": [float(v) for v in rest_local_pos[j]],
            "rotation": [float(v) for v in rest_local_rot[j]],
        }
        if j in children:
            node["children"] = children[j]
        nodes.append(node)
    nodes.append({"name": "character", "mesh": 0, "skin": 0})

    t_acc = b.add_accessor(
        times.astype(np.float32).reshape(-1, 1), _COMP_F32, "SCALAR", minmax=True
    )
    samplers: list[dict] = []
    channels: list[dict] = []
    for j, quats in rotation_channels.items():
        out = b.add_accessor(quats.astype(np.float32), _COMP_F32, "VEC4")
        samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": out})
        channels.append(
            {"sampler": len(samplers) - 1, "target": {"node": int(j), "path": "rotation"}}
        )
    for j, trans in translation_channels.items():
        out = b.add_accessor(trans.astype(np.float32), _COMP_F32, "VEC3")
        samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": out})
        channels.append(
            {
                "sampler": len(samplers) - 1,
                "target": {"node": int(j), "path": "translation"},
            }
        )

    gltf = {
        "asset": {"version": "2.0", "generator": "tongflow-modal-sapiens2"},
        "scene": 0,
        "scenes": [{"nodes": [*roots, n]}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos_acc,
                            "JOINTS_0": jnt_acc,
                            "WEIGHTS_0": wgt_acc,
                        },
                        "indices": idx_acc,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(color),
                    "metallicFactor": 0.05,
                    "roughnessFactor": 0.75,
                },
                "doubleSided": True,
            }
        ],
        "skins": [
            {
                "inverseBindMatrices": ibm_acc,
                "joints": list(range(n)),
                "skeleton": int(roots[0]),
            }
        ],
        "animations": [{"name": "mocap", "samplers": samplers, "channels": channels}],
    }
    return b.finish(gltf)


def point_cloud_glb(points: np.ndarray, colors_rgb: np.ndarray) -> bytes:
    """Colored point cloud GLB. points (N,3) float; colors (N,3) uint8."""
    b = _Builder()
    pos_acc = b.add_accessor(
        points.astype(np.float32), _COMP_F32, "VEC3", target=34962, minmax=True
    )
    col_acc = b.add_accessor(
        colors_rgb.astype(np.uint8), _COMP_U8, "VEC3", target=34962, normalized=True
    )
    gltf = {
        "asset": {"version": "2.0", "generator": "tongflow-modal-sapiens2"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "pointmap", "mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "COLOR_0": col_acc},
                        "mode": 0,
                    }
                ]
            }
        ],
    }
    return b.finish(gltf)
