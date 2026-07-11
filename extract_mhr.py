"""One-off Modal job: extract the MHR character into a plain-numpy bundle.

Run:
  modal run extract_mhr.py::extract

Loads Meta's MHR (Momentum Human Rig, Apache-2.0) via pymomentum, and writes
``/models/sapiens2/mhr/mhr_lod{LOD}.npz`` containing everything the mocap
exporter needs — rest vertices, faces, skin weights, joint hierarchy and rest
transforms — so the inference image never needs pymomentum/fbx tooling.

Also prints the joint list; the retarget name map in mocap_retarget.py is
derived from it.
"""

from __future__ import annotations

import modal

LOD = 3

volume = modal.Volume.from_name("models", create_if_missing=True)
mhr_extractor = modal.App("mhr_extractor")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy", "trimesh>=4.8.3,<5", "pymomentum-cpu>=0.1.90")
    .pip_install("git+https://github.com/facebookresearch/MHR.git")
)


@mhr_extractor.function(image=image, volumes={"/models": volume}, timeout=1800)
def _extract() -> None:
    import subprocess
    from pathlib import Path

    import numpy as np
    import pymomentum.geometry as pym_geometry
    from mhr.io import get_mhr_fbx_path, get_mhr_model_path

    dest = Path("/tmp/mhr")
    assets = dest / "assets"
    if not Path(get_mhr_fbx_path(assets, LOD)).exists():
        subprocess.run(["mhr-download-assets", "--dest", str(dest)], check=True)
    print("assets:", sorted(p.name for p in assets.iterdir()))

    character = pym_geometry.Character.load_fbx(
        get_mhr_fbx_path(assets, LOD), get_mhr_model_path(assets)
    )

    skel = character.skeleton
    names = list(skel.joint_names)
    parents = np.asarray(skel.joint_parents, dtype=np.int32)
    # Rest skeleton state: local offsets + prerotations per joint.
    offsets = np.asarray(skel.offsets, dtype=np.float32)  # (J, 3)
    prerot = np.asarray(skel.pre_rotations, dtype=np.float32)  # (J, 4) quat

    # Zero-pose (bind) global skeleton state via momentum's own FK — the
    # retargeter builds on these instead of re-implementing momentum math.
    import traceback

    import pymomentum.torch.character as torch_character
    import torch

    n_joints = len(names)
    try:
        char_t = torch_character.Character(character)
        zero_joint_params = torch.zeros(1, n_joints * 7)
        skel_state = char_t.joint_parameters_to_skeleton_state(zero_joint_params)
        skel_state = skel_state.reshape(n_joints, 8).detach().cpu().numpy()
    except Exception:
        traceback.print_exc()
        print("torch_character API:", [m for m in dir(torch_character) if not m.startswith("_")])
        try:
            print("Character methods:", [m for m in dir(torch_character.Character) if "skel" in m or "joint" in m or "param" in m])
        except Exception:
            pass
        print("pym_geometry API:", [m for m in dir(pym_geometry) if "skel" in m.lower() or "state" in m.lower() or "model_param" in m.lower()])
        raise
    rest_pos = skel_state[:, 0:3].astype(np.float32)
    rest_rot = skel_state[:, 3:7].astype(np.float32)  # quat xyzw

    mesh = character.mesh
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    sw = character.skin_weights
    weights = np.asarray(sw.weight, dtype=np.float32)  # (V, K)
    indices = np.asarray(sw.index, dtype=np.int32)  # (V, K)

    out_dir = Path("/models/sapiens2/mhr")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"mhr_lod{LOD}.npz",
        joint_names=np.asarray(names),
        parents=parents,
        offsets=offsets,
        pre_rotations=prerot,
        rest_positions=rest_pos,
        rest_rotations=rest_rot,
        vertices=verts,
        faces=faces,
        skin_weights=weights,
        skin_indices=indices,
    )
    volume.commit()

    print(f"LOD{LOD}: {len(verts)} verts, {len(faces)} faces, {len(names)} joints")
    for i, n in enumerate(names):
        print(f"{i:3d} parent={parents[i]:3d} {n}")


@mhr_extractor.local_entrypoint()
def extract() -> None:
    _extract.remote()
