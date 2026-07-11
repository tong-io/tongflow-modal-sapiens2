"""Modal download entry for Sapiens2.

Run:
  modal run download.py::download

Fetches the native-format 1B task checkpoints (each HF repo also carries a
duplicate transformers-format ``model.safetensors`` which the plugin does not
use), the DETR person detector, and — when the gated-access request has been
approved — the SAM 3D Body mocap engine (facebook/sam-3d-body-dinov3, manual
review by Meta) plus the MoGe-2 FOV estimator, into the shared ``models``
volume.

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import os

import modal

MODEL_SIZE = "1b"
TASKS = ("pose", "seg", "normal", "pointmap", "matting")
DETECTOR_REPO = "facebook/detr-resnet-101-dc5"
# Mocap body engine (gated: requires an approved access request on HF).
SAM3D_REPO = os.environ.get("SAM_3D_BODY_MODEL", "facebook/sam-3d-body-dinov3")
MOGE_REPO = "Ruicheng/moge-2-vitl-normal"

volume = modal.Volume.from_name("models", create_if_missing=True)
secrets = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.6.0")
    .env({"HF_HOME": "/models/hf"}),
    volumes={"/models": volume},
    secrets=[secrets],
    timeout=7200,
)
def _download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    for task in TASKS:
        filename = f"sapiens2_{MODEL_SIZE}_{task}.safetensors"
        hf_hub_download(
            repo_id=f"facebook/sapiens2-{task}-{MODEL_SIZE}",
            filename=filename,
            local_dir=f"/models/sapiens2/{task}",
        )
        print(f"Cached sapiens2 {task}")

    snapshot_download(
        repo_id=DETECTOR_REPO,
        local_dir="/models/sapiens2/detector/detr-resnet-101-dc5",
    )
    print(f"Cached {DETECTOR_REPO}")

    # Gated mocap weights: tolerate a pending/denied access request so the
    # five image slots never get blocked by the mocap engine's gate. The
    # hybrid mocap engine falls back to the geometric pipeline until these
    # are available.
    for repo in (SAM3D_REPO, MOGE_REPO):
        try:
            snapshot_download(repo_id=repo)
            print(f"Cached {repo}")
        except Exception as e:  # 403 while Meta reviews the access request
            print(
                f"WARNING: could not cache {repo}: {e}\n"
                "If this is a 403, the Hugging Face gated-access request is "
                "still awaiting approval; video mocap will use the fallback "
                "engine until then."
            )

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
