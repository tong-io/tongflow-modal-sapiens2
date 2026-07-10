"""Modal download entry for Sapiens2.

Run:
  modal run download.py::download

Fetches only the native-format 1B task checkpoints (each HF repo also carries
a duplicate transformers-format ``model.safetensors`` which the plugin does
not use) plus the DETR person detector, into the shared ``models`` volume.

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import modal

MODEL_SIZE = "1b"
TASKS = ("pose", "seg", "normal", "pointmap", "matting")
DETECTOR_REPO = "facebook/detr-resnet-101-dc5"

volume = modal.Volume.from_name("models", create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.6.0")
    .env({"HF_HOME": "/models/hf"}),
    volumes={"/models": volume},
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

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
