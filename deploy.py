"""Modal deploy entry for Sapiens2 (Meta, human-centric vision foundation model).

One app, six slots, all backed by the 1B task checkpoints of
facebookresearch/sapiens2 (ICLR 2026):

- ``image-pose``       308-keypoint skeleton overlay (body + hands + face)
- ``image-body-seg``   29-class body-part segmentation overlay
- ``image-normal``     per-pixel surface normal map (foreground masked)
- ``image-matting``    human foreground as straight-alpha RGBA PNG
- ``image-gen-model``  pointmap -> colored 3D point cloud GLB
- ``video-gen-model``  monocular video motion capture -> animated MHR GLB
                       (pure Sapiens2: 308-keypoint pose + pointmap 3D lift,
                       geometric solve; single subject, largest person tracked)

Five ~6 GB fp32 checkpoints cannot all sit on one L40S, so models load
lazily with a 3-model GPU LRU (evicted to CPU). Pose additionally uses a
DETR person detector.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_body_seg import ImageBodySegInput, ImageBodySegOutput
from tongflow.models.image_gen_model import ImageGenModelInput, ImageGenModelOutput
from tongflow.models.image_matting import ImageMattingInput, ImageMattingOutput
from tongflow.models.image_normal import ImageNormalInput, ImageNormalOutput
from tongflow.models.image_pose import ImagePoseInput, ImagePoseOutput
from tongflow.models.video_gen_model import VideoGenModelInput, VideoGenModelOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = [
    "image-pose",
    "image-normal",
    "image-matting",
]

REPO_URL = "https://github.com/facebookresearch/sapiens2.git"
# Pin the upstream revision so redeploys are reproducible (main moves).
REPO_REV = "7e5bae88456ac418ff0e58e74106c9fe192055d4"
REPO_DIR = "/app/sapiens2"


# Point-cloud density cap for the pointmap slot — NOT an ABI field.
POINTMAP_MAX_POINTS = 400_000

_HERE = Path(__file__).resolve().parent

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(_HERE.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout {REPO_REV}",
        f"pip install -e {REPO_DIR}",
    )
    .pip_install("tongflow==0.2.13", "fastapi[standard]")
    .env(
        {
            "HF_HOME": "/models/hf",
            "SAPIENS2_REPO": REPO_DIR,
            "SAPIENS2_WEIGHTS": "/models/sapiens2",
            "SAPIENS2_DETECTOR": "/models/sapiens2/detector/detr-resnet-101-dc5",
            "PYTHONPATH": "/opt/sapiens2_plugin",
        }
    )
    # Mounted at runtime (copy defaults to False) so every deploy ships the
    # latest vendored pipeline without baking a cacheable image layer.
    .add_local_file(str(_HERE / "sapiens2_runtime.py"), "/opt/sapiens2_plugin/sapiens2_runtime.py")
    .add_local_file(str(_HERE / "mocap_pipeline.py"), "/opt/sapiens2_plugin/mocap_pipeline.py")
    .add_local_file(str(_HERE / "mocap_retarget.py"), "/opt/sapiens2_plugin/mocap_retarget.py")
    .add_local_file(str(_HERE / "gltf_writer.py"), "/opt/sapiens2_plugin/gltf_writer.py")
)

with image.imports():
    import cv2
    import numpy as np

    import mocap_pipeline
    import sapiens2_runtime as rt


def _decode_bgr(data: bytes) -> "np.ndarray":
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("could not decode input image")
    return img


def _png(image_bgr_or_bgra: "np.ndarray") -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr_or_bgra)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


@deploy
@app.cls(
    image=image,
    # Resolved at deploy time; changing it requires a redeploy (touch this
    # file or `modal deploy deploy.py` manually).
    gpu=os.environ.get("SAPIENS2_GPU", "L40S"),
    memory=32768,
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=300,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        self.hub = rt.ModelHub()

    @modal.method()
    @node_slot(NodeSlots.IMAGE_POSE)
    def image_pose(self, input: ImagePoseInput) -> ImagePoseOutput:
        """308-keypoint whole-body pose overlay for every detected person."""
        try:
            img = _decode_bgr(prompt_media_to_bytes(input.image))
            boxes = rt.detect_persons(self.hub, img)
            keypoints, scores = rt.pose_infer(self.hub, img, boxes)
            vis = rt.pose_visualize(self.hub, img, keypoints, scores)
            data = _png(vis)
        except Exception as e:
            return ImagePoseOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImagePoseOutput(
            success=True, image=asset(data, mime="image/png", filename="pose.png")
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_BODY_SEG)
    def image_body_seg(self, input: ImageBodySegInput) -> ImageBodySegOutput:
        """29-class body-part segmentation, color-coded overlay."""
        try:
            img = _decode_bgr(prompt_media_to_bytes(input.image))
            labels = rt.seg_labels(self.hub, img)
            vis = rt.seg_visualize(img, labels)
            data = _png(vis)
        except Exception as e:
            return ImageBodySegOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageBodySegOutput(
            success=True, image=asset(data, mime="image/png", filename="body-seg.png")
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_NORMAL)
    def image_normal(self, input: ImageNormalInput) -> ImageNormalOutput:
        """Per-pixel surface normals, background masked via body-part seg."""
        try:
            img = _decode_bgr(prompt_media_to_bytes(input.image))
            normal = rt.normal_map(self.hub, img)
            mask = rt.seg_labels(self.hub, img) > 0
            vis = rt.normal_visualize(normal, mask)
            data = _png(vis)
        except Exception as e:
            return ImageNormalOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageNormalOutput(
            success=True, image=asset(data, mime="image/png", filename="normal.png")
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_MATTING)
    def image_matting(self, input: ImageMattingInput) -> ImageMattingOutput:
        """Human foreground extraction as a straight-alpha transparent PNG."""
        try:
            img = _decode_bgr(prompt_media_to_bytes(input.image))
            fgr, alpha = rt.matting_infer(self.hub, img)
            rgba = rt.matting_rgba(fgr, alpha)
            data = _png(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        except Exception as e:
            return ImageMattingOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageMattingOutput(
            success=True, image=asset(data, mime="image/png", filename="matting.png")
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_MODEL)
    def image_gen_model(self, input: ImageGenModelInput) -> ImageGenModelOutput:
        """Pointmap -> colored human point cloud GLB.

        text/width/height/seed are part of the image-gen-model contract but
        pointmap estimation is deterministic and image-only; they are ignored.
        """
        try:
            img = _decode_bgr(prompt_media_to_bytes(input.image))
            points = rt.pointmap_metric(self.hub, img)  # (H, W, 3) camera coords
            mask = rt.seg_labels(self.hub, img) > 0
            if not mask.any():
                raise RuntimeError("no person detected in the image")
            pts = points[mask]
            colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[mask]
            if len(pts) > POINTMAP_MAX_POINTS:
                pick = np.random.default_rng(0).choice(
                    len(pts), POINTMAP_MAX_POINTS, replace=False
                )
                pts, colors = pts[pick], colors[pick]
            # Camera coords are y-down / z-forward; glTF viewers expect y-up.
            pts = pts * np.array([1.0, -1.0, -1.0])
            pts -= pts.mean(axis=0)
            from gltf_writer import point_cloud_glb

            data = point_cloud_glb(pts, colors)
        except Exception as e:
            return ImageGenModelOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageGenModelOutput(
            success=True,
            model=asset(data, mime="model/gltf-binary", filename="pointmap.glb"),
        )

    @modal.method()
    @node_slot(NodeSlots.VIDEO_GEN_MODEL)
    def video_gen_model(self, input: VideoGenModelInput) -> VideoGenModelOutput:
        """Monocular video mocap -> skeletal-animation GLB (single subject)."""
        try:
            video = prompt_media_to_bytes(input.video)
            data = mocap_pipeline.capture(self.hub, video, progress=print)
        except Exception as e:
            return VideoGenModelOutput(success=False, error=f"{type(e).__name__}: {e}")
        return VideoGenModelOutput(
            success=True,
            model=asset(data, mime="model/gltf-binary", filename="mocap.glb"),
        )

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

