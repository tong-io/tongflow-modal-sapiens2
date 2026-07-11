"""Sapiens2 model hub + per-task inference, shared by all plugin slots.

Loads the official facebookresearch/sapiens2 repo models via its
``init_model`` API. Five 1B task checkpoints (~6 GB fp32 each) do not all fit
on one L40S together with activations, so models are loaded lazily and evicted
to CPU beyond an LRU budget of 3 GPU residents.

All functions take/return numpy; encoding to PNG/GLB happens in deploy.py.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = os.environ.get("SAPIENS2_REPO", "/app/sapiens2")
WEIGHTS_DIR = os.environ.get("SAPIENS2_WEIGHTS", "/models/sapiens2")
DETECTOR_DIR = os.environ.get(
    "SAPIENS2_DETECTOR", "/models/sapiens2/detector/detr-resnet-101-dc5"
)
MODEL_SIZE = os.environ.get("SAPIENS2_MODEL_SIZE", "1b")
# bf16 halves weight memory + bandwidth (~2x faster forwards) and lets all
# five 1B task models stay GPU-resident on one L40S.
DTYPE = torch.float32 if os.environ.get("SAPIENS2_DTYPE") == "fp32" else torch.bfloat16
GPU_LRU = int(os.environ.get("SAPIENS2_GPU_LRU", "5" if DTYPE is torch.bfloat16 else "3"))
SEG_PALETTE = os.environ.get("SAPIENS2_SEG_PALETTE", "dome29")

_CONFIGS = {
    "pose": f"{REPO_DIR}/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/"
    f"sapiens2_{MODEL_SIZE}_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "seg": f"{REPO_DIR}/sapiens/dense/configs/seg/shutterstock_goliath/"
    f"sapiens2_{MODEL_SIZE}_seg_shutterstock_goliath-1024x768.py",
    "normal": f"{REPO_DIR}/sapiens/dense/configs/normal/metasim_render_people/"
    f"sapiens2_{MODEL_SIZE}_normal_metasim_render_people-1024x768.py",
    "pointmap": f"{REPO_DIR}/sapiens/dense/configs/pointmap/render_people/"
    f"sapiens2_{MODEL_SIZE}_pointmap_render_people-1024x768.py",
    "matting": f"{REPO_DIR}/sapiens/dense/configs/matting/gss_p3m_metasim/"
    f"sapiens2_{MODEL_SIZE}_matting_gss_p3m_metasim-1024x768.py",
}

# pose_render_utils is a script-local module of the official repo, not a package
sys.path.append(f"{REPO_DIR}/sapiens/pose/tools/vis")


class ModelHub:
    """Lazy per-task model loading with a GPU-resident LRU (evict to CPU)."""

    def __init__(self) -> None:
        self._models: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._detector = None
        self._det_proc = None
        self._pose_meta = None
        self._pose_codec = None

    def _checkpoint(self, task: str) -> str:
        return f"{WEIGHTS_DIR}/{task}/sapiens2_{MODEL_SIZE}_{task}.safetensors"

    def get(self, task: str) -> torch.nn.Module:
        if task in self._models:
            self._models.move_to_end(task)
            model = self._models[task]
            if next(model.parameters()).device.type != "cuda":
                model.to("cuda")
            self._evict()
            return model

        if task == "pose":
            from sapiens.pose.models import init_model
        else:
            from sapiens.dense.models import init_model

        model = init_model(_CONFIGS[task], self._checkpoint(task), device="cuda")
        model.to(DTYPE)
        if task == "pose":
            from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo

            self._pose_meta = parse_pose_metainfo(
                dict(
                    from_file=f"{REPO_DIR}/sapiens/pose/configs/_base_/keypoints308.py"
                )
            )
            codec_cfg = dict(model.cfg.codec)
            codec_type = codec_cfg.pop("type")
            assert codec_type == "UDPHeatmap", "only UDPHeatmap supported"
            self._pose_codec = UDPHeatmap(**codec_cfg)
        self._models[task] = model
        self._evict()
        return model

    def _evict(self) -> None:
        on_gpu = [
            t
            for t, m in self._models.items()
            if next(m.parameters()).device.type == "cuda"
        ]
        while len(on_gpu) > GPU_LRU:
            victim = on_gpu.pop(0)
            self._models[victim].to("cpu")
            torch.cuda.empty_cache()

    @property
    def pose_meta(self) -> dict:
        if self._pose_meta is None:
            self.get("pose")
        return self._pose_meta

    @property
    def pose_codec(self):
        if self._pose_codec is None:
            self.get("pose")
        return self._pose_codec

    def detector(self):
        if self._detector is None:
            from transformers import DetrForObjectDetection, DetrImageProcessor

            self._det_proc = DetrImageProcessor.from_pretrained(DETECTOR_DIR)
            self._detector = (
                DetrForObjectDetection.from_pretrained(DETECTOR_DIR).eval().to("cuda")
            )
        return self._det_proc, self._detector


# ---------------------------------------------------------------- detection


def detect_persons(
    hub: ModelHub, image_bgr: np.ndarray, threshold: float = 0.3, nms_thr: float = 0.3
) -> np.ndarray:
    """DETR person boxes (N, 5): x1, y1, x2, y2, score — desc by score."""
    from PIL import Image
    from sapiens.pose.evaluators import nms

    proc, model = hub.detector()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = torch.tensor([rgb.shape[:2]], device="cuda")
    res = proc.post_process_object_detection(
        outputs, target_sizes=sizes, threshold=threshold
    )[0]
    keep = res["labels"] == 1  # COCO person
    boxes = res["boxes"][keep].cpu().numpy()
    scores = res["scores"][keep].cpu().numpy().reshape(-1, 1)
    if len(boxes) == 0:
        h, w = rgb.shape[:2]
        return np.array([[0, 0, w - 1, h - 1, 0.0]], dtype=np.float32)
    dets = np.concatenate([boxes, scores], axis=1).astype(np.float32)
    dets = dets[nms(dets, nms_thr)]
    return dets[np.argsort(-dets[:, 4])]


# ---------------------------------------------------------------- pose


def pose_infer(
    hub: ModelHub, image_bgr: np.ndarray, bboxes: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """308-keypoint top-down pose for the given person boxes (N, 4+).

    Returns (keypoints [(K,2) image coords], scores [(K,)]) per box.
    """
    model = hub.get("pose")
    inputs_list, samples_list = [], []
    for bbox in bboxes:
        data = model.pipeline(
            dict(
                img=image_bgr,
                bbox=np.asarray(bbox[:4], dtype=np.float32)[None],
                bbox_score=np.ones(1, dtype=np.float32),
            )
        )
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0).to(DTYPE)
    with torch.no_grad():
        pred = model(inputs).float()
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            flipped = model(inputs.flip(-1)).float().flip(-1)
            flip_indices = hub.pose_meta["flip_indices"]
            pred = (pred + flipped[:, flip_indices]) / 2.0
    pred = pred.cpu().numpy()

    keypoints, scores = [], []
    for i, samples in enumerate(samples_list):
        kpts, kpt_scores = hub.pose_codec.decode(pred[i])
        meta = samples["meta"]
        kpts = (
            kpts / meta["input_size"] * meta["bbox_scale"]
            + meta["bbox_center"]
            - 0.5 * meta["bbox_scale"]
        )
        keypoints.append(kpts[0])
        scores.append(kpt_scores[0])
    return keypoints, scores


def pose_visualize(
    hub: ModelHub,
    image_bgr: np.ndarray,
    keypoints: list[np.ndarray],
    scores: list[np.ndarray],
    kpt_thr: float = 0.3,
) -> np.ndarray:
    from pose_render_utils import visualize_keypoints

    meta = hub.pose_meta
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis = visualize_keypoints(
        image=rgb,
        keypoints=keypoints,
        keypoints_visible=[np.ones_like(s) > 0 for s in scores],
        keypoint_scores=scores,
        radius=max(2, image_bgr.shape[1] // 400),
        thickness=max(2, image_bgr.shape[1] // 500),
        kpt_thr=kpt_thr,
        skeleton=meta["skeleton_links"],
        kpt_color=meta["keypoint_colors"],
        link_color=meta["skeleton_link_colors"],
    )
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- dense tasks


def _dense_forward(model, image_bgr: np.ndarray):
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    return data["inputs"], data["data_samples"]


def _unpad_resize(t: torch.Tensor, samples, hw: tuple[int, int]) -> torch.Tensor:
    """Crop the test-pipeline padding then resize to the original image size."""
    meta = samples["meta"] if isinstance(samples, dict) else samples
    pad = meta.get("padding_size") if hasattr(meta, "get") else None
    if pad is not None:
        pl, pr, pt, pb = pad
        t = t[:, :, int(pt) : t.shape[2] - int(pb), int(pl) : t.shape[3] - int(pr)]
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)


def seg_labels(hub: ModelHub, image_bgr: np.ndarray) -> np.ndarray:
    """(H, W) int labels, 0 = background."""
    model = hub.get("seg")
    inputs, _ = _dense_forward(model, image_bgr)
    with torch.no_grad():
        logits = model(inputs.to(DTYPE)).float()
    logits = F.interpolate(logits, size=image_bgr.shape[:2], mode="bilinear")
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def seg_visualize(image_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from sapiens.dense.visualizers import SegVisualizer

    vis = SegVisualizer(class_palette_type=SEG_PALETTE, with_labels=False)
    return vis._visualize_segmentation(image_bgr, labels)


def normal_map(hub: ModelHub, image_bgr: np.ndarray) -> np.ndarray:
    """(H, W, 3) unit normals in [-1, 1]."""
    model = hub.get("normal")
    inputs, samples = _dense_forward(model, image_bgr)
    with torch.no_grad():
        normal = model(inputs.to(DTYPE)).float()
        normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)
    normal = _unpad_resize(normal, samples, image_bgr.shape[:2])
    return normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def normal_visualize(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = normal.copy()
    n[mask == 0] = -1
    vis = ((n + 1) / 2 * 255).astype(np.uint8)
    return vis[:, :, ::-1]  # RGB -> BGR


def pointmap_metric(
    hub: ModelHub, image_bgr: np.ndarray
) -> np.ndarray:
    """(H, W, 3) metric camera-space points (x right, y down, z forward)."""
    model = hub.get("pointmap")
    inputs, samples = _dense_forward(model, image_bgr)
    with torch.no_grad():
        pointmap, scale = model(inputs.to(DTYPE))
        pointmap = pointmap.float() / scale.float()
    pointmap = _unpad_resize(pointmap, samples, image_bgr.shape[:2])
    return pointmap.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def matting_infer(
    hub: ModelHub, image_bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(fgr_rgb premultiplied [0,1] (H,W,3), alpha [0,1] (H,W))."""
    model = hub.get("matting")
    inputs, _ = _dense_forward(model, image_bgr)
    with torch.no_grad():
        outputs = model(inputs.to(DTYPE)).float()
    outputs = F.interpolate(
        outputs, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
    )
    outputs = outputs.squeeze(0).float().cpu().numpy()
    fgr = outputs[0:3].clip(0, 1).transpose(1, 2, 0)
    alpha = outputs[3].clip(0, 1)
    return fgr, alpha


# ------------------------------------------------------------- batched (video)


def detect_persons_batch(
    hub: ModelHub,
    images_bgr: list[np.ndarray],
    threshold: float = 0.3,
    nms_thr: float = 0.3,
) -> list[np.ndarray]:
    """DETR person boxes for a batch of frames; one (N, 5) array per frame."""
    from PIL import Image
    from sapiens.pose.evaluators import nms

    proc, model = hub.detector()
    pils = [
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_bgr
    ]
    inputs = proc(images=pils, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = torch.tensor([img.shape[:2] for img in images_bgr], device="cuda")
    results = proc.post_process_object_detection(
        outputs, target_sizes=sizes, threshold=threshold
    )
    out: list[np.ndarray] = []
    for img, res in zip(images_bgr, results):
        keep = res["labels"] == 1
        boxes = res["boxes"][keep].cpu().numpy()
        scores = res["scores"][keep].cpu().numpy().reshape(-1, 1)
        if len(boxes) == 0:
            h, w = img.shape[:2]
            out.append(np.array([[0, 0, w - 1, h - 1, 0.0]], dtype=np.float32))
            continue
        dets = np.concatenate([boxes, scores], axis=1).astype(np.float32)
        dets = dets[nms(dets, nms_thr)]
        out.append(dets[np.argsort(-dets[:, 4])])
    return out


def pose_infer_frames(
    hub: ModelHub, frames: list[np.ndarray], boxes: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Batched top-down pose, one box per frame -> ((F, K, 2), (F, K))."""
    model = hub.get("pose")
    inputs_list, samples_list = [], []
    for frame, bbox in zip(frames, boxes):
        data = model.pipeline(
            dict(
                img=frame,
                bbox=np.asarray(bbox[:4], dtype=np.float32)[None],
                bbox_score=np.ones(1, dtype=np.float32),
            )
        )
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0).to(DTYPE)
    with torch.no_grad():
        pred = model(inputs).float()
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            flipped = model(inputs.flip(-1)).float().flip(-1)
            pred = (pred + flipped[:, hub.pose_meta["flip_indices"]]) / 2.0
    pred = pred.cpu().numpy()

    kpts_out, scores_out = [], []
    for i, samples in enumerate(samples_list):
        kpts, kpt_scores = hub.pose_codec.decode(pred[i])
        meta = samples["meta"]
        kpts = (
            kpts / meta["input_size"] * meta["bbox_scale"]
            + meta["bbox_center"]
            - 0.5 * meta["bbox_scale"]
        )
        kpts_out.append(kpts[0])
        scores_out.append(kpt_scores[0])
    return np.stack(kpts_out), np.stack(scores_out)


def pointmap_metric_frames(
    hub: ModelHub, frames: list[np.ndarray]
) -> np.ndarray:
    """Batched metric pointmaps -> (F, H, W, 3). Frames share one size."""
    model = hub.get("pointmap")
    inputs_list, samples = [], None
    for frame in frames:
        data = model.pipeline(dict(img=frame))
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        samples = data["data_samples"]

    inputs = torch.cat(inputs_list, dim=0).to(DTYPE)
    with torch.no_grad():
        pointmap, scale = model(inputs)
        pointmap = pointmap.float() / scale.float().reshape(-1, 1, 1, 1)
    pointmap = _unpad_resize(pointmap, samples, frames[0].shape[:2])
    return pointmap.cpu().numpy().transpose(0, 2, 3, 1)


def matting_rgba(fgr_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Straight-alpha RGBA uint8 from premultiplied foreground."""
    a = alpha[..., None]
    straight = np.where(a > 1e-4, fgr_rgb / np.maximum(a, 1e-4), 0.0).clip(0, 1)
    rgba = np.concatenate([straight, a], axis=-1)
    return (rgba * 255).astype(np.uint8)
