# -*- coding: utf-8 -*-
"""SAM ViT-B ONNX 推理封装。

本模块只依赖 onnxruntime 和 OpenCV，不导入 PyTorch。输入图片的坐标系
始终是“EXIF 摆正后的原图像素”，SAM 内部缩放和 padding 都在这里完成。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SAM_INPUT_SIZE = 1024
SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], np.float32)
PREPROCESS_VERSION = "sam-vit-b-runtime-v1"


class SamModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class SamGeometry:
    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]

    @classmethod
    def create(cls, original_hw):
        h, w = int(original_hw[0]), int(original_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError("图片尺寸必须为正数")
        scale = SAM_INPUT_SIZE / float(max(h, w))
        rh = int(np.floor(h * scale + 0.5))
        rw = int(np.floor(w * scale + 0.5))
        return cls((h, w), (max(1, rh), max(1, rw)))

    @property
    def scale_x(self):
        return self.resized_hw[1] / float(self.original_hw[1])

    @property
    def scale_y(self):
        return self.resized_hw[0] / float(self.original_hw[0])

    def points_to_input(self, points_xy):
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        out = pts.copy()
        out[:, 0] = np.clip(out[:, 0], 0, self.original_hw[1] - 1) * self.scale_x
        out[:, 1] = np.clip(out[:, 1], 0, self.original_hw[0] - 1) * self.scale_y
        return out

    def box_to_input(self, box_xyxy):
        x0, y0, x1, y1 = normalize_box(box_xyxy)
        return self.points_to_input(np.array([[x0, y0], [x1, y1]], np.float32))

    def original_to_preview(self, points_xy, preview_hw):
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).copy()
        pts[:, 0] *= preview_hw[1] / float(self.original_hw[1])
        pts[:, 1] *= preview_hw[0] / float(self.original_hw[0])
        return pts


@dataclass(frozen=True)
class SamCacheKey:
    path: str
    size: int
    mtime_ns: int
    original_hw: tuple[int, int]
    manifest_sha256: str
    preprocess_version: str = PREPROCESS_VERSION


@dataclass
class SamImage:
    embedding: np.ndarray
    geometry: SamGeometry
    cache_key: SamCacheKey


@dataclass
class SamPrompt:
    box_xyxy: tuple[float, float, float, float] | None
    points_xy: np.ndarray
    point_labels: np.ndarray
    mask_input: np.ndarray | None = None

    def __post_init__(self):
        self.points_xy = np.asarray(self.points_xy, dtype=np.float32).reshape(-1, 2)
        self.point_labels = np.asarray(self.point_labels, dtype=np.int64).reshape(-1)
        if len(self.points_xy) != len(self.point_labels):
            raise ValueError("点坐标和标签数量不一致")
        if not np.all(np.isin(self.point_labels, (0, 1))):
            raise ValueError("交互点标签只能是 0（排除）或 1（加入）")
        if self.box_xyxy is not None:
            self.box_xyxy = normalize_box(self.box_xyxy)
        if self.mask_input is not None:
            m = np.asarray(self.mask_input, np.float32)
            if m.shape == (256, 256):
                m = m[None, None]
            elif m.shape == (1, 256, 256):
                m = m[None]
            if m.shape != (1, 1, 256, 256):
                raise ValueError("mask_input 形状应为 1x1x256x256")
            self.mask_input = m

    @property
    def has_prompt(self):
        return self.box_xyxy is not None or len(self.points_xy) > 0

    @property
    def prompt_point_count(self):
        return (0 if self.box_xyxy is None else 2) + len(self.points_xy)


@dataclass
class SamResult:
    masks: np.ndarray
    scores: np.ndarray
    low_res_logits: np.ndarray
    geometry: SamGeometry
    selected: int = 0
    constraint_ok: bool = True

    @property
    def count(self):
        return int(self.masks.shape[0])

    def mask_logits(self, index=None):
        i = self.selected if index is None else int(index)
        if i < 0 or i >= self.count:
            raise IndexError("候选蒙版序号越界")
        return self.masks[i]

    def score(self, index=None):
        i = self.selected if index is None else int(index)
        return float(self.scores[i])

    def next_mask_input(self, index=None):
        i = self.selected if index is None else int(index)
        return self.low_res_logits[i][None, None].astype(np.float32)


def normalize_box(box_xyxy):
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return x0, y0, x1, y1


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sample_manifest_hash(models_dir):
    path = Path(models_dir) / "sam_vit_b_manifest.json"
    if not path.is_file():
        raise SamModelError("缺少 SAM manifest：%s" % path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        enc = data["encoder"]["sha256"]
        dec = data["decoder"]["sha256"]
        return hashlib.sha256((enc + ":" + dec).encode("ascii")).hexdigest()
    except Exception as exc:
        raise SamModelError("SAM manifest 无效：%s" % exc) from exc


def make_cache_key(path, original_hw, manifest_sha256):
    st = os.stat(path)
    return SamCacheKey(
        path=os.path.abspath(path),
        size=int(st.st_size),
        mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        original_hw=(int(original_hw[0]), int(original_hw[1])),
        manifest_sha256=str(manifest_sha256),
    )


class SamOnnxEngine:
    """一个进程只创建一个实例；encoder 会话和 decoder 会话都只由所属线程使用。"""

    def __init__(self, encoder_path, decoder_path, intra_op_threads=6):
        import onnxruntime as ort

        encoder_path = str(encoder_path)
        decoder_path = str(decoder_path)
        for path in (encoder_path, decoder_path):
            if not os.path.isfile(path):
                raise SamModelError("缺少 SAM 模型文件：%s" % path)

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(intra_op_threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(
            encoder_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(
            decoder_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.contract = self.validate_contract()

    def validate_contract(self):
        enc_inputs = self.encoder.get_inputs()
        enc_outputs = self.encoder.get_outputs()
        dec_inputs = self.decoder.get_inputs()
        dec_outputs = self.decoder.get_outputs()
        names = [x.name for x in dec_inputs]
        required = ["image_embeddings", "point_coords", "point_labels",
                    "mask_input", "has_mask_input", "orig_im_size"]
        if enc_inputs[0].name != "image":
            raise SamModelError("encoder 输入名应为 image，实际为 %s" % enc_inputs[0].name)
        if enc_outputs[0].name != "image_embeddings":
            raise SamModelError("encoder 输出名应为 image_embeddings")
        if names != required:
            raise SamModelError("decoder 输入不符合 SAM 契约：%s" % names)
        out_names = [x.name for x in dec_outputs]
        if not {"masks", "iou_predictions", "low_res_masks"}.issubset(set(out_names)):
            raise SamModelError("decoder 输出不符合 SAM 契约：%s" % out_names)
        return {
            "encoder_inputs": [[x.name, x.type, x.shape] for x in enc_inputs],
            "encoder_outputs": [[x.name, x.type, x.shape] for x in enc_outputs],
            "decoder_inputs": [[x.name, x.type, x.shape] for x in dec_inputs],
            "decoder_outputs": [[x.name, x.type, x.shape] for x in dec_outputs],
        }

    def preprocess(self, rgb):
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("需要 HxWx3 RGB 图像")
        geometry = SamGeometry.create(rgb.shape[:2])
        rh, rw = geometry.resized_hw
        interp = cv2.INTER_AREA if (rh < rgb.shape[0] or rw < rgb.shape[1]) else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (rw, rh), interpolation=interp).astype(np.float32)
        resized = (resized - SAM_MEAN) / SAM_STD
        tensor = np.zeros((SAM_INPUT_SIZE, SAM_INPUT_SIZE, 3), np.float32)
        tensor[:rh, :rw] = resized
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), geometry

    def encode(self, rgb, cache_key):
        tensor, geometry = self.preprocess(rgb)
        embedding = self.encoder.run(
            None, {self.encoder.get_inputs()[0].name: tensor})[0]
        if embedding.ndim != 4 or embedding.shape[0] != 1:
            raise SamModelError("encoder 输出形状异常：%s" % (embedding.shape,))
        return SamImage(np.asarray(embedding, np.float32), geometry, cache_key)

    def decode(self, image: SamImage, prompt: SamPrompt):
        if not prompt.has_prompt:
            raise ValueError("尚未指定框或点")
        geometry = image.geometry
        coords = []
        labels = []
        if prompt.box_xyxy is not None:
            box_pts = geometry.box_to_input(prompt.box_xyxy)
            coords.extend(box_pts.tolist())
            labels.extend([2, 3])
        if len(prompt.points_xy):
            pts = geometry.points_to_input(prompt.points_xy)
            coords.extend(pts.tolist())
            labels.extend(prompt.point_labels.tolist())

        point_coords = np.asarray(coords, dtype=np.float32).reshape(1, -1, 2)
        point_labels = np.asarray(labels, dtype=np.float32).reshape(1, -1)
        if prompt.mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), np.float32)
            has_mask = np.zeros((1,), np.float32)
        else:
            mask_input = np.asarray(prompt.mask_input, np.float32)
            has_mask = np.ones((1,), np.float32)
        orig_size = np.array(geometry.resized_hw, dtype=np.float32)

        feed = {
            "image_embeddings": image.embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
            "orig_im_size": orig_size,
        }
        try:
            outputs = self.decoder.run(
                ["masks", "iou_predictions", "low_res_masks"], feed)
        except Exception as exc:
            raise SamModelError("SAM decoder 推理失败：%s" % exc) from exc
        masks = np.asarray(outputs[0], np.float32)
        scores = np.asarray(outputs[1], np.float32).reshape(-1)
        low = np.asarray(outputs[2], np.float32)
        if masks.ndim != 4 or masks.shape[0] != 1 or masks.shape[1] < 1:
            raise SamModelError("decoder masks 形状异常：%s" % (masks.shape,))
        if low.ndim != 4 or low.shape[0] != 1:
            raise SamModelError("decoder low_res_masks 形状异常：%s" % (low.shape,))
        if not np.all(np.isfinite(masks)) or not np.all(np.isfinite(scores)):
            raise SamModelError("SAM 输出包含非有限数值")

        result = SamResult(masks[0], scores, low[0], geometry)
        result.selected, result.constraint_ok = select_candidate(result, prompt, geometry)
        return result


def _point_constraint_quality(mask_logits, points_xy, labels, geometry):
    if len(points_xy) == 0:
        return 1.0
    pts = geometry.points_to_input(points_xy)
    h, w = mask_logits.shape
    x = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)
    fg = mask_logits > 0.0
    hits = np.where(labels > 0, fg[y, x], ~fg[y, x])
    return float(hits.mean())


def select_candidate(result: SamResult, prompt: SamPrompt, geometry: SamGeometry):
    """按 SAM 的单/多候选规则选默认结果，并检查正负点约束。

    单个框或单点时优先 predicted IoU 最高的候选；框加修正点后，SAM 的
    第 0 个 mask token 是 single-mask 输出，优先使用它，但仍验证正负点
    是否真的落在对应区域，避免把明显违背用户修正的候选当成最终结果。
    """
    n = result.count
    if n <= 0:
        raise SamModelError("SAM 没有返回候选蒙版")
    num_points = prompt.prompt_point_count
    scores = result.scores.astype(np.float32)
    if num_points >= 3:
        # 框角点加至少一个修正点后，第 0 个 token 就是 SAM 的 single-mask 输出。
        preferred = 0
    elif n >= 4:
        # ONNX decoder 直接调用 predict_masks，返回 4 个 token：
        # 第 0 个是 single-mask，后 3 个是 multimask。初始提示从后 3 个里选。
        preferred = int(np.argmax(scores[1:4])) + 1
    else:
        preferred = int(np.argmax(scores))

    qualities = np.array([
        _point_constraint_quality(result.masks[i], prompt.points_xy,
                                  prompt.point_labels, geometry)
        for i in range(n)
    ], np.float32)
    preferred_ok = qualities[preferred] >= 1.0 - 1e-6
    if preferred_ok:
        return preferred, True

    valid = np.flatnonzero(qualities >= 1.0 - 1e-6)
    if len(valid):
        best = int(valid[np.argmax(scores[valid])])
        return best, True
    best = int(np.lexsort((scores, qualities))[-1])
    return best, False
