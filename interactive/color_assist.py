# -*- coding: utf-8 -*-
"""颜色先验：只生成受限的 SAM 建议点，不直接改写蒙版。"""
from __future__ import annotations

import cv2
import numpy as np


def color_similarity(rgb, samples, tolerance=0.22):
    """Return a [0, 1] similarity map using Lab distance to sampled colors."""
    image = np.asarray(rgb, dtype=np.uint8)
    sample_array = np.asarray(samples, dtype=np.uint8).reshape(-1, 1, 3)
    if sample_array.size == 0:
        raise ValueError("至少需要一个颜色样本")
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    sample_lab = cv2.cvtColor(sample_array, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    distance = np.linalg.norm(
        image_lab[:, :, None, :] - sample_lab[None, None, :, :], axis=3)
    nearest = np.min(distance, axis=2)
    scale = max(1.0, float(tolerance) * 100.0)
    return np.exp(-0.5 * (nearest / scale) ** 2).astype(np.float32)


def suggest_points(rgb, samples, box_xyxy, max_points=5, tolerance=0.22,
                   min_score=0.62, min_spacing=32):
    """Return at most max_points original-image coordinates for SAM positive prompts."""
    image = np.asarray(rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if x1 <= x0 or y1 <= y0:
        return []
    score = color_similarity(image, samples, tolerance=tolerance)
    region = score[y0:y1, x0:x1]
    region = np.where(region >= float(min_score), region, 0.0)
    points = []
    work = region.copy()
    radius = max(2, int(min_spacing))
    for _ in range(max(0, int(max_points))):
        iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
        value = float(work[iy, ix])
        if value < float(min_score):
            break
        points.append((float(x0 + ix), float(y0 + iy), value))
        yy0 = max(0, iy - radius)
        yy1 = min(work.shape[0], iy + radius + 1)
        xx0 = max(0, ix - radius)
        xx1 = min(work.shape[1], ix + radius + 1)
        work[yy0:yy1, xx0:xx1] = 0.0
    return points
