# -*- coding: utf-8 -*-
"""反事实效应分析：蒙版指标、差异图和实际链路重放计划。

本模块不依赖 Qt，也不持有正式提示、蒙版或工程包状态。重放计划只描述
“按历史如何重跑 decoder”，真正的推理由 worker 在后台执行。
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import cv2
import numpy as np

from .session import PromptPoint


@dataclass(frozen=True)
class ReplayStep:
    """一次 decoder 输入中的完整提示点集合。

    input_candidate 为 None 时，使用上一步反事实结果自动选择的候选；
    仅在历史中连续的步骤才保留原来的候选序号。
    """

    points: tuple[PromptPoint, ...]
    input_candidate: int | None = 0


@dataclass(frozen=True)
class EffectPlan:
    """从无 mask_input 节点开始的完整重放计划。"""

    kind: str
    label: str
    box_xyxy: tuple[float, float, float, float] | None
    steps: tuple[ReplayStep, ...]
    target_point_index: int | None = None

    @property
    def key(self):
        if self.target_point_index is None:
            return self.kind
        return "%s:%d" % (self.kind, self.target_point_index)


@dataclass(frozen=True)
class MaskMetrics:
    total_pixels: int
    before_pixels: int
    after_pixels: int
    added_pixels: int
    removed_pixels: int
    changed_pixels: int
    area_delta_pixels: int
    area_delta_ratio: float | None
    changed_image_ratio: float
    iou: float
    dice: float
    boundary_p95: float | None
    before_components: int
    after_components: int
    before_holes: int
    after_holes: int


@dataclass(frozen=True)
class PointLocalMetrics:
    center_ok: bool
    support: float
    boundary_proximity: float
    radius: int


@dataclass(frozen=True)
class EffectResult:
    plan: EffectPlan
    current_mask: np.ndarray
    counterfactual_mask: np.ndarray
    current_candidate: int
    counterfactual_candidate: int
    metrics: MaskMetrics
    decodes: int
    cache_hits: int
    seconds: float

    @property
    def candidate_changed(self):
        return int(self.current_candidate) != int(self.counterfactual_candidate)


def _without_point(points, point_index):
    if point_index < 0 or point_index >= len(points):
        return tuple(points)
    return tuple(points[:point_index]) + tuple(points[point_index + 1:])


def build_point_loo_plan(states, point_index):
    """构造“去掉当前第 i 个点”的实际链路重放计划。

    states 必须是从旧到新的当前撤销分支，最后一项为当前快照。最近一个
    mask_input 为空的快照作为起点；其后按原顺序重放，并跳过目标点。
    """
    states = tuple(states)
    if not states:
        raise ValueError("历史为空，无法构造重放计划")
    current = states[-1]
    if not current.points:
        raise ValueError("当前没有可分析的点")
    point_index = int(point_index)
    if point_index < 0 or point_index >= len(current.points):
        raise IndexError("点序号越界")

    baseline_index = None
    for index, snapshot in enumerate(states):
        if snapshot.mask_input is None:
            baseline_index = index
    if baseline_index is None:
        raise ValueError("找不到无 mask_input 的重放起点")
    baseline = states[baseline_index]
    if baseline.box_xyxy is None:
        raise ValueError("重放起点没有框")
    if tuple(current.points[:len(baseline.points)]) != tuple(baseline.points):
        raise ValueError("历史点序列不是当前点的前缀，无法忠实重放")

    first_points = _without_point(baseline.points, point_index)
    steps = [ReplayStep(first_points, 0)]
    last_included_index = baseline_index
    for index in range(baseline_index + 1, len(states)):
        previous = states[index - 1]
        snapshot = states[index]
        points = _without_point(snapshot.points, point_index)
        if points == steps[-1].points:
            continue
        if index == last_included_index + 1:
            input_candidate = int(previous.candidate)
        else:
            input_candidate = None
        steps.append(ReplayStep(points, input_candidate))
        last_included_index = index

    expected = _without_point(current.points, point_index)
    if steps[-1].points != expected:
        raise ValueError("重放计划未到达当前提示集合")
    label = "点 %d 的留一影响" % (point_index + 1)
    return EffectPlan(
        kind="point_loo",
        label=label,
        box_xyxy=tuple(float(v) for v in baseline.box_xyxy),
        steps=tuple(steps),
        target_point_index=point_index,
    )


def build_box_only_plan(states):
    """构造“保留当前框、去掉全部修正点”的单步计划。"""
    states = tuple(states)
    if not states:
        raise ValueError("历史为空，无法构造重放计划")
    current = states[-1]
    if current.box_xyxy is None:
        raise ValueError("当前没有框，无法做框-only 对比")
    return EffectPlan(
        kind="box_only",
        label="框-only 对比",
        box_xyxy=tuple(float(v) for v in current.box_xyxy),
        steps=(ReplayStep((), 0),),
    )


def point_display_name(points, point_index):
    if point_index < 0 or point_index >= len(points):
        return "点 ?"
    point = points[point_index]
    prefix = "正" if int(point.label) == 1 else "负"
    number = sum(
        1 for index, item in enumerate(points[:point_index + 1])
        if int(item.label) == int(point.label))
    return "%s%d" % (prefix, number)


def _boundary(mask):
    source = np.asarray(mask, np.uint8)
    if source.size == 0 or source.max() == 0:
        return None
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(source, cv2.MORPH_GRADIENT, kernel)


def _boundary_p95(reference_mask, current_mask):
    reference_edge = _boundary(reference_mask)
    current_edge = _boundary(current_mask)
    if reference_edge is None or current_edge is None:
        return None
    to_current = cv2.distanceTransform(
        1 - (current_edge > 0).astype(np.uint8), cv2.DIST_L2, 5)
    to_reference = cv2.distanceTransform(
        1 - (reference_edge > 0).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([
        to_current[reference_edge > 0],
        to_reference[current_edge > 0],
    ])
    if distances.size == 0:
        return None
    return float(np.percentile(distances, 95))


def _structure_counts(mask):
    source = np.asarray(mask, np.uint8)
    if source.size == 0 or source.max() == 0:
        return 0, 0
    count, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        source, connectivity=8)
    contours, hierarchy = cv2.findContours(
        source, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = int(sum(1 for item in hierarchy[0] if int(item[3]) >= 0))
    return max(0, int(count) - 1), holes


def compare_masks(reference_mask, current_mask):
    """比较反事实参照（before）和当前结果（after）。"""
    before = np.asarray(reference_mask, bool)
    after = np.asarray(current_mask, bool)
    if before.shape != after.shape:
        raise ValueError("两个蒙版尺寸不一致")
    total = int(before.size)
    before_pixels = int(before.sum())
    after_pixels = int(after.sum())
    intersection = int(np.logical_and(before, after).sum())
    union = int(np.logical_or(before, after).sum())
    added = int(np.logical_and(after, ~before).sum())
    removed = int(np.logical_and(before, ~after).sum())
    changed = added + removed
    iou = 1.0 if union == 0 else float(intersection) / float(union)
    denominator = before_pixels + after_pixels
    dice = 1.0 if denominator == 0 else 2.0 * intersection / float(denominator)
    area_delta = after_pixels - before_pixels
    area_ratio = None if before_pixels == 0 else area_delta / float(before_pixels)
    before_components, before_holes = _structure_counts(before)
    after_components, after_holes = _structure_counts(after)
    return MaskMetrics(
        total_pixels=total,
        before_pixels=before_pixels,
        after_pixels=after_pixels,
        added_pixels=added,
        removed_pixels=removed,
        changed_pixels=changed,
        area_delta_pixels=area_delta,
        area_delta_ratio=area_ratio,
        changed_image_ratio=(changed / float(total)) if total else 0.0,
        iou=iou,
        dice=dice,
        boundary_p95=_boundary_p95(before, after),
        before_components=before_components,
        after_components=after_components,
        before_holes=before_holes,
        after_holes=after_holes,
    )


def point_local_metrics(mask, point, original_hw, radius=None):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("蒙版不可用")
    h, w = mask.shape
    original_h, original_w = int(original_hw[0]), int(original_hw[1])
    px = min(w - 1, max(0, int(round(float(point.x) * w / max(1, original_w)))))
    py = min(h - 1, max(0, int(round(float(point.y) * h / max(1, original_h)))))
    if radius is None:
        radius = max(3, min(h, w) // 80)
    radius = max(1, int(radius))
    x0, x1 = max(0, px - radius), min(w, px + radius + 1)
    y0, y1 = max(0, py - radius), min(h, py + radius + 1)
    patch = mask[y0:y1, x0:x1]
    expected = int(point.label) == 1
    support = float(np.mean(patch == expected)) if patch.size else 0.0

    ring = []
    for yy in range(max(0, py - radius - 1), min(h, py + radius + 2)):
        for xx in range(max(0, px - radius - 1), min(w, px + radius + 2)):
            distance = max(abs(xx - px), abs(yy - py))
            if distance == radius + 1:
                ring.append(bool(mask[yy, xx]) == expected)
    proximity = 1.0 - (float(sum(ring)) / len(ring) if ring else 1.0)
    center_ok = bool(mask[py, px]) == expected
    return PointLocalMetrics(center_ok, support, proximity, radius)


def point_constraint_rate(mask, points, original_hw):
    mask = np.asarray(mask, bool)
    if not points:
        return 1.0
    if mask.ndim != 2 or mask.size == 0:
        return 0.0
    h, w = mask.shape
    original_h, original_w = int(original_hw[0]), int(original_hw[1])
    hits = []
    for point in points:
        px = min(w - 1, max(0, int(round(float(point.x) * w / max(1, original_w)))))
        py = min(h - 1, max(0, int(round(float(point.y) * h / max(1, original_h)))))
        expected = int(point.label) == 1
        hits.append(bool(mask[py, px]) == expected)
    return float(sum(hits)) / len(hits)


def mask_to_preview(mask, preview_hw, interpolation=cv2.INTER_NEAREST):
    target_h, target_w = int(preview_hw[0]), int(preview_hw[1])
    source = np.asarray(mask, np.uint8)
    if source.shape[:2] == (target_h, target_w):
        return source.astype(bool, copy=False)
    resized = cv2.resize(source, (target_w, target_h), interpolation=interpolation)
    return resized > 0


def build_diff_overlay(rgb_preview, current_mask, reference_mask):
    """当前结果相对反事实的新增为绿色，移除为红色，边界加深。"""
    rgb = np.asarray(rgb_preview, np.uint8)
    current = np.asarray(current_mask, bool)
    reference = np.asarray(reference_mask, bool)
    if current.shape != rgb.shape[:2] or reference.shape != rgb.shape[:2]:
        raise ValueError("蒙版与预览图尺寸不一致")
    out = rgb.astype(np.float32)
    added = np.logical_and(current, ~reference)
    removed = np.logical_and(reference, ~current)
    green = np.array([46, 160, 90], np.float32)
    red = np.array([214, 70, 70], np.float32)
    for region, color in ((added, green), (removed, red)):
        if np.any(region):
            out[region] = out[region] * 0.42 + color * 0.58
    for region, color in ((added, green), (removed, red)):
        edge = _boundary(region)
        if edge is not None:
            pixels = edge > 0
            out[pixels] = out[pixels] * 0.25 + color * 0.75
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def image_cache_token(cache_key):
    if cache_key is None:
        return "none"
    fields = [
        str(getattr(cache_key, "path", "")),
        str(getattr(cache_key, "size", "")),
        str(getattr(cache_key, "mtime_ns", "")),
        str(getattr(cache_key, "original_hw", "")),
        str(getattr(cache_key, "manifest_sha256", "")),
        str(getattr(cache_key, "preprocess_version", "")),
    ]
    return "\x1f".join(fields)


def prompt_fingerprint(cache_key, box_xyxy, points, mask_input=None):
    """完整提示指纹；坐标使用二进制双精度值，不做取整。"""
    digest = hashlib.sha256()
    digest.update(image_cache_token(cache_key).encode("utf-8"))
    digest.update(b"|box|")
    if box_xyxy is None:
        digest.update(b"none")
    else:
        digest.update(struct.pack("<4d", *[float(v) for v in box_xyxy]))
    for point in points:
        digest.update(b"|p|")
        digest.update(struct.pack("<2d", float(point.x), float(point.y)))
        digest.update(struct.pack("<i", int(point.label)))
    digest.update(b"|mask|")
    if mask_input is None:
        digest.update(b"none")
    else:
        array = np.ascontiguousarray(mask_input, np.float32)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_effect_result(plan, current_mask, current_candidate,
                       counterfactual_mask, counterfactual_candidate,
                       decodes=0, cache_hits=0, seconds=0.0):
    current = np.asarray(current_mask, bool)
    counterfactual = np.asarray(counterfactual_mask, bool)
    return EffectResult(
        plan=plan,
        current_mask=current,
        counterfactual_mask=counterfactual,
        current_candidate=int(current_candidate),
        counterfactual_candidate=int(counterfactual_candidate),
        metrics=compare_masks(counterfactual, current),
        decodes=int(decodes),
        cache_hits=int(cache_hits),
        seconds=float(seconds),
    )
