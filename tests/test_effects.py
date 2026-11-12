from __future__ import annotations

import numpy as np

from interactive.effects import (
    build_box_only_plan,
    build_diff_overlay,
    build_point_loo_plan,
    compare_masks,
    mask_to_preview,
    point_display_name,
    point_local_metrics,
    prompt_fingerprint,
)
from interactive.session import PromptHistory, PromptPoint, PromptSnapshot


def test_compare_masks_identical_empty_and_shifted():
    mask = np.zeros((20, 30), bool)
    mask[3:12, 4:18] = True
    same = compare_masks(mask, mask)
    assert same.iou == 1.0
    assert same.changed_pixels == 0
    assert same.area_delta_pixels == 0
    assert same.boundary_p95 == 0.0

    shifted = np.zeros_like(mask)
    shifted[3:12, 6:20] = True
    metrics = compare_masks(mask, shifted)
    assert metrics.added_pixels > 0
    assert metrics.removed_pixels > 0
    assert metrics.changed_pixels == metrics.added_pixels + metrics.removed_pixels
    assert 0.0 < metrics.iou < 1.0
    assert metrics.boundary_p95 is not None
    assert metrics.boundary_p95 >= 1.0

    empty = compare_masks(np.zeros_like(mask), np.zeros_like(mask))
    assert empty.iou == 1.0
    assert empty.boundary_p95 is None


def test_compare_masks_reports_structure_and_relative_area():
    before = np.zeros((30, 30), bool)
    before[5:25, 5:25] = True
    before[12:16, 12:16] = False
    after = before.copy()
    after[5:12, 5:12] = True
    metrics = compare_masks(before, after)
    assert metrics.before_holes == 1
    assert metrics.before_components == 1
    assert metrics.area_delta_ratio is not None


def test_build_diff_overlay_marks_added_and_removed():
    rgb = np.full((12, 16, 3), 120, np.uint8)
    before = np.zeros((12, 16), bool)
    before[2:6, 2:8] = True
    after = np.zeros_like(before)
    after[6:10, 8:14] = True
    overlay = build_diff_overlay(rgb, after, before)
    assert overlay.shape == rgb.shape
    assert overlay.dtype == np.uint8
    assert np.any(overlay[3, 3] != rgb[3, 3])
    assert np.any(overlay[7, 10] != rgb[7, 10])


def test_mask_to_preview_and_point_local_metrics():
    mask = np.zeros((40, 60), bool)
    mask[10:30, 15:45] = True
    point = PromptPoint(30, 20, 1)
    local = point_local_metrics(mask, point, (40, 60))
    assert local.center_ok
    assert 0.0 <= local.support <= 1.0
    preview = mask_to_preview(mask, (20, 30))
    assert preview.shape == (20, 30)
    assert preview.mean() > 0


def test_build_point_loo_plan_replays_without_target_point():
    history = PromptHistory()
    p1 = PromptPoint(20, 20, 1)
    p2 = PromptPoint(30, 25, 0)
    history.push(PromptSnapshot(box_xyxy=(5, 5, 50, 45)))
    history.push(PromptSnapshot(
        box_xyxy=(5, 5, 50, 45), points=(p1,),
        mask_input=np.zeros((1, 1, 256, 256), np.float32), candidate=2))
    history.push(PromptSnapshot(
        box_xyxy=(5, 5, 50, 45), points=(p1, p2),
        mask_input=np.ones((1, 1, 256, 256), np.float32), candidate=1))
    states = history.current_branch()

    first = build_point_loo_plan(states, 0)
    assert first.box_xyxy == (5.0, 5.0, 50.0, 45.0)
    assert first.steps[0].points == ()
    assert first.steps[-1].points == (p2,)
    assert first.steps[-1].input_candidate is None

    second = build_point_loo_plan(states, 1)
    assert second.steps[0].points == ()
    assert second.steps[-1].points == (p1,)
    assert point_display_name((p1, p2), 1) == "负1"


def test_point_loo_uses_latest_box_adjusted_baseline():
    history = PromptHistory()
    p1 = PromptPoint(10, 10, 1)
    p2 = PromptPoint(20, 20, 0)
    p3 = PromptPoint(30, 30, 1)
    history.push(PromptSnapshot(box_xyxy=(0, 0, 40, 40)))
    history.push(PromptSnapshot(
        box_xyxy=(0, 0, 40, 40), points=(p1,),
        mask_input=np.zeros((1, 1, 256, 256), np.float32)))
    history.push(PromptSnapshot(
        box_xyxy=(0, 0, 40, 40), points=(p1, p2),
        mask_input=np.ones((1, 1, 256, 256), np.float32)))
    history.push(PromptSnapshot(
        box_xyxy=(2, 2, 38, 38), points=(p1, p2), mask_input=None))
    history.push(PromptSnapshot(
        box_xyxy=(2, 2, 38, 38), points=(p1, p2, p3),
        mask_input=np.zeros((1, 1, 256, 256), np.float32)))
    plan = build_point_loo_plan(history.current_branch(), 0)
    assert plan.box_xyxy == (2.0, 2.0, 38.0, 38.0)
    assert plan.steps[0].points == (p2,)
    assert plan.steps[-1].points == (p2, p3)


def test_box_only_plan_and_distinct_exact_fingerprints():
    history = PromptHistory()
    history.push(PromptSnapshot(box_xyxy=(1, 2, 30, 40)))
    history.push(PromptSnapshot(
        box_xyxy=(1, 2, 30, 40), points=(PromptPoint(3, 4, 1),),
        mask_input=np.zeros((1, 1, 256, 256), np.float32)))
    plan = build_box_only_plan(history.current_branch())
    assert len(plan.steps) == 1
    assert plan.steps[0].points == ()

    key = object()
    first = prompt_fingerprint(key, (0, 0, 10, 10), (PromptPoint(1.0, 2.0, 1),))
    second = prompt_fingerprint(key, (0, 0, 10, 10), (PromptPoint(1.000001, 2.0, 1),))
    assert first != second
