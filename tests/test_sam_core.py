from __future__ import annotations

import numpy as np

from interactive.sam_onnx import (
    SamGeometry,
    SamPrompt,
    SamResult,
    normalize_box,
    select_candidate,
    sigmoid,
)
from interactive.session import PromptHistory, PromptPoint, PromptSnapshot


def test_geometry_longest_side_and_coordinates():
    g = SamGeometry.create((6000, 8000))
    assert g.resized_hw == (768, 1024)
    assert g.scale_x == 1024 / 8000
    assert g.scale_y == 768 / 6000
    pts = g.points_to_input([[0, 0], [7999, 5999]])
    assert np.allclose(pts[0], [0, 0], atol=1e-4)
    assert pts[1, 0] <= 1024 and pts[1, 1] <= 768

    box = g.box_to_input([7900, 5900, 100, 100])
    assert box.shape == (2, 2)
    assert box[0, 0] < box[1, 0]
    assert box[0, 1] < box[1, 1]


def test_normalize_box_and_sigmoid():
    assert normalize_box([9, 8, 1, 2]) == (1.0, 2.0, 9.0, 8.0)
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0], np.float32))
    assert out[0] == 0.0
    assert abs(float(out[1]) - 0.5) < 1e-6
    assert out[2] == 1.0


def _result(logits, scores):
    masks = np.stack(logits).astype(np.float32)
    low = np.zeros((len(logits), 256, 256), np.float32)
    return SamResult(masks, np.array(scores, np.float32),
                     low, SamGeometry((100, 100), (100, 100)))


def test_candidate_prefers_official_multimask_then_point_constraints():
    base = np.zeros((100, 100), np.float32)
    cand0 = base.copy()
    cand0[10:90, 10:90] = 5
    cand1 = base.copy()
    cand1[10:50, 10:50] = 5
    cand2 = base.copy()
    cand2[55:90, 55:90] = 5
    cand3 = base.copy()
    cand3[20:80, 20:80] = 5
    result = _result([cand0, cand1, cand2, cand3], [0.99, 0.95, 0.80, 0.70])

    box_prompt = SamPrompt((10, 10, 90, 90), np.zeros((0, 2)), np.zeros((0,)))
    selected, ok = select_candidate(result, box_prompt, result.geometry)
    assert ok
    assert selected == 1

    # 框 + 正点位于左上半区后，应优先第 0 个 single-mask 候选。
    point_prompt = SamPrompt(
        (10, 10, 90, 90), np.array([[25, 25]], np.float32), np.array([1]))
    selected, ok = select_candidate(result, point_prompt, result.geometry)
    assert ok
    assert selected == 0

    # 第 0 个候选包住右下区域，负点要求排除，必须改选不包含该点的候选。
    point_prompt = SamPrompt(
        (10, 10, 90, 90), np.array([[70, 70]], np.float32), np.array([0]))
    selected, ok = select_candidate(result, point_prompt, result.geometry)
    assert ok
    assert selected in (1, 3)
    assert result.masks[selected][70, 70] <= 0


def test_prompt_history_undo_reset_and_branch():
    history = PromptHistory(max_steps=4)
    assert not history.can_undo
    p1 = PromptPoint(1, 2, 1)
    history.push(PromptSnapshot(box_xyxy=(0, 0, 10, 10)))
    history.push(PromptSnapshot(box_xyxy=(0, 0, 10, 10), points=(p1,)))
    assert history.current.points == (p1,)
    assert history.current_branch()[-1] == history.current
    history.undo()
    assert history.current.points == ()
    p2 = PromptPoint(3, 4, 0)
    history.push(PromptSnapshot(box_xyxy=(0, 0, 10, 10), points=(p2,)))
    assert not history.can_redo
    assert history.current.points == (p2,)
    history.set_candidate(2)
    assert history.current.candidate == 2
    history.reset()
    assert not history.current.has_prompt
    history.undo()
    assert history.current.candidate == 2


def test_prompt_point_source_and_backward_compatibility():
    manual = PromptPoint(1, 2, 1)
    color = PromptPoint(3, 4, 1, source="color")
    assert manual.source == "manual"
    assert color.source == "color"
    try:
        PromptPoint(3, 4, 1, source="unknown")
    except ValueError as exc:
        assert "来源" in str(exc)
    else:
        raise AssertionError("非法来源应被拒绝")


def test_sam_prompt_conversion():
    snap = PromptSnapshot(
        box_xyxy=(5, 8, 1, 2),
        points=(PromptPoint(3, 4, 1), PromptPoint(6, 7, 0)),
        candidate=1,
    )
    prompt = snap.as_sam_prompt()
    assert prompt.box_xyxy == (1.0, 2.0, 5.0, 8.0)
    assert prompt.point_labels.tolist() == [1, 0]
    assert prompt.prompt_point_count == 4
