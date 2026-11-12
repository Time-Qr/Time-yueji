# -*- coding: utf-8 -*-
"""交互提示的纯数据模型与撤销历史，不依赖 Qt。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .sam_onnx import SamPrompt, normalize_box


@dataclass(frozen=True)
class PromptPoint:
    x: float
    y: float
    label: int
    source: str = "manual"

    def __post_init__(self):
        if int(self.label) not in (0, 1):
            raise ValueError("点标签只能是 0（排除）或 1（加入）")
        if self.source not in ("manual", "color"):
            raise ValueError("点来源只能是 manual 或 color")


@dataclass(frozen=True)
class PromptSnapshot:
    box_xyxy: tuple[float, float, float, float] | None = None
    points: tuple[PromptPoint, ...] = ()
    mask_input: np.ndarray | None = None
    candidate: int = 0

    def as_sam_prompt(self) -> SamPrompt:
        if self.points:
            pts = np.array([[p.x, p.y] for p in self.points], np.float32)
            labels = np.array([p.label for p in self.points], np.int64)
        else:
            pts = np.zeros((0, 2), np.float32)
            labels = np.zeros((0,), np.int64)
        box = normalize_box(self.box_xyxy) if self.box_xyxy is not None else None
        return SamPrompt(box, pts, labels, self.mask_input)

    @property
    def has_prompt(self):
        return self.box_xyxy is not None or bool(self.points)


EMPTY_SNAPSHOT = PromptSnapshot()


class PromptHistory:
    """按编辑顺序保存提示快照，支持撤销且限制内存占用。"""

    def __init__(self, max_steps=20):
        self.max_steps = max(2, int(max_steps))
        self._states = [EMPTY_SNAPSHOT]
        self._index = 0

    @property
    def current(self):
        return self._states[self._index]

    @property
    def can_undo(self):
        return self._index > 0

    @property
    def can_redo(self):
        return self._index < len(self._states) - 1

    @property
    def depth(self):
        return self._index

    @property
    def current_index(self):
        return self._index

    def current_branch(self):
        """返回当前撤销分支，供反事实重放使用。"""
        return tuple(self._states[:self._index + 1])

    def push(self, snapshot):
        if not isinstance(snapshot, PromptSnapshot):
            raise TypeError("snapshot 必须是 PromptSnapshot")
        self._states = self._states[:self._index + 1]
        self._states.append(snapshot)
        self._index += 1
        while len(self._states) > self.max_steps:
            self._states.pop(0)
            self._index -= 1
        return snapshot

    def undo(self):
        if self.can_undo:
            self._index -= 1
        return self.current

    def redo(self):
        if self.can_redo:
            self._index += 1
        return self.current

    def set_candidate(self, index):
        updated = replace(self.current, candidate=int(index))
        self._states[self._index] = updated
        return updated

    def reset(self):
        return self.push(EMPTY_SNAPSHOT)


def copy_mask_input(mask_input):
    if mask_input is None:
        return None
    arr = np.asarray(mask_input, np.float32)
    return arr.copy()
