from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QPushButton, QScrollArea

from interactive.candidate_workbench import CandidatePreviewDialog, CandidateWorkbench
from interactive.canvas import numpy_rgb_to_qimage


def test_candidate_workbench_has_detail_preview(qtbot):
    image = numpy_rgb_to_qimage(np.zeros((40, 60, 3), dtype=np.uint8))
    candidates = [{"title": "候选 1", "score": 0.9, "image": image}]
    dialog = CandidateWorkbench(candidates, "plant.png", lambda *args: None)
    qtbot.addWidget(dialog)
    preview_buttons = dialog.findChildren(QPushButton)
    assert any(button.text() == "放大查看细节" for button in preview_buttons)
    assert dialog.findChildren(QScrollArea)
    assert len(dialog._cards) == 1
    dialog._select_card(0)
    assert dialog._cards[0].isChecked()
    preview = CandidatePreviewDialog(image, "候选 1", dialog)
    qtbot.addWidget(preview)
    assert preview.windowTitle() == "细节预览 — 候选 1"
