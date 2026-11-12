from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal

from interactive.effects import make_effect_result
from interactive.main_window import MainWindow
from interactive.sam_onnx import SamGeometry, SamResult, make_cache_key
from interactive.worker import (
    DecodedEvent,
    EffectItemEvent,
    EffectProgressEvent,
    EffectsFinishedEvent,
    EncodedEvent,
    LoadedEvent,
)


class FakeWorker(QObject):
    loaded = Signal(object)
    encoded = Signal(object)
    decoded = Signal(object)
    exported = Signal(object)
    failed = Signal(object)
    phase = Signal(str)
    effect_ready = Signal(object)
    effect_progress = Signal(object)
    effects_finished = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loads = []
        self.decodes = []
        self.exports = []
        self.effects = []

    def start(self):
        pass

    def isRunning(self):
        return False

    def stop(self):
        pass

    def wait(self, *args):
        return True

    def request_load(self, document_id, path):
        self.loads.append((document_id, path))

    def request_decode(self, document_id, revision, snapshot):
        self.decodes.append((document_id, revision, snapshot))

    def request_export(self, *args):
        self.exports.append(args)

    def request_effects(self, document_id, revision, request_id,
                        current_mask, current_candidate, plans):
        self.effects.append((
            document_id, revision, request_id, current_mask,
            current_candidate, tuple(plans)))

    def cancel_effects(self):
        pass


def _window_with_point(qtbot, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 100, np.uint8), "RGB").save(src)
    window = MainWindow(
        models_dir=models, out_dir=tmp_path / "out", worker_factory=FakeWorker)
    qtbot.addWidget(window)
    window._set_folder(tmp_path, [str(src)])
    doc_id = window._document_id
    preview = np.full((40, 60, 3), 100, np.uint8)
    cache_key = make_cache_key(str(src), (40, 60), "f" * 64)
    window._on_loaded(
        LoadedEvent(doc_id, str(src), preview, (40, 60), cache_key, 0.1))
    window._on_encoded(EncodedEvent(doc_id, cache_key, 0.1, False))
    window.on_box(5, 5, 55, 35)
    rev = window.worker.decodes[-1][1]
    logits = np.full((3, 64, 80), -10.0, np.float32)
    logits[:, 10:50, 10:70] = 10.0
    result = SamResult(
        logits, np.array([0.9, 0.8, 0.7], np.float32),
        np.zeros((3, 256, 256), np.float32),
        SamGeometry((40, 60), (40, 60)), selected=0)
    window._on_decoded(DecodedEvent(doc_id, rev, result, 0.1))
    for x, y, label, source in (
            (20, 20, 1, "manual"), (30, 25, 1, "color")):
        window.on_point(x, y, label, source=source)
        rev = window.worker.decodes[-1][1]
        window._on_decoded(DecodedEvent(doc_id, rev, result, 0.1))
    return window, doc_id, rev


def test_effect_dialog_is_nonmodal_and_replaces_detail_popup(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    window, doc_id, rev = _window_with_point(qtbot, tmp_path)
    window.show_active_effects()
    dialog = window._effects_dialog
    qtbot.addWidget(dialog)
    assert dialog is not None
    assert not dialog.isModal()
    assert dialog.section_list.count() == 6
    assert dialog.point_table.rowCount() == 2
    assert dialog.point_table.columnCount() == 8
    assert dialog.candidate_table.rowCount() == 3
    assert dialog.point_table.item(1, 1).text() == "颜色"
    assert "模型自评" in dialog.candidate_table.horizontalHeaderItem(1).text()
    assert "工程包未参与 SAM 解码" in dialog.profile_impact.text()
    assert "matting" in dialog.export_summary.text()
    window._last_export_summary = {
        "fg_path": "C:/out/plant_fg.png",
        "mask_path": "C:/out/plant_fg_mask.png",
        "seconds": 1.25,
        "time": "2026-11-12 09:30:00",
    }
    dialog._fill_export()
    assert "plant_fg.png" in dialog.export_detail.toPlainText()

    dialog.point_filter.setCurrentIndex(1)
    assert dialog.point_table.rowCount() == 2
    dialog.point_filter.setCurrentIndex(2)
    assert dialog.point_table.rowCount() == 0
    dialog.point_filter.setCurrentIndex(4)
    assert dialog.point_table.rowCount() == 1
    assert dialog._selected_point_index() == 1
    dialog.point_filter.setCurrentIndex(0)
    assert dialog.point_table.rowCount() == 2
    dialog.point_table.selectRow(0)
    assert dialog.selected_point_btn.isEnabled()
    dialog.selected_point_btn.click()
    assert len(window.worker.effects) == 1
    _doc, effect_rev, request_id, current_mask, candidate, plans = (
        window.worker.effects[-1])
    assert effect_rev == rev
    assert candidate == 0
    assert len(plans) == 1
    plan = plans[0]
    assert plan.kind == "point_loo"
    assert plan.target_point_index == 0

    counter = current_mask.copy()
    counter[10:20, 10:30] = False
    effect_result = make_effect_result(
        plan, current_mask, 0, counter, 2,
        decodes=2, cache_hits=1, seconds=0.2)
    window._on_effect_progress(EffectProgressEvent(
        doc_id, rev, request_id, 0, 1, "正在分析第 1/1 项"))
    window._on_effect_ready(EffectItemEvent(
        doc_id, rev, request_id, effect_result, 1, 1, 0.2))
    assert dialog._results[plan.key] is effect_result
    assert "完成" in dialog.point_table.item(0, 3).text()
    assert "翻转像素" in dialog.point_detail.toPlainText()
    assert dialog.heat_label._source is not None
    assert dialog.ab_reference._source is not None
    window._on_effects_finished(EffectsFinishedEvent(
        doc_id, rev, request_id, 1, 1, 0.3))
    assert not dialog._analysis_active

    window._dirty = False
    dialog.close()
    window.close()
