from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal

from interactive.main_window import MainWindow
from interactive.sam_onnx import SamGeometry, SamResult, make_cache_key
from interactive.worker import DecodedEvent, EncodedEvent, LoadedEvent


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
        self.cancelled_effects = 0

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
        self.cancelled_effects += 1


def test_main_window_box_decode_candidate_and_export_request(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    models = tmp_path / "models"
    models.mkdir()
    out = tmp_path / "out"
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 100, np.uint8), "RGB").save(src)

    window = MainWindow(models_dir=models, out_dir=out, worker_factory=FakeWorker)
    qtbot.addWidget(window)
    window.show()
    window._set_folder(tmp_path, [str(src)])
    doc_id = window._document_id
    assert window.worker.loads and window.worker.loads[0][0] == doc_id

    preview = np.full((40, 60, 3), 100, np.uint8)
    cache_key = make_cache_key(str(src), (40, 60), "f" * 64)
    window._on_loaded(LoadedEvent(doc_id, str(src), preview, (40, 60), cache_key, 0.1))
    window._on_encoded(EncodedEvent(doc_id, cache_key, 0.1, False))
    assert window.canvas.has_image

    window.on_box(5, 5, 55, 35)
    assert window.worker.decodes
    _, rev, snap = window.worker.decodes[-1]
    assert snap.box_xyxy == (5.0, 5.0, 55.0, 35.0)

    logits = np.full((3, 64, 80), -10.0, np.float32)
    logits[:, 10:50, 10:70] = 10.0
    result = SamResult(
        logits, np.array([0.9, 0.8, 0.7], np.float32),
        np.zeros((3, 256, 256), np.float32),
        SamGeometry((40, 60), (40, 60)),
        selected=0,
    )
    window._on_decoded(DecodedEvent(doc_id, rev, result, 0.1))
    assert window._result_valid()
    assert window.export_action.isEnabled()
    assert "候选 1/3" in window.candidate_label.text()

    before = window._candidate
    window.step_candidate(1)
    assert window._candidate == (before + 1) % 3
    assert window._result_valid()

    assert window.export_current()
    assert len(window.worker.exports) == 1
    args = window.worker.exports[0]
    assert args[0] == doc_id
    assert args[2] == str(src)
    window._dirty = False
    window.close()


def test_main_window_adjusted_box_preserves_points_and_output_dir(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    models = tmp_path / "models"
    models.mkdir()
    out = tmp_path / "out"
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 100, np.uint8), "RGB").save(src)

    window = MainWindow(models_dir=models, out_dir=out, worker_factory=FakeWorker)
    qtbot.addWidget(window)
    window._set_folder(tmp_path, [str(src)])
    doc_id = window._document_id
    preview = np.full((40, 60, 3), 100, np.uint8)
    cache_key = make_cache_key(str(src), (40, 60), "f" * 64)
    window._on_loaded(LoadedEvent(doc_id, str(src), preview, (40, 60), cache_key, 0.1))
    window._on_encoded(EncodedEvent(doc_id, cache_key, 0.1, False))

    window.on_box(5, 5, 55, 35)
    window.on_point(20, 20, 1)
    window.on_box_adjusted(8, 6, 50, 34)
    snap = window._history.current
    assert snap.box_xyxy == (8.0, 6.0, 50.0, 34.0)
    assert [(p.x, p.y, p.label) for p in snap.points] == [(20.0, 20.0, 1)]

    extra = tmp_path / "extra.png"
    Image.fromarray(np.full((20, 30, 3), 80, np.uint8), "RGB").save(extra)
    window._append_files([str(src), str(extra)])
    assert window._files == [str(src.resolve()), str(extra.resolve())]

    selected_out = tmp_path / "selected-out"
    selected_out.mkdir()
    window.out_dir = selected_out
    assert window.out_dir == selected_out.resolve()
    window._dirty = False
    window.close()


def test_main_window_color_assist_marks_generated_points(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    monkeypatch.setattr(
        "interactive.main_window.suggest_points",
        lambda *args, **kwargs: [(20.0, 20.0, 0.9)])
    models = tmp_path / "models"
    models.mkdir()
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 100, np.uint8), "RGB").save(src)
    window = MainWindow(models_dir=models, out_dir=tmp_path / "out",
                        worker_factory=FakeWorker)
    qtbot.addWidget(window)
    window._set_folder(tmp_path, [str(src)])
    doc_id = window._document_id
    preview = np.full((40, 60, 3), 100, np.uint8)
    cache_key = make_cache_key(str(src), (40, 60), "f" * 64)
    window._on_loaded(LoadedEvent(doc_id, str(src), preview, (40, 60), cache_key, 0.1))
    window._on_encoded(EncodedEvent(doc_id, cache_key, 0.1, False))
    window.on_box(5, 5, 55, 35)
    window._color_samples = [np.array([20, 150, 30], np.uint8)]
    window._color_assist_active = True
    window._generate_color_points()
    points = window._history.current.points
    assert len(points) == 1
    assert points[0].source == "color"
    window._dirty = False
    window.close()


def test_main_window_actions_expose_context_help(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    models = tmp_path / "models"
    models.mkdir()
    window = MainWindow(models_dir=models, out_dir=tmp_path / "out",
                        worker_factory=FakeWorker)
    qtbot.addWidget(window)
    assert "选择一个图片文件夹" in window.open_action.toolTip()
    assert window.box_action.whatsThis()
    assert window.canvas.whatsThis()
    assert "Shift+F1" in window.whats_this_action.toolTip()
    assert window.profile_label.text() == "工程包：未选择"
    assert "当前图片" in window.sync_profile_action.toolTip()
    assert window.toolbar_profile_label.text() == "当前工程包：未选择"
    assert "下一步" in window.status_label.text()
    window.close()


def test_main_window_syncs_current_operation_to_profile(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    monkeypatch.setattr(
        "interactive.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("室内绿萝示例", True))
    models = tmp_path / "models"
    models.mkdir()
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 100, np.uint8), "RGB").save(src)
    window = MainWindow(models_dir=models, out_dir=tmp_path / "out",
                        worker_factory=FakeWorker)
    qtbot.addWidget(window)
    window._set_folder(tmp_path, [str(src)])
    doc_id = window._document_id
    preview = np.full((40, 60, 3), 100, np.uint8)
    cache_key = make_cache_key(str(src), (40, 60), "f" * 64)
    window._on_loaded(LoadedEvent(doc_id, str(src), preview, (40, 60), cache_key, 0.1))
    window._on_encoded(EncodedEvent(doc_id, cache_key, 0.1, False))
    window.on_box(5, 5, 55, 35)
    window.on_point(20, 20, 1)

    profile_path = tmp_path / "profile.json"
    from interactive.profile import ObjectProfile, load_profile
    window._profile = ObjectProfile("绿萝")
    window._profile_path = profile_path
    window._last_color_samples = [np.array([20, 150, 30], np.uint8)]
    window.sync_current_to_profile()

    loaded = load_profile(profile_path)
    assert len(loaded.examples) == 1
    assert loaded.examples[0]["name"] == "室内绿萝示例"
    assert loaded.examples[0]["points"][0]["label"] == 1
    assert loaded.examples[0]["points"][0]["source"] == "manual"
    assert loaded.examples[0]["color_samples_rgb"] == [[20, 150, 30]]
    window._dirty = False
    window.close()


def test_main_window_records_candidate_feedback_in_profile(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("interactive.main_window.sample_manifest_hash", lambda _p: "f" * 64)
    monkeypatch.setattr(MainWindow, "_show_model_error", lambda *a, **k: None)
    models = tmp_path / "models"
    models.mkdir()
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 100, np.uint8), "RGB").save(src)
    window = MainWindow(models_dir=models, out_dir=tmp_path / "out",
                        worker_factory=FakeWorker)
    qtbot.addWidget(window)
    window._set_folder(tmp_path, [str(src)])
    doc_id = window._document_id
    preview = np.full((40, 60, 3), 100, np.uint8)
    cache_key = make_cache_key(str(src), (40, 60), "f" * 64)
    window._on_loaded(LoadedEvent(doc_id, str(src), preview, (40, 60), cache_key, 0.1))
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

    profile_path = tmp_path / "profile.json"
    from interactive.profile import ObjectProfile
    window._profile = ObjectProfile("绿萝")
    window._profile_path = profile_path
    window._update_profile_indicator()
    assert window.profile_label.text() == "工程包：绿萝"
    assert window.toolbar_profile_label.text() == "当前工程包：绿萝"
    window._record_candidate_feedback(1, 4, ["保留花盆"], "测试反馈")
    assert window._profile.feedback[0].selected == 1
    assert profile_path.is_file()
    window._dirty = False
    window.close()
