from __future__ import annotations

import json
import threading

import numpy as np
from PIL import Image

from interactive.effects import build_box_only_plan, build_point_loo_plan
from interactive.sam_onnx import SamGeometry, SamResult
from interactive.session import PromptHistory, PromptPoint, PromptSnapshot
from interactive.worker import InteractiveWorker


class _FakeImage:
    def __init__(self, cache_key, geometry):
        self.cache_key = cache_key
        self.geometry = geometry


class _FakeEngine:
    def __init__(self):
        self.encodes = 0
        self.decodes = 0

    def encode(self, rgb, cache_key):
        self.encodes += 1
        return _FakeImage(cache_key, SamGeometry.create(rgb.shape[:2]))

    def decode(self, image, prompt):
        self.decodes += 1
        h, w = 64, 80
        logits = np.full((3, h, w), -10.0, np.float32)
        logits[:, 10:50, 15:65] = 10.0
        result = SamResult(
            logits, np.array([0.9, 0.7, 0.5], np.float32),
            np.zeros((3, 256, 256), np.float32),
            SamGeometry((h, w), (h, w)),
        )
        result.selected = 0
        return result


class _BlockingEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def decode(self, image, prompt):
        if self.block:
            self.started.set()
            self.release.wait(5)
        return super().decode(image, prompt)


def _make_models(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "sam_vit_b_manifest.json").write_text(json.dumps({
        "encoder": {"sha256": "a" * 64},
        "decoder": {"sha256": "b" * 64},
    }), encoding="utf-8")
    return models


def test_worker_encodes_once_and_latest_decode_wins(qtbot, tmp_path):
    models = _make_models(tmp_path)
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 128, np.uint8), "RGB").save(src)
    engine = _FakeEngine()
    worker = InteractiveWorker(
        models, tmp_path / "out", threads=1,
        engine_factory=lambda *args: engine,
    )
    worker.start()
    qtbot.waitUntil(lambda: worker.isRunning(), timeout=3000)
    loaded_events = []
    encoded_events = []
    decoded_events = []
    worker.loaded.connect(loaded_events.append)
    worker.encoded.connect(encoded_events.append)
    worker.decoded.connect(decoded_events.append)
    try:
        worker.request_load(1, str(src))
        qtbot.waitUntil(lambda: bool(loaded_events and encoded_events), timeout=5000)
        assert loaded_events[0].original_hw == (40, 60)
        assert encoded_events[0].document_id == 1
        assert engine.encodes == 1

        snapshot = PromptSnapshot(box_xyxy=(5, 5, 55, 35))
        worker.request_decode(1, 1, snapshot)
        qtbot.waitUntil(lambda: len(decoded_events) == 1, timeout=5000)
        assert decoded_events[0].document_id == 1
        assert decoded_events[0].revision == 1
        assert engine.decodes == 1

        # 新提示只重跑 decoder，不重新 encode。
        worker.request_decode(1, 2, snapshot)
        qtbot.waitUntil(lambda: len(decoded_events) == 2, timeout=5000)
        assert engine.encodes == 1
        assert engine.decodes == 2

        # 过期结果不覆盖新 revision：最新一次请求获胜。
        worker.request_decode(1, 3, snapshot)
        worker.request_decode(1, 4, snapshot)
        qtbot.waitUntil(lambda: any(x.revision == 4 for x in decoded_events), timeout=5000)
        assert all(x.revision != 3 for x in decoded_events)

        exported_events = []
        worker.exported.connect(exported_events.append)
        logits = np.full((64, 80), -10.0, np.float32)
        logits[10:50, 15:65] = 10.0
        worker.request_export(
            1, 4, str(src), logits, encoded_events[0].cache_key,
            64, False, tmp_path / "out")
        qtbot.waitUntil(lambda: len(exported_events) == 1, timeout=10000)
        result = exported_events[0].result
        assert result["fg_path"] and result["mask_path"]
        assert (tmp_path / "out").exists()
    finally:
        worker.stop()
        assert worker.wait(5000)


def test_worker_cancels_inflight_effect_without_emitting_stale_result(
        qtbot, tmp_path):
    models = _make_models(tmp_path)
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 128, np.uint8), "RGB").save(src)
    engine = _BlockingEngine()
    worker = InteractiveWorker(
        models, tmp_path / "out", threads=1,
        engine_factory=lambda *args: engine,
    )
    worker.start()
    qtbot.waitUntil(lambda: worker.isRunning(), timeout=3000)
    encoded_events = []
    decoded_events = []
    effect_items = []
    worker.encoded.connect(encoded_events.append)
    worker.decoded.connect(decoded_events.append)
    worker.effect_ready.connect(effect_items.append)
    try:
        worker.request_load(1, str(src))
        qtbot.waitUntil(lambda: bool(encoded_events), timeout=5000)
        worker.request_decode(1, 1, PromptSnapshot(box_xyxy=(5, 5, 55, 35)))
        qtbot.waitUntil(lambda: len(decoded_events) == 1, timeout=5000)

        history = PromptHistory()
        history.push(PromptSnapshot(box_xyxy=(5, 5, 55, 35)))
        history.push(PromptSnapshot(
            box_xyxy=(5, 5, 55, 35), points=(PromptPoint(20, 20, 1),),
            mask_input=np.zeros((1, 1, 256, 256), np.float32)))
        plan = build_point_loo_plan(history.current_branch(), 0)
        current_mask = np.zeros((64, 80), bool)
        current_mask[10:50, 15:65] = True
        engine.block = True
        worker.request_effects(1, 2, 11, current_mask, 0, [plan])
        qtbot.waitUntil(lambda: engine.started.is_set(), timeout=5000)
        worker.cancel_effects()
        engine.release.set()
        qtbot.wait(250)
        assert effect_items == []
    finally:
        engine.release.set()
        worker.stop()
        assert worker.wait(5000)


def test_worker_runs_effect_batch_without_touching_formal_result(
        qtbot, tmp_path):
    models = _make_models(tmp_path)
    src = tmp_path / "plant.png"
    Image.fromarray(np.full((40, 60, 3), 128, np.uint8), "RGB").save(src)
    engine = _FakeEngine()
    worker = InteractiveWorker(
        models, tmp_path / "out", threads=1,
        engine_factory=lambda *args: engine,
    )
    worker.start()
    qtbot.waitUntil(lambda: worker.isRunning(), timeout=3000)
    encoded_events = []
    decoded_events = []
    effect_items = []
    finished = []
    worker.encoded.connect(encoded_events.append)
    worker.decoded.connect(decoded_events.append)
    worker.effect_ready.connect(effect_items.append)
    worker.effects_finished.connect(finished.append)
    try:
        worker.request_load(1, str(src))
        qtbot.waitUntil(lambda: bool(encoded_events), timeout=5000)
        worker.request_decode(1, 1, PromptSnapshot(box_xyxy=(5, 5, 55, 35)))
        qtbot.waitUntil(lambda: len(decoded_events) == 1, timeout=5000)

        history = PromptHistory()
        history.push(PromptSnapshot(box_xyxy=(5, 5, 55, 35)))
        history.push(PromptSnapshot(
            box_xyxy=(5, 5, 55, 35), points=(PromptPoint(20, 20, 1),),
            mask_input=np.zeros((1, 1, 256, 256), np.float32), candidate=0))
        states = history.current_branch()
        plans = [
            build_box_only_plan(states),
            build_point_loo_plan(states, 0),
        ]
        current_mask = np.zeros((64, 80), bool)
        current_mask[10:50, 15:65] = True
        before_decodes = engine.decodes
        formal_count = len(decoded_events)
        worker.request_effects(1, 2, 7, current_mask, 0, plans)
        qtbot.waitUntil(lambda: len(effect_items) == 2, timeout=5000)
        qtbot.waitUntil(lambda: len(finished) == 1, timeout=5000)
        assert finished[0].request_id == 7
        assert finished[0].completed == 2
        assert effect_items[0].result.metrics.changed_pixels == 0
        assert effect_items[1].result.plan.kind == "point_loo"
        assert len(decoded_events) == formal_count
        assert engine.decodes > before_decodes

        after_first = engine.decodes
        worker.request_effects(1, 2, 8, current_mask, 0, plans)
        qtbot.waitUntil(lambda: len(effect_items) == 4, timeout=5000)
        qtbot.waitUntil(lambda: len(finished) == 2, timeout=5000)
        assert engine.decodes == after_first
        assert effect_items[-1].result.cache_hits >= 1
    finally:
        worker.stop()
        assert worker.wait(5000)
