# -*- coding: utf-8 -*-
"""交互界面的单后台线程：模型会话、缓存、任务代次和导出都在这里串行执行。"""
from __future__ import annotations

import os
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QThread, Signal

import generate_masks as gm
from . import pipeline
from .effects import make_effect_result, prompt_fingerprint
from .sam_onnx import (
    SamCacheKey,
    SamOnnxEngine,
    SamPrompt,
    make_cache_key,
    sample_manifest_hash,
)


@dataclass
class LoadedEvent:
    document_id: int
    path: str
    preview_rgb: np.ndarray
    original_hw: tuple[int, int]
    cache_key: SamCacheKey
    seconds: float


@dataclass
class EncodedEvent:
    document_id: int
    cache_key: SamCacheKey
    seconds: float
    cached: bool


@dataclass
class DecodedEvent:
    document_id: int
    revision: int
    result: object
    seconds: float


@dataclass
class ExportedEvent:
    document_id: int
    revision: int
    result: dict
    seconds: float


@dataclass
class WorkerError:
    document_id: int | None
    revision: int | None
    kind: str
    message: str
    trace: str = ""
    request_id: int | None = None


@dataclass
class EffectItemEvent:
    document_id: int
    revision: int
    request_id: int
    result: object
    completed: int
    total: int
    seconds: float


@dataclass
class EffectsFinishedEvent:
    document_id: int
    revision: int
    request_id: int
    completed: int
    total: int
    seconds: float


@dataclass
class EffectProgressEvent:
    document_id: int
    revision: int
    request_id: int
    completed: int
    total: int
    message: str


@dataclass
class _LoadJob:
    seq: int
    document_id: int
    path: str


@dataclass
class _DecodeJob:
    seq: int
    document_id: int
    revision: int
    snapshot: object


@dataclass
class _ExportJob:
    seq: int
    document_id: int
    revision: int
    path: str
    selected_logits: np.ndarray
    cache_key: SamCacheKey
    work_size: int
    use_matte: bool
    out_dir: str


@dataclass
class _DecodedEffect:
    selected: int
    low_res_logits: np.ndarray
    mask: np.ndarray

    def next_mask_input(self, candidate):
        index = self.selected if candidate is None else int(candidate)
        index = min(max(0, index), int(self.low_res_logits.shape[0]) - 1)
        return self.low_res_logits[index][None, None].astype(np.float32)


@dataclass
class _EffectJob:
    seq: int
    document_id: int
    revision: int
    request_id: int
    current_mask: np.ndarray
    current_candidate: int
    plans: tuple
    started_at: float = field(default_factory=time.time)
    plan_index: int = 0
    step_index: int = 0
    last_decoded: object = None
    plan_decodes: int = 0
    plan_cache_hits: int = 0
    plan_started_at: float = field(default_factory=time.time)
    total_decodes: int = 0
    total_cache_hits: int = 0


class InteractiveWorker(QThread):
    loaded = Signal(object)
    encoded = Signal(object)
    decoded = Signal(object)
    exported = Signal(object)
    failed = Signal(object)
    phase = Signal(str)
    effect_ready = Signal(object)
    effect_progress = Signal(object)
    effects_finished = Signal(object)

    def __init__(self, models_dir, out_dir, threads=6, parent=None,
                 engine_factory=None, preview_side=1800):
        super().__init__(parent)
        self.models_dir = os.path.abspath(models_dir)
        self.out_dir = os.path.abspath(out_dir)
        self.threads = int(threads)
        self.preview_side = int(preview_side)
        self._engine_factory = engine_factory
        self._cond = threading.Condition()
        self._pending_load = None
        self._pending_decode = None
        self._pending_export = None
        self._pending_effect = None
        self._load_seq = 0
        self._decode_seq = 0
        self._export_seq = 0
        self._effect_seq = 0
        self._stopping = False
        self._engine = None
        self._manifest_sha = None
        self._cache = OrderedDict()
        self._cache_limit = 2
        self._effect_cache = OrderedDict()
        self._effect_cache_limit = 16
        self._active_doc = None
        self._active_image = None
        self._active_key = None

    def request_load(self, document_id, path):
        with self._cond:
            self._load_seq += 1
            self._pending_load = _LoadJob(self._load_seq, int(document_id), os.path.abspath(path))
            self._pending_decode = None
            self._pending_export = None
            self._pending_effect = None
            self._effect_seq += 1
            self._cond.notify()

    def request_decode(self, document_id, revision, snapshot):
        with self._cond:
            self._decode_seq += 1
            self._pending_decode = _DecodeJob(
                self._decode_seq, int(document_id), int(revision), snapshot)
            self._pending_effect = None
            self._effect_seq += 1
            self._cond.notify()

    def request_effects(self, document_id, revision, request_id,
                        current_mask, current_candidate, plans):
        plans = tuple(plans)
        if not plans:
            return
        with self._cond:
            self._effect_seq += 1
            self._pending_effect = _EffectJob(
                self._effect_seq, int(document_id), int(revision), int(request_id),
                np.array(current_mask, bool, copy=True), int(current_candidate), plans)
            self._cond.notify()

    def cancel_effects(self):
        with self._cond:
            self._effect_seq += 1
            self._pending_effect = None

    def request_export(self, document_id, revision, path, selected_logits,
                       cache_key, work_size, use_matte, out_dir):
        with self._cond:
            self._export_seq += 1
            self._pending_export = _ExportJob(
                self._export_seq, int(document_id), int(revision), os.path.abspath(path),
                np.array(selected_logits, np.float32, copy=True), cache_key,
                int(work_size), bool(use_matte), os.path.abspath(out_dir))
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stopping = True
            self._cond.notify_all()

    def _is_stale_decode(self, job):
        with self._cond:
            return job.seq != self._decode_seq

    def _emit_error(self, exc, doc_id=None, revision=None, kind="error",
                    request_id=None):
        self.failed.emit(WorkerError(
            doc_id, revision, kind,
            "%s: %s" % (type(exc).__name__, str(exc)),
            traceback.format_exc(),
            request_id,
        ))

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if self._manifest_sha is None:
            self._manifest_sha = sample_manifest_hash(self.models_dir)
        enc = os.path.join(self.models_dir, "sam_vit_b_encoder.onnx")
        dec = os.path.join(self.models_dir, "sam_vit_b_decoder.onnx")
        if self._engine_factory is not None:
            self._engine = self._engine_factory(enc, dec, self.threads)
        else:
            self._engine = SamOnnxEngine(enc, dec, intra_op_threads=self.threads)
        return self._engine

    def _cache_put(self, image):
        key = image.cache_key
        self._cache.pop(key, None)
        self._cache[key] = image
        while len(self._cache) > self._cache_limit:
            old_key, _old = self._cache.popitem(last=False)
            if old_key == self._active_key and len(self._cache) == 1:
                self._cache[old_key] = _old
                break

    @staticmethod
    def _effect_cache_get(cache, key):
        value = cache.pop(key, None)
        if value is not None:
            cache[key] = value
        return value

    def _effect_cache_put(self, key, value):
        self._effect_cache.pop(key, None)
        self._effect_cache[key] = value
        while len(self._effect_cache) > self._effect_cache_limit:
            self._effect_cache.popitem(last=False)

    def _load(self, job):
        t0 = time.time()
        self.phase.emit("读取图片…")
        rgb, _exif, _icc, _src_alpha = gm.load_image(job.path)
        if rgb is None or rgb.size == 0:
            raise RuntimeError("图片读取失败")
        with self._cond:
            if job.seq != self._load_seq:
                return
        original_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        key = make_cache_key(job.path, original_hw, self._manifest_sha)
        preview = pipeline.make_preview_rgb(rgb, self.preview_side)
        with self._cond:
            if job.seq != self._load_seq:
                return
            self._active_doc = job.document_id
            self._active_key = key
            self._active_image = None
        self.loaded.emit(LoadedEvent(
            job.document_id, job.path, preview, original_hw, key, time.time() - t0))

        image = self._cache.get(key)
        cached = image is not None
        t1 = time.time()
        if image is None:
            self.phase.emit("正在分析图像，首次等待会久一些…")
            image = self._ensure_engine().encode(rgb, key)
            self._cache_put(image)
        with self._cond:
            if job.seq != self._load_seq:
                return
            self._active_image = image
        self.encoded.emit(EncodedEvent(
            job.document_id, key, time.time() - t1, cached))

    def _decode(self, job):
        t0 = time.time()
        with self._cond:
            if job.document_id != self._active_doc or self._active_image is None:
                raise RuntimeError("图像编码尚未完成，或当前图片已切换")
            image = self._active_image
        self.phase.emit("按提示生成蒙版…")
        result = self._ensure_engine().decode(image, job.snapshot.as_sam_prompt())
        if self._is_stale_decode(job):
            return
        self.decoded.emit(DecodedEvent(
            job.document_id, job.revision, result, time.time() - t0))

    def _effect_job_is_stale(self, job):
        with self._cond:
            return (
                job.seq != self._effect_seq or
                job.document_id != self._active_doc or
                self._active_image is None
            )

    def _requeue_effect(self, job):
        with self._cond:
            if (job.seq == self._effect_seq and
                    job.document_id == self._active_doc and
                    self._active_image is not None and
                    self._pending_effect is None):
                self._pending_effect = job
                return True
        return False

    def _advance_effect(self, job):
        """每次只执行一个 decoder 调用，之后重新排队让高优任务先行。"""
        if self._effect_job_is_stale(job):
            return
        plan_count = len(job.plans)
        if job.plan_index >= plan_count:
            return
        plan = job.plans[job.plan_index]
        if not plan.steps:
            raise RuntimeError("效应计划没有 decoder 步骤")
        if job.step_index >= len(plan.steps):
            raise RuntimeError("效应计划步骤状态异常")

        if job.step_index == 0:
            job.last_decoded = None
            job.plan_decodes = 0
            job.plan_cache_hits = 0
            job.plan_started_at = time.time()
            self.effect_progress.emit(EffectProgressEvent(
                job.document_id, job.revision, job.request_id,
                job.plan_index, plan_count,
                "正在分析第 %d/%d 项：%s" % (
                    job.plan_index + 1, plan_count, plan.label)))

        step = plan.steps[job.step_index]
        if step.points:
            points_xy = np.array(
                [[float(point.x), float(point.y)] for point in step.points],
                np.float32)
            labels = np.array([int(point.label) for point in step.points], np.int64)
        else:
            points_xy = np.zeros((0, 2), np.float32)
            labels = np.zeros((0,), np.int64)
        mask_input = None
        if job.step_index > 0:
            if job.last_decoded is None:
                raise RuntimeError("效应重放缺少上一步解码结果")
            mask_input = job.last_decoded.next_mask_input(step.input_candidate)
        prompt = SamPrompt(plan.box_xyxy, points_xy, labels, mask_input)
        with self._cond:
            if self._effect_job_is_stale(job):
                return
            image = self._active_image
            cache_key = self._active_key
        fingerprint = prompt_fingerprint(
            cache_key, plan.box_xyxy, step.points, mask_input)
        decoded = self._effect_cache_get(self._effect_cache, fingerprint)
        if decoded is None:
            result = self._ensure_engine().decode(image, prompt)
            decoded = _DecodedEffect(
                selected=int(result.selected),
                low_res_logits=np.array(result.low_res_logits, np.float32, copy=True),
                mask=np.array(result.mask_logits(result.selected) > 0.0, bool, copy=True),
            )
            self._effect_cache_put(fingerprint, decoded)
            job.plan_decodes += 1
            job.total_decodes += 1
        else:
            job.plan_cache_hits += 1
            job.total_cache_hits += 1
        if self._effect_job_is_stale(job):
            return
        job.last_decoded = decoded
        job.step_index += 1

        if job.step_index < len(plan.steps):
            self._requeue_effect(job)
            return

        effect_result = make_effect_result(
            plan=plan,
            current_mask=job.current_mask,
            current_candidate=job.current_candidate,
            counterfactual_mask=decoded.mask,
            counterfactual_candidate=decoded.selected,
            decodes=job.plan_decodes,
            cache_hits=job.plan_cache_hits,
            seconds=time.time() - job.plan_started_at,
        )
        completed = job.plan_index + 1
        self.effect_ready.emit(EffectItemEvent(
            job.document_id, job.revision, job.request_id, effect_result,
            completed, plan_count, effect_result.seconds))
        job.plan_index += 1
        job.step_index = 0
        job.last_decoded = None

        if job.plan_index < plan_count:
            self._requeue_effect(job)
            return
        self.effects_finished.emit(EffectsFinishedEvent(
            job.document_id, job.revision, job.request_id,
            plan_count, plan_count, time.time() - job.started_at))

    def _export(self, job):
        t0 = time.time()
        with self._cond:
            if job.document_id != self._active_doc or self._active_key != job.cache_key:
                raise RuntimeError("当前图片已切换，拒绝导出旧任务")
        try:
            st = os.stat(job.path)
            if st.st_size != job.cache_key.size or int(
                    getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))) != job.cache_key.mtime_ns:
                raise RuntimeError("源文件在打开后发生了变化，请重新打开图片")
        except FileNotFoundError as exc:
            raise RuntimeError("源文件不存在，无法导出") from exc

        self.phase.emit("读取原图并精修边缘…")
        rgb, exif, icc, src_alpha = gm.load_image(job.path)
        self.phase.emit("精修边缘并写出结果…")
        result = pipeline.export_interactive(
            job.path, job.out_dir, rgb, exif, icc, src_alpha,
            job.selected_logits, work_size=job.work_size, use_matte=job.use_matte)
        self.phase.emit("写出完成")
        self.exported.emit(ExportedEvent(
            job.document_id, job.revision, result, time.time() - t0))

    def run(self):
        try:
            self._ensure_engine()
        except Exception as exc:
            self._emit_error(exc, kind="model")
            return
        while True:
            with self._cond:
                while (not self._stopping and
                       self._pending_load is None and self._pending_decode is None and
                       self._pending_export is None and self._pending_effect is None):
                    self._cond.wait()
                if self._stopping:
                    return
                if self._pending_load is not None:
                    job, self._pending_load = self._pending_load, None
                    kind = "load"
                elif self._pending_export is not None:
                    job, self._pending_export = self._pending_export, None
                    kind = "export"
                elif self._pending_decode is not None:
                    job, self._pending_decode = self._pending_decode, None
                    kind = "decode"
                elif self._pending_effect is not None:
                    job, self._pending_effect = self._pending_effect, None
                    kind = "effect"
                else:
                    continue
            try:
                if kind == "load":
                    self._load(job)
                elif kind == "export":
                    self._export(job)
                elif kind == "effect":
                    self._advance_effect(job)
                else:
                    self._decode(job)
            except MemoryError:
                self._emit_error(
                    RuntimeError("内存不足，请关闭其他程序后重试"),
                    getattr(job, "document_id", None), getattr(job, "revision", None),
                    kind, getattr(job, "request_id", None))
            except Exception as exc:
                self._emit_error(
                    exc, getattr(job, "document_id", None),
                    getattr(job, "revision", None), kind,
                    getattr(job, "request_id", None))
