# -*- coding: utf-8 -*-
"""交互抠图主窗口。"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QWhatsThis,
    QStatusBar,
    QToolBar,
    QToolButton,
)

import generate_masks as gm
from . import pipeline
from .canvas import (
    MODE_BOX,
    MODE_NEGATIVE,
    MODE_PAN,
    MODE_POSITIVE,
    MODE_COLOR_SAMPLE,
    ImageCanvas,
    numpy_rgb_to_qimage,
)
from .sam_onnx import SamModelError, sample_manifest_hash
from .session import PromptHistory, PromptPoint, PromptSnapshot, copy_mask_input
from .color_assist import suggest_points
from .candidate_workbench import CandidateWorkbench
from .effect_dialog import ActiveEffectsDialog
from .effects import build_box_only_plan, build_point_loo_plan
from .profile import FeedbackRecord, ObjectProfile, load_profile, save_profile
from .worker import InteractiveWorker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output_interactive"


class MainWindow(QMainWindow):
    def __init__(self, models_dir=None, out_dir=None, worker_factory=None):
        super().__init__()
        self.models_dir = Path(models_dir or (PROJECT_ROOT / "models")).resolve()
        self.out_dir = Path(out_dir or DEFAULT_OUTPUT).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.setWindowTitle("交互抠图 — 框选目标，正负点修正")
        self.resize(1280, 860)

        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        self._folder = None
        self._files = []
        self._index = -1
        self._document_id = 0
        self._revision = 0
        self._prompt_revision = 0
        self._result_prompt_revision = -1
        self._result = None
        self._candidate = 0
        self._dirty = False
        self._loading = False
        self._encoding = False
        self._decoding = False
        self._exporting = False
        self._model_error = False
        self._model_error_shown = False
        self._pending_nav = None
        self._shutting_down = False
        self._show_original = False
        self._canvas_initialized = False
        self._path = None
        self._original_hw = (0, 0)
        self._preview_rgb = None
        self._cache_key = None
        self._history = PromptHistory(max_steps=20)
        self._color_samples = []
        self._last_color_samples = []
        self._color_assist_active = False
        self._last_phase = ""
        self._profile = None
        self._profile_path = None
        self._candidate_draft = {}
        self._effects_dialog = None
        self._effect_request_id = 0
        self._last_export_summary = None

        self._build_actions()
        self._build_status()
        self._update_profile_indicator()

        self.worker = (worker_factory or InteractiveWorker)(
            self.models_dir, self.out_dir,
            threads=max(2, min(8, (os.cpu_count() or 4) // 2)),
            parent=self,
        )
        self.worker.loaded.connect(self._on_loaded)
        self.worker.encoded.connect(self._on_encoded)
        self.worker.decoded.connect(self._on_decoded)
        self.worker.exported.connect(self._on_exported)
        self.worker.effect_ready.connect(self._on_effect_ready)
        self.worker.effect_progress.connect(self._on_effect_progress)
        self.worker.effects_finished.connect(self._on_effects_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.phase.connect(self._on_phase)
        self.worker.start()

        try:
            sample_manifest_hash(self.models_dir)
        except SamModelError as exc:
            self._model_error = True
            message = str(exc)
            QTimer.singleShot(0, lambda msg=message: self._show_model_error(msg))
        self._update_actions()
        self._refresh_busy()
        self._render_selection()
        self._status("准备就绪")

    def _build_actions(self):
        tb = QToolBar("主要操作", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.toolbar = tb

        self.open_action = self._action("打开文件夹", self.open_folder, QKeySequence.Open)
        self.add_images_action = self._action("添加图片", self.add_images)
        self.output_action = self._action("输出目录", self.choose_output_dir)
        self.profile_action = self._action("工程包", self.manage_profile)
        self.sync_profile_action = self._action(
            "同步到工程包", self.sync_current_to_profile)
        self.candidates_action = self._action("候选工作台", self.open_candidate_workbench)
        self.effects_action = self._action("有效状态", self.show_active_effects)
        self.toolbar_profile_label = QLabel("当前工程包：未选择")
        self.toolbar_profile_label.setMinimumWidth(150)
        self.toolbar_profile_label.setToolTip("当前使用的对象工程包；未选择时不会记录候选反馈。")
        self.color_action = self._action(
            "颜色辅助", self.toggle_color_assist, checkable=True)
        self.prev_action = self._action("上一张", self.previous_image, QKeySequence(Qt.Key_PageUp))
        self.next_action = self._action("下一张", self.next_image, QKeySequence(Qt.Key_PageDown))
        file_button = QToolButton(self)
        file_button.setText("文件")
        file_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        file_menu = QMenu(file_button)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.add_images_action)
        file_button.setMenu(file_menu)
        file_button.setPopupMode(QToolButton.InstantPopup)
        tb.addWidget(file_button)
        tb.addAction(self.output_action)
        tb.addAction(self.profile_action)
        tb.addAction(self.sync_profile_action)
        tb.addWidget(self.toolbar_profile_label)
        tb.addAction(self.candidates_action)
        tb.addAction(self.effects_action)
        tb.addAction(self.color_action)
        tb.addAction(self.prev_action)
        tb.addAction(self.next_action)
        tb.addSeparator()

        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.box_action = self._mode_action("框选", "B", MODE_BOX, True)
        self.pos_action = self._mode_action("正点", "P", MODE_POSITIVE)
        self.neg_action = self._mode_action("负点", "N", MODE_NEGATIVE)
        self.pan_action = self._mode_action("平移", "H", MODE_PAN)
        for action in (self.box_action, self.pos_action, self.neg_action, self.pan_action):
            tb.addAction(action)
        tb.addSeparator()

        self.undo_action = self._action("撤销", self.undo, QKeySequence.Undo)
        self.reset_action = self._action("重置", self.reset_prompt, QKeySequence(Qt.Key_R))
        self.fit_action = self._action("适应窗口", self.canvas.fit_to_view, QKeySequence(Qt.Key_F))
        tb.addAction(self.undo_action)
        tb.addAction(self.reset_action)
        tb.addAction(self.fit_action)
        tb.addSeparator()

        self.prev_candidate_action = self._action(
            "上一个候选", lambda: self.step_candidate(-1), QKeySequence(Qt.Key_BracketLeft))
        self.next_candidate_action = self._action(
            "下一个候选", lambda: self.step_candidate(1), QKeySequence(Qt.Key_BracketRight))
        tb.addAction(self.prev_candidate_action)
        tb.addAction(self.next_candidate_action)

        self.original_action = self._action("显示原图", self.toggle_original, QKeySequence(Qt.Key_O),
                                           checkable=True)
        tb.addAction(self.original_action)
        tb.addSeparator()
        self.export_action = self._action("导出结果", self.export_current, QKeySequence.Save)
        tb.addAction(self.export_action)

        tb.addSeparator()
        self.help_action = self._action("操作说明", self.show_help, QKeySequence.HelpContents)
        tb.addAction(self.help_action)
        self.whats_this_action = self._action(
            "上下文帮助", lambda _checked=False: QWhatsThis.enterWhatsThisMode(),
            QKeySequence("Shift+F1"))
        tb.addAction(self.whats_this_action)
        self._configure_action_help()

        self.canvas.boxSelected.connect(self.on_box)
        self.canvas.boxAdjusted.connect(self.on_box_adjusted)
        self.canvas.pointSelected.connect(self.on_point)
        self.canvas.colorSampled.connect(self.on_color_sample)
        self.canvas.zoomChanged.connect(self._update_zoom_label)

    def _action(self, text, slot, shortcut=None, checkable=False):
        action = QAction(text, self)
        action.setCheckable(checkable)
        if shortcut is not None:
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.WindowShortcut)
        action.triggered.connect(slot)
        action.setToolTip(text + (("  " + action.shortcut().toString()) if shortcut else ""))
        return action

    def _configure_action_help(self):
        help_text = {
            self.open_action: "选择一个图片文件夹并替换当前图片列表。",
            self.add_images_action: "选择一张或多张图片追加到当前列表，重复文件会忽略。",
            self.output_action: "选择透明 PNG 和蒙版的保存目录，不会修改原图。",
            self.profile_action: "创建或导入对象工程包，保存颜色、偏好和候选反馈。",
            self.sync_profile_action: "手动把当前图片已确认的框选、正负点、颜色样本和候选选择保存为工程包示例。",
            self.candidates_action: "一次比较当前 SAM 返回的多个候选，并记录评分、标签和备注；"
                                      "每个候选都可打开细节预览。",
            self.effects_action: "打开只读状态与反事实分析面板；查看框、点、候选、"
                                 "颜色辅助、反馈、工程包和导出精修的真实作用。",
            self.color_action: "第一次点击进入颜色采样；在目标上采样后再次点击，"
                              "只在当前框内生成最多 5 个建议正点。",
            self.prev_action: "切换到上一张图片；有未导出修改时会先询问。",
            self.next_action: "切换到下一张图片；有未导出修改时会先询问。",
            self.box_action: "拖出目标框生成初始蒙版；已有框的边线和角点可拖动调整。",
            self.pos_action: "点击漏选区域，将其加入前景。",
            self.neg_action: "点击误选区域，将其排除；右键也可添加负点。",
            self.pan_action: "拖动画布查看大图；按住空格也可以临时平移。",
            self.undo_action: "撤销最近一次框选、框调整、正点、负点或候选切换。",
            self.reset_action: "清除当前图片的框和所有正负点，重新开始。",
            self.fit_action: "将整张图片缩放到画布可见区域。",
            self.prev_candidate_action: "查看 SAM 返回的上一个候选蒙版。",
            self.next_candidate_action: "查看 SAM 返回的下一个候选蒙版。",
            self.original_action: "切换原图和当前抠图预览，用于检查边缘和误选区域。",
            self.export_action: "按当前候选蒙版进行原尺寸精修，并写出透明 PNG 和蒙版。",
            self.help_action: "打开完整操作手册。",
            self.whats_this_action: "按 Shift+F1 后点击按钮，可查看该控件的上下文帮助。",
        }
        for action, description in help_text.items():
            shortcut = action.shortcut().toString()
            title = action.text() + (("  [" + shortcut + "]") if shortcut else "")
            action.setToolTip(title + "\n" + description)
            action.setStatusTip(description)
            action.setWhatsThis(description)
            widget = self.toolbar.widgetForAction(action)
            if widget is not None:
                widget.setWhatsThis(description)

    def _mode_action(self, text, key, mode, checked=False):
        action = QAction(text, self)
        action.setCheckable(True)
        action.setChecked(checked)
        action.setShortcut(QKeySequence(key))
        action.setShortcutContext(Qt.WindowShortcut)
        action.setToolTip("%s  %s" % (text, key))
        action.triggered.connect(lambda _checked, m=mode: self.set_mode(m))
        self.mode_group.addAction(action)
        return action

    def _build_status(self):
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self.status_label = QLabel()
        self.status_label.setMinimumWidth(280)
        self.status_label.setWordWrap(True)
        bar.addWidget(self.status_label, 1)
        self.profile_label = QLabel()
        self.profile_label.setMinimumWidth(150)
        self.profile_label.setToolTip("当前使用的对象工程包；未选择时不会记录候选反馈。")
        bar.addPermanentWidget(self.profile_label)
        self.candidate_label = QLabel("")
        self.dims_label = QLabel("")
        self.zoom_label = QLabel("")
        for label in (self.candidate_label, self.dims_label, self.zoom_label):
            label.setMinimumWidth(70)
            bar.addPermanentWidget(label)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(110)
        self.progress.setVisible(False)
        bar.addPermanentWidget(self.progress)

    def _update_profile_indicator(self):
        if self._profile is None:
            self.profile_label.setText("工程包：未选择")
            self.profile_label.setStyleSheet("color: #777;")
            self.toolbar_profile_label.setText("当前工程包：未选择")
            self.toolbar_profile_label.setStyleSheet("color: #777;")
        else:
            self.profile_label.setText("工程包：%s" % self._profile.name)
            self.profile_label.setStyleSheet("color: #1769aa; font-weight: bold;")
            self.toolbar_profile_label.setText("当前工程包：%s" % self._profile.name)
            self.toolbar_profile_label.setStyleSheet("color: #1769aa; font-weight: bold;")

    def _next_step_hint(self):
        if self._model_error:
            return "修复 models 中的 SAM 制品后重新启动。"
        if not self._files:
            return "用“打开文件夹”或“添加图片”载入图片。"
        if self._loading or self._encoding:
            return "等待图像分析完成。"
        if not self._history.current.has_prompt:
            return "按 B 框住目标；同类图片可先选择或创建工程包。"
        if self._decoding:
            return "等待蒙版更新，然后检查候选结果。"
        if not self._result_valid():
            return "等待蒙版生成，或用正点/负点修正提示。"
        if self._dirty and self._profile is not None:
            return "可打开候选工作台记录偏好，满意后按 Ctrl+S 导出。"
        if self._dirty:
            return "可用候选工作台比较结果，或按 Ctrl+S 导出。"
        return "可用“下一张”继续处理，或打开候选工作台积累工程包反馈。"

    def _show_model_error(self, message):
        if self._model_error_shown:
            return
        self._model_error_shown = True
        QMessageBox.critical(
            self, "SAM 模型不可用",
            "交互功能需要 SAM ViT-B ONNX 制品。\n\n%s\n\n"
            "请把 sam_vit_b_encoder.onnx、sam_vit_b_decoder.onnx、"
            "sam_vit_b_manifest.json 放入：\n%s\n\n"
            "生成方法见使用说明。" % (message, self.models_dir))

    # ------------------------------------------------------------------
    # 图片导航
    # ------------------------------------------------------------------
    def open_path(self, path):
        if self._model_error:
            QMessageBox.warning(self, "SAM 模型不可用",
                                "请先修复 models 目录中的 SAM 模型。")
            return
        path = Path(path)
        if path.is_dir():
            files = gm.collect_files(str(path), False)
            if not files:
                QMessageBox.information(self, "没有图片", "该文件夹里没有支持的图片。")
                return
            cb = lambda: self._set_folder(path, files)
            self._guard_navigation(cb)
        elif path.is_file():
            files = [str(path)]
            cb = lambda: self._set_folder(path.parent, files)
            self._guard_navigation(cb)

    def open_folder(self):
        start = str(self._folder or (PROJECT_ROOT / "input"))
        folder = QFileDialog.getExistingDirectory(self, "打开图片文件夹", start)
        if not folder:
            return
        files = gm.collect_files(folder, False)
        if not files:
            QMessageBox.information(self, "没有图片", "该文件夹里没有支持的图片。")
            return
        self._guard_navigation(lambda: self._set_folder(Path(folder), files))

    def add_images(self):
        if self._model_error:
            QMessageBox.warning(self, "SAM 模型不可用",
                                "请先修复 models 目录中的 SAM 模型。")
            return
        start = str(self._folder or (PROJECT_ROOT / "input"))
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self, "添加图片", start,
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp "
            "*.dng *.cr2 *.nef *.arw *.raf *.rw2 *.orf *.pef);;"
            "所有文件 (*.*)",
        )
        if not files:
            return
        self._guard_navigation(lambda: self._append_files(files))

    def choose_output_dir(self):
        start = str(self.out_dir)
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if not folder:
            return
        try:
            output = Path(folder).resolve()
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "输出目录不可用",
                                "无法使用该目录：%s" % exc)
            return
        self.out_dir = output
        self._refresh_effect_dialog()
        self._status("输出目录已设置：%s" % output)

    def manage_profile(self):
        choice = QMessageBox(self)
        choice.setWindowTitle("对象工程包")
        choice.setText("工程包用于保存对象偏好、颜色样本和候选反馈。")
        open_button = choice.addButton("导入工程包", QMessageBox.AcceptRole)
        new_button = choice.addButton("新建工程包", QMessageBox.ActionRole)
        cancel_button = choice.addButton("取消", QMessageBox.RejectRole)
        choice.setDefaultButton(open_button)
        choice.exec()
        clicked = choice.clickedButton()
        if clicked is open_button:
            path, _ = QFileDialog.getOpenFileName(
                self, "导入对象工程包", str(self._profile_path or PROJECT_ROOT),
                "对象工程包 (*.json);;所有文件 (*.*)")
            if not path:
                return
            try:
                self._profile = load_profile(path)
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.warning(self, "工程包无效", str(exc))
                return
            self._profile_path = Path(path).resolve()
            self._candidate_draft = {}
            self._update_profile_indicator()
            self._update_actions()
            self._refresh_effect_dialog()
            self._status("已导入工程包：%s" % self._profile.name)
            return
        if clicked is not new_button:
            return
        name, ok = QInputDialog.getText(self, "新建工程包", "对象名称：")
        if not ok or not name.strip():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存对象工程包", str(PROJECT_ROOT / (name.strip() + ".json")),
            "对象工程包 (*.json)")
        if not path:
            return
        self._profile = ObjectProfile(name.strip())
        self._profile_path = Path(path).resolve()
        self._candidate_draft = {}
        try:
            save_profile(self._profile, self._profile_path)
        except OSError as exc:
            self._profile = None
            self._profile_path = None
            QMessageBox.warning(self, "工程包保存失败", str(exc))
            return
        self._update_profile_indicator()
        self._update_actions()
        self._refresh_effect_dialog()
        self._status("已创建工程包：%s" % self._profile.name)

    def sync_current_to_profile(self):
        if self._profile is None or self._profile_path is None:
            self._status("请先通过“工程包”导入或创建工程包。")
            return
        if not self._path:
            self._status("请先载入图片。")
            return
        snapshot = self._history.current
        if not snapshot.has_prompt:
            self._status("请先框选目标或添加正负点，再同步到工程包。")
            return
        default_name = Path(self._path).stem
        name, ok = QInputDialog.getText(
            self, "同步当前操作", "示例名称：", text=default_name)
        if not ok or not name.strip():
            return
        points = [
            {"x": float(point.x), "y": float(point.y),
             "label": int(point.label), "source": point.source}
            for point in snapshot.points
        ]
        colors = [
            [int(value) for value in np.asarray(sample).reshape(-1)[:3]]
            for sample in (self._last_color_samples or self._color_samples)
        ]
        example = {
            "name": name.strip(),
            "image": str(Path(self._path).resolve()),
            "box_xyxy": list(snapshot.box_xyxy) if snapshot.box_xyxy else None,
            "points": points,
            "color_samples_rgb": colors,
            "selected_candidate": int(self._candidate),
            "feedback": dict(self._candidate_draft),
        }
        replaced = False
        for index, existing in enumerate(self._profile.examples):
            if (existing.get("name") == example["name"] and
                    existing.get("image") == example["image"]):
                self._profile.examples[index] = example
                replaced = True
                break
        if not replaced:
            self._profile.examples.append(example)
        if colors:
            self._profile.color_samples_rgb = colors
        try:
            save_profile(self._profile, self._profile_path)
        except OSError as exc:
            QMessageBox.warning(self, "工程包保存失败", str(exc))
            return
        self._refresh_effect_dialog()
        self._status("%s工程包示例：%s" %
                     ("已更新" if replaced else "已添加", name.strip()))

    def open_candidate_workbench(self):
        if not self._result_valid() or self._preview_rgb is None:
            self._status("请先框选目标并等待蒙版生成。")
            return
        candidates = []
        for index in range(self._result.count):
            alpha = pipeline.preview_alpha(
                self._result.mask_logits(index), self._preview_rgb.shape[:2])
            preview = pipeline.compose_cutout_preview(
                self._preview_rgb, alpha, False)
            candidates.append({
                "title": "候选 %d" % (index + 1),
                "score": self._result.score(index),
                "image": numpy_rgb_to_qimage(preview),
                "original": numpy_rgb_to_qimage(self._preview_rgb),
                "index": index,
            })
        dialog = CandidateWorkbench(
            candidates, os.path.basename(self._path), self._record_candidate_feedback,
            initial=self._candidate_draft, parent=self)
        dialog.exec()

    def show_active_effects(self):
        if self._effects_dialog is None:
            self._effects_dialog = ActiveEffectsDialog(self, self)
        self._effects_dialog.refresh()
        self._effects_dialog.show()
        self._effects_dialog.raise_()
        self._effects_dialog.activateWindow()

    def _effect_inputs(self):
        if not self._result_valid():
            raise RuntimeError("当前还没有有效蒙版，请先框选目标并等待生成")
        try:
            mask = np.asarray(
                self._result.mask_logits(self._candidate) > 0.0, bool)
        except (IndexError, AttributeError, TypeError) as exc:
            raise RuntimeError("当前候选蒙版不可用：%s" % exc) from exc
        return mask, self._history.current_branch()

    def _start_effect_plans(self, plans):
        plans = tuple(plans)
        if not plans:
            raise ValueError("没有可分析的点或框")
        current_mask, _states = self._effect_inputs()
        self._effect_request_id += 1
        request_id = self._effect_request_id
        if self._effects_dialog is not None:
            self._effects_dialog.on_analysis_started(request_id, plans)
        self.worker.request_effects(
            self._document_id, self._prompt_revision, request_id,
            current_mask, self._candidate, plans)
        self._status("已开始反事实分析：%d 项；正式编辑会被优先处理" %
                     len(plans))
        return request_id

    def request_point_effect(self, point_index):
        _mask, states = self._effect_inputs()
        if not states:
            raise RuntimeError("当前提示历史为空")
        plan = build_point_loo_plan(states, int(point_index))
        return self._start_effect_plans((plan,))

    def request_all_point_effects(self):
        _mask, states = self._effect_inputs()
        if not states or not states[-1].points:
            raise ValueError("当前没有可分析的点")
        plans = [
            build_point_loo_plan(states, index)
            for index in range(len(states[-1].points))
        ]
        return self._start_effect_plans(plans)

    def request_box_only_effect(self):
        _mask, states = self._effect_inputs()
        plan = build_box_only_plan(states)
        return self._start_effect_plans((plan,))

    def cancel_effect_analysis(self):
        self._effect_request_id += 1
        self.worker.cancel_effects()
        if self._effects_dialog is not None:
            self._effects_dialog.on_analysis_cancelled("分析已取消")
        self._status("已取消反事实分析")

    def _invalidate_effect_analysis(self):
        self._effect_request_id += 1
        if self._effects_dialog is not None:
            self.worker.cancel_effects()
            self._effects_dialog.on_state_changed()

    def _refresh_effect_dialog(self):
        if self._effects_dialog is not None and self._effects_dialog.isVisible():
            self._effects_dialog.refresh()

    def _on_effect_progress(self, event):
        if event.document_id != self._document_id:
            return
        if event.revision != self._prompt_revision:
            return
        if event.request_id != self._effect_request_id:
            return
        if self._effects_dialog is not None:
            self._effects_dialog.on_effect_progress(event)

    def _on_effect_ready(self, event):
        if event.document_id != self._document_id:
            return
        if event.revision != self._prompt_revision:
            return
        if event.request_id != self._effect_request_id:
            return
        if self._effects_dialog is not None:
            self._effects_dialog.on_effect_ready(event)

    def _on_effects_finished(self, event):
        if event.document_id != self._document_id:
            return
        if event.revision != self._prompt_revision:
            return
        if event.request_id != self._effect_request_id:
            return
        if self._effects_dialog is not None:
            self._effects_dialog.on_effects_finished(event)
        self._status("反事实分析完成：%d 项，耗时 %.2fs" %
                     (event.total, event.seconds))

    def _record_candidate_feedback(self, draft, close=False, *legacy):
        if not isinstance(draft, dict):
            draft = {
                "selected": int(draft),
                "feedback": {str(int(draft)): {
                    "rating": close,
                    "tags": list(legacy[0]) if legacy else [],
                    "note": str(legacy[1]) if len(legacy) > 1 else "",
                }},
            }
            close = False
        self._candidate_draft = dict(draft)
        self._refresh_effect_dialog()
        if self._profile is None or self._profile_path is None:
            self._status("反馈未保存：请先通过“工程包”创建或导入工程包。")
            return
        selected = int(draft.get("selected", 0))
        self._candidate = selected
        self._history.set_candidate(selected)
        candidates = [
            {
                "index": i,
                "sam_score": float(self._result.score(i)),
                "feedback": draft.get("feedback", {}).get(str(i), {}),
            }
            for i in range(self._result.count)
        ]
        record = FeedbackRecord(
            image=os.path.abspath(self._path),
            candidates=candidates,
            selected=selected,
            worst=draft.get("worst"),
            rating=draft.get("feedback", {}).get(str(selected), {}).get("rating"),
            tags=draft.get("feedback", {}).get(str(selected), {}).get("tags", []),
            note=draft.get("feedback", {}).get(str(selected), {}).get("note", ""),
        )
        self._profile.feedback = [
            item for item in self._profile.feedback if item.image != record.image]
        self._profile.add_feedback(record)
        self._profile.settings["last_selected_candidate"] = selected
        try:
            save_profile(self._profile, self._profile_path)
        except OSError as exc:
            QMessageBox.warning(self, "反馈保存失败", str(exc))
            return
        self._revision += 1
        self._dirty = True
        self._invalidate_effect_analysis()
        self._render_all()
        self._update_actions()
        self._status("已记录候选反馈，选择了候选 %d" % (selected + 1))

    def _set_folder(self, folder, files):
        self._folder = Path(folder)
        self._files = self._unique_files(files)
        self._index = 0
        self._load_current()

    @staticmethod
    def _unique_files(files):
        result = []
        seen = set()
        for path in files:
            absolute = os.path.abspath(str(path))
            key = os.path.normcase(absolute)
            if key not in seen:
                seen.add(key)
                result.append(absolute)
        return result

    def _append_files(self, files):
        added = []
        existing = {os.path.normcase(os.path.abspath(path)) for path in self._files}
        for path in files:
            absolute = os.path.abspath(str(path))
            key = os.path.normcase(absolute)
            if key not in existing and os.path.isfile(absolute):
                existing.add(key)
                added.append(absolute)
        if not added:
            self._status("没有添加新的图片")
            return
        was_empty = not self._files
        self._files.extend(added)
        if self._folder is None:
            self._folder = Path(added[0]).parent
        if was_empty:
            self._index = 0
            self._load_current()
        else:
            self._update_actions()
            self._status("已添加 %d 张图片，当前共 %d 张" %
                         (len(added), len(self._files)))

    def previous_image(self):
        if self._index <= 0:
            return
        self._guard_navigation(lambda: self._show_index(self._index - 1))

    def next_image(self):
        if self._index < 0 or self._index + 1 >= len(self._files):
            return
        self._guard_navigation(lambda: self._show_index(self._index + 1))

    def _show_index(self, index):
        if 0 <= index < len(self._files):
            self._index = int(index)
            self._load_current()

    def _load_current(self):
        if not (0 <= self._index < len(self._files)):
            return
        path = self._files[self._index]
        self._document_id += 1
        self._revision = 0
        self._prompt_revision = 0
        self._result_prompt_revision = -1
        self._result = None
        self._candidate = 0
        self._dirty = False
        self._path = path
        self._original_hw = (0, 0)
        self._preview_rgb = None
        self._cache_key = None
        self._history = PromptHistory(max_steps=20)
        self._candidate_draft = {}
        self._color_samples = []
        self._last_color_samples = []
        self._color_assist_active = False
        self._last_export_summary = None
        self.color_action.setChecked(False)
        self._canvas_initialized = False
        self._invalidate_effect_analysis()
        self._loading = True
        self._encoding = False
        self._decoding = False
        self._exporting = False
        self._last_phase = "读取图片…"
        self.canvas.reset_document()
        self.canvas.set_mode(MODE_BOX)
        self.box_action.setChecked(True)
        self.setWindowTitle("交互抠图 — %s (%d/%d)" %
                            (os.path.basename(path), self._index + 1, len(self._files)))
        self._status("读取图片…")
        self._refresh_busy()
        self._update_actions()
        self.worker.request_load(self._document_id, path)

    # ------------------------------------------------------------------
    # 状态与渲染
    # ------------------------------------------------------------------
    def _status(self, text):
        self.status_label.setText("%s  | 下一步：%s" % (text, self._next_step_hint()))

    def _refresh_busy(self):
        busy = self._loading or self._encoding or self._decoding or self._exporting
        self.progress.setVisible(busy)
        if busy and self._last_phase:
            self._status(self._last_phase)

    def _on_phase(self, text):
        self._last_phase = str(text)
        if self._loading or self._encoding or self._decoding or self._exporting:
            self._status(self._last_phase)

    def _update_zoom_label(self):
        zoom = self.canvas.transform().m11() * 100.0
        self.zoom_label.setText("%.0f%%" % zoom)

    def _update_dimensions(self):
        if not self._original_hw or not self._original_hw[0]:
            self.dims_label.setText("")
        else:
            self.dims_label.setText("%d × %d" % (
                self._original_hw[1], self._original_hw[0]))

    def _result_valid(self):
        return (self._result is not None and
                self._result_prompt_revision == self._prompt_revision)

    def _alpha_for_preview(self):
        if not self._result_valid() or self._preview_rgb is None:
            return None
        try:
            logits = self._result.mask_logits(self._candidate)
        except (IndexError, AttributeError):
            return None
        return pipeline.preview_alpha(logits, self._preview_rgb.shape[:2])

    def _render_image(self):
        if self._preview_rgb is None:
            return
        alpha = self._alpha_for_preview()
        composed = pipeline.compose_cutout_preview(
            self._preview_rgb, alpha, self._show_original)
        qimg = numpy_rgb_to_qimage(composed)
        if not self._canvas_initialized:
            self.canvas.set_image(qimg, self._original_hw)
            self._canvas_initialized = True
        else:
            self.canvas.update_image(qimg)
        self._update_zoom_label()

    def _render_selection(self):
        snap = self._history.current
        self.canvas.set_selection(
            box_xyxy=snap.box_xyxy,
            points=snap.points,
            active=not self._result_valid() and snap.has_prompt,
        )

    def _render_all(self):
        self._render_image()
        self._render_selection()

    def _update_candidate_label(self):
        if not self._result_valid():
            self.candidate_label.setText("")
            return
        try:
            self.candidate_label.setText(
                "候选 %d/%d · %.3f" %
                (self._candidate + 1, self._result.count, self._result.score(self._candidate)))
        except (IndexError, AttributeError):
            self.candidate_label.setText("")

    def _update_actions(self):
        has_doc = bool(self._path and self._original_hw[0]) and not self._model_error
        has_files = bool(self._files)
        snap = self._history.current
        valid = self._result_valid()
        can_export = valid and not self._exporting
        self.open_action.setEnabled(not self._model_error)
        self.add_images_action.setEnabled(not self._model_error)
        self.output_action.setEnabled(not self._exporting)
        self.profile_action.setEnabled(not self._exporting)
        self.sync_profile_action.setEnabled(
            has_doc and not self._exporting and self._profile is not None)
        self.candidates_action.setEnabled(valid and not self._exporting)
        self.color_action.setEnabled(has_doc and not self._exporting)
        self.prev_action.setEnabled(has_files and self._index > 0)
        self.next_action.setEnabled(has_files and 0 <= self._index < len(self._files) - 1)
        for action in (self.box_action, self.pos_action, self.neg_action, self.pan_action):
            action.setEnabled(has_doc)
        self.undo_action.setEnabled(has_doc and self._history.can_undo)
        self.reset_action.setEnabled(has_doc and snap.has_prompt)
        self.prev_candidate_action.setEnabled(valid and self._result.count > 1)
        self.next_candidate_action.setEnabled(valid and self._result.count > 1)
        self.original_action.setEnabled(has_doc)
        self.export_action.setEnabled(can_export)
        self.fit_action.setEnabled(has_doc)
        self._update_candidate_label()
        self._update_dimensions()

    # ------------------------------------------------------------------
    # 编辑操作
    # ------------------------------------------------------------------
    def set_mode(self, mode):
        if mode != MODE_COLOR_SAMPLE and self._color_assist_active:
            self._color_assist_active = False
            self._color_samples = []
            self._last_color_samples = []
            self.color_action.setChecked(False)
        self.canvas.set_mode(mode)
        table = {
            MODE_BOX: self.box_action,
            MODE_POSITIVE: self.pos_action,
            MODE_NEGATIVE: self.neg_action,
            MODE_PAN: self.pan_action,
            MODE_COLOR_SAMPLE: self.color_action,
        }
        action = table.get(mode)
        if action is not None:
            action.setChecked(True)

    def toggle_color_assist(self, checked=False):
        if not self._path or not self._history.current.box_xyxy:
            self._status("请先框选目标，再使用颜色辅助。")
            self.color_action.setChecked(False)
            return
        if self._color_assist_active:
            self._generate_color_points()
            return
        self._color_assist_active = True
        self.color_action.setChecked(True)
        self.set_mode(MODE_COLOR_SAMPLE)
        self._status("颜色辅助：在目标主体上点击采样，最多 8 个样本；再次点击按钮生成建议正点。")

    def on_color_sample(self, x, y):
        if not self._color_assist_active or self._preview_rgb is None:
            return
        if len(self._color_samples) >= 8:
            self._status("颜色样本已达到上限（8 个），请生成建议正点。")
            return
        h, w = self._preview_rgb.shape[:2]
        px = min(w - 1, max(0, int(round(x * w / self._original_hw[1]))))
        py = min(h - 1, max(0, int(round(y * h / self._original_hw[0]))))
        self._color_samples.append(self._preview_rgb[py, px].copy())
        self._last_color_samples = [sample.copy() for sample in self._color_samples]
        self._refresh_effect_dialog()
        self._status("已采样 %d/8 个颜色；请继续采样或再次点击“颜色辅助”生成建议正点。"
                     % len(self._color_samples))

    def _generate_color_points(self):
        snap = self._history.current

        if not self._color_samples or snap.box_xyxy is None or self._preview_rgb is None:
            self._status("请先在目标上采样至少一个颜色。")
            return
        points = suggest_points(
            self._preview_rgb, self._color_samples,
            tuple(v * self._preview_rgb.shape[1] / self._original_hw[1]
                  if i % 2 == 0 else
                  v * self._preview_rgb.shape[0] / self._original_hw[0]
                  for i, v in enumerate(snap.box_xyxy)),
            max_points=5,
            tolerance=0.22,
            min_score=0.62,
            min_spacing=max(12, min(self._original_hw) // 40),
        )
        existing = {(round(p.x), round(p.y)) for p in snap.points}
        added = 0
        for x, y, _score in points:
            ox = x * self._original_hw[1] / self._preview_rgb.shape[1]
            oy = y * self._original_hw[0] / self._preview_rgb.shape[0]
            if (round(ox), round(oy)) not in existing:
                self.on_point(ox, oy, 1, source="color")
                existing.add((round(ox), round(oy)))
                added += 1
        self._last_color_samples = [sample.copy() for sample in self._color_samples]
        self._color_samples = []
        self._color_assist_active = False
        self.color_action.setChecked(False)
        self.set_mode(MODE_BOX)
        self._status("颜色辅助已生成 %d 个建议正点；仍可用负点修正。" % added)

    def on_box(self, x0, y0, x1, y1):
        if not self._path:
            return
        self._revision += 1
        self._prompt_revision += 1
        self._candidate = 0
        snap = PromptSnapshot(
            box_xyxy=(float(x0), float(y0), float(x1), float(y1)),
            points=(),
            mask_input=None,
            candidate=0,
        )
        self._history.push(snap)
        self._dirty = True
        self._result = None
        self._invalidate_effect_analysis()
        self._sync_selection_status()
        self._request_decode()

    def on_box_adjusted(self, x0, y0, x1, y1):
        if not self._path:
            return
        current = self._history.current
        self._revision += 1
        self._prompt_revision += 1
        self._candidate = 0
        snap = PromptSnapshot(
            box_xyxy=(float(x0), float(y0), float(x1), float(y1)),
            points=current.points,
            mask_input=None,
            candidate=0,
        )
        self._history.push(snap)
        self._dirty = True
        self._result = None
        self._invalidate_effect_analysis()
        self._sync_selection_status()
        self._request_decode()

    def on_point(self, x, y, label, source="manual"):
        if not self._path:
            return
        snap = self._history.current
        if snap.box_xyxy is None:
            self._status("请先用“框选”圈住目标，再用正点或负点修正。")
            return
        mask_input = None
        if self._result_valid():
            try:
                mask_input = copy_mask_input(self._result.next_mask_input(self._candidate))
            except (IndexError, AttributeError):
                mask_input = None
        point = PromptPoint(float(x), float(y), int(label), source=source)
        new_snap = PromptSnapshot(
            box_xyxy=snap.box_xyxy,
            points=snap.points + (point,),
            mask_input=mask_input,
            candidate=self._candidate,
        )
        self._history.push(new_snap)
        self._revision += 1
        self._prompt_revision += 1
        self._dirty = True
        self._result = None
        self._invalidate_effect_analysis()
        self._sync_selection_status()
        self._request_decode()

    def undo(self):
        if not self._history.can_undo:
            return
        snap = self._history.undo()
        self._revision += 1
        self._prompt_revision += 1
        self._candidate = int(snap.candidate)
        self._result = None
        self._dirty = True
        self._invalidate_effect_analysis()
        self._sync_selection_status()
        if snap.has_prompt:
            self._request_decode()
        else:
            self._decoding = False
            self._status("提示已清空")
            self._refresh_busy()
            self._update_actions()

    def reset_prompt(self):
        if not self._history.current.has_prompt:
            return
        self._history.reset()
        self._revision += 1
        self._prompt_revision += 1
        self._candidate = 0
        self._result = None
        self._dirty = True
        self._decoding = False
        self._invalidate_effect_analysis()
        self._sync_selection_status()
        self._status("已重置提示")
        self._refresh_busy()
        self._update_actions()

    def step_candidate(self, delta):
        if not self._result_valid() or self._result.count <= 1:
            return
        self._candidate = (self._candidate + int(delta)) % self._result.count
        self._history.set_candidate(self._candidate)
        self._revision += 1
        self._dirty = True
        self._invalidate_effect_analysis()
        self._render_image()
        self._render_selection()
        self._update_candidate_label()
        self._update_actions()
        self._status("已切换到候选 %d/%d" % (self._candidate + 1, self._result.count))

    def toggle_original(self, checked):
        self._show_original = bool(checked)
        self._render_image()

    def _sync_selection_status(self):
        self._render_all()
        self._update_candidate_label()
        self._update_actions()

    def _request_decode(self):
        if not self._path or not self._history.current.has_prompt:
            return
        self._decoding = True
        self._last_phase = "按提示生成蒙版…"
        self._refresh_busy()
        self._update_actions()
        self.worker.request_decode(
            self._document_id, self._prompt_revision, self._history.current)

    # ------------------------------------------------------------------
    # worker 回调
    # ------------------------------------------------------------------
    def _on_loaded(self, event):
        if event.document_id != self._document_id:
            return
        self._loading = False
        self._encoding = True
        self._path = event.path
        self._original_hw = event.original_hw
        self._preview_rgb = event.preview_rgb
        self._cache_key = event.cache_key
        self._canvas_initialized = False
        self._render_all()
        self._last_phase = "正在分析图像，首次等待会久一些…"
        self._status(self._last_phase)
        self._refresh_busy()
        self._update_actions()

    def _on_encoded(self, event):
        if event.document_id != self._document_id:
            return
        self._encoding = False
        if not self._decoding and not self._exporting:
            self._status("模型就绪，拖框选择目标；框的边和角可拖动调整，正点加入、负点排除。")
        self._refresh_busy()
        self._update_actions()

    def _on_decoded(self, event):
        if event.document_id != self._document_id:
            return
        if event.revision != self._prompt_revision:
            return
        self._decoding = False
        self._result = event.result
        self._result_prompt_revision = event.revision
        self._candidate = min(max(0, self._result.selected),
                              max(0, self._result.count - 1))
        self._history.set_candidate(self._candidate)
        self._invalidate_effect_analysis()
        self._render_all()
        self._update_candidate_label()
        self._refresh_busy()
        self._update_actions()
        if not self._result.constraint_ok:
            self._status("提示可能存在冲突；请撤销最近的点或切换候选。")
        else:
            self._status("蒙版已更新 · 候选 %d/%d · IoU %.3f" %
                         (self._candidate + 1, self._result.count,
                          self._result.score(self._candidate)))

    def _on_exported(self, event):
        if event.document_id != self._document_id:
            return
        self._exporting = False
        same_revision = (event.revision == self._revision)
        if same_revision:
            self._dirty = False
            self._candidate_draft = {}
        self._refresh_busy()
        self._update_actions()
        result = event.result
        fg = result.get("fg_path") or ""
        mask = result.get("mask_path") or ""
        self._last_export_summary = {
            "fg_path": fg,
            "mask_path": mask,
            "seconds": float(event.seconds),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._refresh_effect_dialog()
        self._status("导出完成：%s" % fg)

        pending = self._pending_nav
        self._pending_nav = None
        if pending is not None and same_revision:
            QTimer.singleShot(0, pending)
            return
        if pending is not None and not same_revision:
            self._status("导出完成，但导出期间又有修改；已取消自动切换。")

        box = QMessageBox(self)
        box.setWindowTitle("导出完成")
        box.setIcon(QMessageBox.Information)
        box.setText("透明 PNG 与蒙版已保存。")
        box.setInformativeText("前景：%s\n蒙版：%s\n耗时：%.1fs" %
                               (fg, mask, event.seconds))
        open_btn = box.addButton("打开输出文件夹", QMessageBox.ActionRole)
        box.addButton("关闭", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is open_btn and fg:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(fg)))

    def _on_failed(self, error):
        kind = getattr(error, "kind", "error")
        doc_id = getattr(error, "document_id", None)
        revision = getattr(error, "revision", None)
        request_id = getattr(error, "request_id", None)
        if kind == "effect":
            if doc_id is not None and doc_id != self._document_id:
                return
            if revision is not None and revision != self._prompt_revision:
                return
            if request_id is not None and request_id != self._effect_request_id:
                return
            if self._effects_dialog is not None:
                self._effects_dialog.on_effect_failed(
                    request_id, error.message)
            self._status("反事实分析失败：%s" % error.message)
            return
        if kind == "model":
            self._model_error = True
            self._loading = self._encoding = self._decoding = self._exporting = False
            self._refresh_busy()
            self._update_actions()
            self._show_model_error(error.message)
            return
        if doc_id is not None and doc_id != self._document_id:
            return
        if kind == "load":
            self._loading = False
            self._encoding = False
        elif kind == "decode":
            if revision is not None and revision != self._prompt_revision:
                return
            self._decoding = False
        elif kind == "export":
            self._exporting = False
            self._pending_nav = None
        self._refresh_busy()
        self._update_actions()
        self._status("处理失败：%s" % error.message)
        if kind in ("load", "export", "model"):
            QMessageBox.warning(self, "操作失败", error.message)
        elif kind == "decode" and revision == self._prompt_revision:
            QMessageBox.warning(self, "无法生成蒙版", error.message)

    # ------------------------------------------------------------------
    # 导出与离开保护
    # ------------------------------------------------------------------
    def _can_export_now(self):
        return bool(self._result_valid() and self._cache_key is not None and
                    not self._exporting and not self._model_error)

    def export_current(self):
        if not self._can_export_now():
            self._status("当前还没有可导出的蒙版，请先框选目标。")
            return False
        try:
            logits = np.array(
                self._result.mask_logits(self._candidate), np.float32, copy=True)
        except (IndexError, AttributeError) as exc:
            QMessageBox.warning(self, "无法导出", "候选蒙版无效：%s" % exc)
            return False
        self._exporting = True
        self._last_phase = "正在精修并导出…"
        self._refresh_busy()
        self._update_actions()
        self.worker.request_export(
            self._document_id, self._revision, self._path, logits,
            self._cache_key, 1536, True, self.out_dir)
        return True

    def _guard_navigation(self, callback):
        if self._exporting:
            QMessageBox.information(
                self, "正在导出",
                "后台正在精修并写出当前结果，请等导出完成后再切换或关闭。")
            return False
        if not self._dirty:
            callback()
            return True
        box = QMessageBox(self)
        box.setWindowTitle("当前结果尚未导出")
        box.setIcon(QMessageBox.Warning)
        box.setText("当前图片有未导出的修改，切换前要导出吗？")
        box.setInformativeText("如果放弃，本次框选和点修正不会保留。")
        save_btn = None
        if self._can_export_now():
            save_btn = box.addButton("导出后继续", QMessageBox.AcceptRole)
        discard_btn = box.addButton("放弃修改", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(save_btn or cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_btn and save_btn is not None:
            self._pending_nav = callback
            if not self.export_current():
                self._pending_nav = None
            return True
        if clicked is discard_btn:
            self._dirty = False
            callback()
            return True
        return False

    def show_help(self):
        QMessageBox.information(
            self, "交互抠图操作",
            "打开文件夹后逐张处理。\n\n"
            "框选 B：拖框圈住目标，松手生成初始蒙版。\n"
            "橙色框的整条边线或角点都可拖动缩放；\n"
            "在框外重新拖框会替换整个选区。\n"
            "正点 P：在漏选处点击，加入目标。\n"
            "负点 N：在误选处点击（右键也可），排除该区域。\n"
            "平移 H / 空格：拖动画面。滚轮：以鼠标为中心缩放。\n\n"
            "撤销 Ctrl+Z；重置 R；适应窗口 F；导出 Ctrl+S。\n"
            "方括号切换候选蒙版；O 切换显示原图。\n"
            "有效状态：查看真实设置；点或框的反事实影响需按需计算，\n"
            "分析只读，不会改当前蒙版。\n"
            "悬停按钮查看简短提示；按 Shift+F1 后点击控件查看上下文帮助。\n\n"
            "导出会做原尺寸 matting、引导滤波和去污；\n"
            "输出到：%s\n"
            "打开文件夹：选择一个文件夹作为当前图片列表；\n"
            "添加图片：把单张或多张图片加入当前列表；\n"
            "输出目录：更改导出位置。" % self.out_dir)

    def _shutdown_worker(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        if self._effects_dialog is not None:
            self._effects_dialog.close()
            self._effects_dialog = None
        if hasattr(self.worker, "cancel_effects"):
            self.worker.cancel_effects()
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

    def closeEvent(self, event):
        if self._dirty and not self._shutting_down:
            event.ignore()
            self._guard_navigation(self.close)
            return
        self._shutdown_worker()
        event.accept()
