# -*- coding: utf-8 -*-
"""「当前有效操作与设置」非模态详情面板。"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import pipeline
from .canvas import numpy_rgb_to_qimage
from .effects import (
    build_diff_overlay,
    mask_to_preview,
    point_constraint_rate,
    point_display_name,
    point_local_metrics,
)

SECTION_TITLES = (
    "框选与正负点",
    "当前候选",
    "颜色辅助",
    "候选偏好反馈",
    "工程包",
    "导出精修",
)


def _fmt_int(value):
    return "{:,}".format(int(value))


def _fmt_ratio(value):
    if value is None:
        return "—"
    return "%.2f%%" % (float(value) * 100.0)


def _fmt_optional_float(value, digits=2):
    if value is None:
        return "—"
    return ("%.*f" % (digits, float(value)))


class ScaledImageLabel(QLabel):
    """保存原始 QPixmap，并在控件尺寸变化时按比例重绘。"""

    def __init__(self, placeholder="尚无结果", parent=None):
        super().__init__(parent)
        self._source = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QLabel { background: #f7f6f2; color: #777; border: 1px solid #ddd9d0; }")
        self.setText(placeholder)

    def set_rgb(self, rgb):
        if rgb is None:
            self.set_placeholder(self._placeholder)
            return
        image = numpy_rgb_to_qimage(np.asarray(rgb, np.uint8))
        self._source = QPixmap.fromImage(image)
        self._update_pixmap()

    def set_placeholder(self, text):
        self._source = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def _update_pixmap(self):
        if self._source is None:
            return
        target = self.size() - QSize(8, 8)
        if target.width() <= 0 or target.height() <= 0:
            return
        self.setPixmap(self._source.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()


class ActiveEffectsDialog(QDialog):
    """只读的状态总览与反事实分析面板。"""

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window_ref = window
        self.setWindowTitle("当前有效操作与设置")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setSizeGripEnabled(True)
        self.resize(1120, 760)
        self.setMinimumSize(860, 560)
        self._results = {}
        self._pending_keys = set()
        self._analysis_active = False
        self._analysis_request_id = None
        self._analysis_total = 0
        self._analysis_completed = 0
        self._last_selected_point = -1
        self._displayed_result = None
        self._build()
        self.refresh()

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        header = QHBoxLayout()
        self.file_label = QLabel("当前没有图片")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_label.setStyleSheet("font-size: 15px; font-weight: 500;")
        self.header_status = QLabel("当前结果 · 只读分析")
        self.header_status.setStyleSheet("color: #5f5e5a;")
        header.addWidget(self.file_label, 1)
        header.addWidget(self.header_status)
        root.addLayout(header)

        hint = QLabel(
            "左侧选择因素，右侧查看当前真实状态。反事实计算只在点击后运行，"
            "不会修改正式蒙版、提示历史或工程包。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5f5e5a;")
        root.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal, self)
        self.section_list = QListWidget()
        self.section_list.setMinimumWidth(210)
        self.section_list.setMaximumWidth(290)
        self.section_list.setAlternatingRowColors(True)
        self.section_list.setSpacing(2)
        for title in SECTION_TITLES:
            item = QListWidgetItem(title)
            item.setSizeHint(QSize(220, 58))
            self.section_list.addItem(item)
        splitter.addWidget(self.section_list)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_points_page())
        self.stack.addWidget(self._build_candidate_page())
        self.stack.addWidget(self._build_color_page())
        self.stack.addWidget(self._build_feedback_page())
        self.stack.addWidget(self._build_profile_page())
        self.stack.addWidget(self._build_export_page())
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 860])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        disclaimer = QLabel("变化大小不等于质量改善；模型自评不是真实分割质量。")
        disclaimer.setStyleSheet("color: #854f0b;")
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        footer.addWidget(disclaimer, 1)
        footer.addWidget(close)
        root.addLayout(footer)

        self.section_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.section_list.setCurrentRow(0)

    def _page(self, title, description):
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 14px; font-weight: 500;")
        detail = QLabel(description)
        detail.setWordWrap(True)
        detail.setStyleSheet("color: #5f5e5a;")
        layout.addWidget(heading)
        layout.addWidget(detail)
        return page, layout

    def _build_points_page(self):
        page, layout = self._page(
            "框选与正负点",
            "逐点留一分析会从最近的无 mask_input 节点重放，重建后续每一步的 mask_input。")

        self.point_summary = QLabel("尚未载入提示")
        self.point_summary.setWordWrap(True)
        layout.addWidget(self.point_summary)

        tools = QHBoxLayout()
        self.selected_point_btn = QPushButton("计算选中点影响")
        self.all_points_btn = QPushButton("按需计算全部点")
        self.box_only_btn = QPushButton("框-only 对比")
        self.cancel_effect_btn = QPushButton("取消分析")
        for button in (self.selected_point_btn, self.all_points_btn,
                       self.box_only_btn, self.cancel_effect_btn):
            tools.addWidget(button)
        tools.addStretch(1)
        layout.addLayout(tools)
        self.selected_point_btn.clicked.connect(self._request_selected_point)
        self.all_points_btn.clicked.connect(self._request_all_points)
        self.box_only_btn.clicked.connect(self._request_box_only)
        self.cancel_effect_btn.clicked.connect(self._cancel_analysis)

        progress_row = QHBoxLayout()
        self.effect_progress = QProgressBar()
        self.effect_progress.setRange(0, 1)
        self.effect_progress.setValue(0)
        self.effect_progress.setTextVisible(False)
        self.effect_progress.setVisible(False)
        self.effect_status = QLabel("尚未计算反事实影响")
        self.effect_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.effect_status.setStyleSheet("color: #5f5e5a;")
        progress_row.addWidget(self.effect_progress, 1)
        progress_row.addWidget(self.effect_status, 2)
        layout.addLayout(progress_row)
        box_result_row = QHBoxLayout()
        self.box_only_result = QLabel("框-only：尚未计算")
        self.box_only_result.setWordWrap(True)
        self.box_only_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.box_only_result.setStyleSheet("color: #5f5e5a;")
        self.box_only_view_btn = QPushButton("查看框-only变化")
        self.box_only_view_btn.clicked.connect(self._show_box_only_views)
        box_result_row.addWidget(self.box_only_result, 1)
        box_result_row.addWidget(self.box_only_view_btn)
        layout.addLayout(box_result_row)

        self.point_tabs = QTabWidget()
        list_tab = QWidget()
        list_layout = QVBoxLayout(list_tab)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("筛选"))
        self.point_filter = QComboBox()
        self.point_filter.addItems(["全部点", "仅正点", "仅负点", "仅人工", "仅颜色"])
        self.point_filter.setMinimumWidth(110)
        self.point_filter.currentIndexChanged.connect(self._fill_points)
        filter_row.addWidget(self.point_filter)
        filter_row.addStretch(1)
        list_layout.addLayout(filter_row)
        self.point_table = QTableWidget(0, 8)
        self.point_table.setHorizontalHeaderLabels([
            "点", "来源", "中心", "反事实", "翻转像素", "占画面", "Δ面积", "IoU"])
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.point_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.point_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.point_table.setAlternatingRowColors(True)
        header = self.point_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.point_table.itemSelectionChanged.connect(self._on_point_selected)
        list_layout.addWidget(self.point_table, 3)
        self.point_detail = QTextBrowser()
        self.point_detail.setMinimumHeight(130)
        self.point_detail.setMaximumHeight(240)
        self.point_detail.setPlaceholderText("选择一行查看约束、反事实指标和口径。")
        list_layout.addWidget(self.point_detail, 2)
        self.point_tabs.addTab(list_tab, "点列表与指标")

        heat_tab = QWidget()
        heat_layout = QVBoxLayout(heat_tab)
        self.heat_label = ScaledImageLabel("选择已计算的点查看差异热图")
        self.heat_legend = QLabel(
            "绿色：保留该点后新增；红色：去掉该点后仍保留、但当前缺失。"
            "颜色只表示变化方向，不代表好坏。")
        self.heat_legend.setWordWrap(True)
        self.heat_legend.setStyleSheet("color: #5f5e5a;")
        heat_layout.addWidget(self.heat_label, 1)
        heat_layout.addWidget(self.heat_legend)
        self.point_tabs.addTab(heat_tab, "差异热图")

        ab_tab = QWidget()
        ab_layout = QVBoxLayout(ab_tab)
        ab_row = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.ab_current_title = QLabel("当前结果")
        self.ab_current = ScaledImageLabel("当前结果不可用")
        self.ab_reference_title = QLabel("去点后的反事实结果")
        self.ab_reference = ScaledImageLabel("尚未计算")
        left.addWidget(self.ab_current_title)
        left.addWidget(self.ab_current, 1)
        right.addWidget(self.ab_reference_title)
        right.addWidget(self.ab_reference, 1)
        ab_row.addLayout(left, 1)
        ab_row.addLayout(right, 1)
        ab_layout.addLayout(ab_row, 1)
        ab_note = QLabel(
            "两边使用同一预览尺寸与完整原图范围；反事实自动选择的候选可能不同于当前候选。")
        ab_note.setWordWrap(True)
        ab_note.setStyleSheet("color: #5f5e5a;")
        ab_layout.addWidget(ab_note)
        self.point_tabs.addTab(ab_tab, "A/B 对比")

        layout.addWidget(self.point_tabs, 1)
        return page

    def _build_candidate_page(self):
        page, layout = self._page(
            "当前候选",
            "候选差异直接复用本次 SAM 输出，不重新编码或解码。"
            "SAM 评分统一表示模型自评，不是真实质量。")
        self.candidate_note = QLabel("")
        self.candidate_note.setWordWrap(True)
        layout.addWidget(self.candidate_note)
        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels([
            "候选", "模型自评", "约束", "前景占比", "与当前差异", "状态"])
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setAlternatingRowColors(True)
        header = self.candidate_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.candidate_table, 1)
        note = QLabel(
            "切换候选请使用主窗口的方括号按钮或候选工作台。"
            "评分和标签只保存偏好记录，不参与 decoder。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f5e5a;")
        layout.addWidget(note)
        return page

    def _build_color_page(self):
        page, layout = self._page(
            "颜色辅助",
            "采样只生成颜色样本；真正进入 SAM 的，是随后生成并标记为“颜色”的建议正点。")
        self.color_summary = QLabel("")
        self.color_summary.setWordWrap(True)
        layout.addWidget(self.color_summary)
        self.color_samples_table = QTableWidget(0, 3)
        self.color_samples_table.setHorizontalHeaderLabels(["样本", "RGB", "色块"])
        self.color_samples_table.verticalHeader().setVisible(False)
        self.color_samples_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.color_samples_table.setMaximumHeight(190)
        self.color_samples_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self.color_samples_table)
        points_title = QLabel("由颜色辅助生成的点")
        points_title.setStyleSheet("font-weight: 500;")
        layout.addWidget(points_title)
        self.color_points_table = QTableWidget(0, 4)
        self.color_points_table.setHorizontalHeaderLabels([
            "点", "坐标（原图像素）", "类型", "当前中心约束"])
        self.color_points_table.verticalHeader().setVisible(False)
        self.color_points_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.color_points_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self.color_points_table, 1)
        note = QLabel(
            "采样本身不修改蒙版；未生成点或尚未完成采样时，对当前结果没有可定义的 decoder 输入。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f5e5a;")
        layout.addWidget(note)
        return page

    def _build_feedback_page(self):
        page, layout = self._page(
            "候选偏好反馈",
            "评分、标签和备注用于工程包记录；选择候选本身会影响预览和导出。")
        self.feedback_summary = QLabel("")
        self.feedback_summary.setWordWrap(True)
        layout.addWidget(self.feedback_summary)
        self.feedback_table = QTableWidget(0, 4)
        self.feedback_table.setHorizontalHeaderLabels(["候选", "评分", "标签", "备注"])
        self.feedback_table.verticalHeader().setVisible(False)
        self.feedback_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.feedback_table.setAlternatingRowColors(True)
        header = self.feedback_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.feedback_table, 1)
        self.feedback_note = QLabel("")
        self.feedback_note.setWordWrap(True)
        self.feedback_note.setStyleSheet("color: #5f5e5a;")
        layout.addWidget(self.feedback_note)
        return page

    def _build_profile_page(self):
        page, layout = self._page(
            "工程包",
            "工程包保存对象偏好、候选评分、问题标签、备注和确认示例。")
        self.profile_impact = QLabel("")
        self.profile_impact.setWordWrap(True)
        self.profile_impact.setStyleSheet(
            "background: #faeeda; color: #633806; padding: 8px; border-radius: 5px;")
        layout.addWidget(self.profile_impact)
        self.profile_detail = QTextBrowser()
        layout.addWidget(self.profile_detail, 1)
        return page

    def _build_export_page(self):
        page, layout = self._page(
            "导出精修",
            "导出阶段对当前候选执行原尺寸边缘精修和去污，并写出透明 PNG 与蒙版。")
        self.export_summary = QLabel("")
        self.export_summary.setWordWrap(True)
        layout.addWidget(self.export_summary)
        self.export_detail = QTextBrowser()
        layout.addWidget(self.export_detail, 1)
        note = QLabel("这里只展示实际设置和最近一次真实导出，不估算未运行的精修质量。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f5e5a;")
        layout.addWidget(note)
        return page

    # ------------------------------------------------------------------
    # 刷新
    # ------------------------------------------------------------------
    def refresh(self):
        self._update_header()
        self._fill_section_summaries()
        self._fill_points()
        self._fill_candidates()
        self._fill_color()
        self._fill_feedback()
        self._fill_profile()
        self._fill_export()
        self._update_analysis_buttons()

    def _update_header(self):
        path = getattr(self.window_ref, "_path", None)
        if path:
            name = os.path.basename(str(path))
            self.file_label.setText(name)
            self.file_label.setToolTip(str(path))
            self.header_status.setText(
                "当前结果 · 提示版本 %d · 只读分析" %
                int(getattr(self.window_ref, "_prompt_revision", 0)))
        else:
            self.file_label.setText("当前没有图片")
            self.file_label.setToolTip("")
            self.header_status.setText("等待载入图片")

    def _fill_section_summaries(self):
        window = self.window_ref
        snap = window._history.current
        positive = sum(1 for point in snap.points if int(point.label) == 1)
        negative = len(snap.points) - positive
        if window._result_valid():
            candidate = "候选 %d/%d · 自评 %.3f" % (
                window._candidate + 1, window._result.count,
                window._result.score(window._candidate))
        else:
            candidate = "等待蒙版"
        color_points = sum(1 for point in snap.points if point.source == "color")
        feedback_count = len(window._candidate_draft.get("feedback", {}))
        profile = window._profile.name if window._profile else "未选择"
        export = getattr(window, "_last_export_summary", None)
        export_text = "最近导出已完成" if export else "尚未导出"
        summaries = (
            "框%s · 正 %d / 负 %d" % (
                "已设置" if snap.box_xyxy else "未设置", positive, negative),
            candidate,
            "采样 %d · 颜色点 %d" % (len(window._color_samples), color_points),
            "草稿 %d 项" % feedback_count,
            "%s · 不参与本次推理" % profile,
            export_text,
        )
        for index, summary in enumerate(summaries):
            item = self.section_list.item(index)
            item.setText("%s\n%s" % (SECTION_TITLES[index], summary))
            item.setToolTip("%s：%s" % (SECTION_TITLES[index], summary))

    def _current_masks(self):
        window = self.window_ref
        if not window._result_valid():
            return None, None
        try:
            current_logits = window._result.mask_logits(window._candidate)
            current = np.asarray(current_logits) > 0.0
            return current, window._result
        except (IndexError, AttributeError, TypeError):
            return None, None

    def _visible_points(self, points):
        mode = self.point_filter.currentIndex()
        if mode == 1:
            return [(index, point) for index, point in enumerate(points)
                    if int(point.label) == 1]
        if mode == 2:
            return [(index, point) for index, point in enumerate(points)
                    if int(point.label) == 0]
        if mode == 3:
            return [(index, point) for index, point in enumerate(points)
                    if point.source == "manual"]
        if mode == 4:
            return [(index, point) for index, point in enumerate(points)
                    if point.source == "color"]
        return list(enumerate(points))

    def _fill_points(self):
        window = self.window_ref
        snap = window._history.current
        points = snap.points
        current_mask, _result = self._current_masks()
        self.point_summary.setText(
            "框：%s　　正点 %d · 负点 %d　　人工 %d · 颜色 %d" % (
                "已设置" if snap.box_xyxy else "未设置",
                sum(1 for p in points if int(p.label) == 1),
                sum(1 for p in points if int(p.label) == 0),
                sum(1 for p in points if p.source == "manual"),
                sum(1 for p in points if p.source == "color")))
        table = self.point_table
        selected = self._selected_point_index()
        if selected < 0:
            selected = self._last_selected_point
        visible = self._visible_points(points)
        table.blockSignals(True)
        table.setRowCount(len(visible))
        for row, (point_index, point) in enumerate(visible):
            key = "point_loo:%d" % point_index
            result = self._results.get(key)
            center = "—"
            if current_mask is not None:
                try:
                    local = point_local_metrics(
                        current_mask, point, window._original_hw)
                    center = "符合" if local.center_ok else "不符合"
                except (ValueError, TypeError):
                    center = "—"
            if result is not None:
                status = "完成"
                if result.candidate_changed:
                    status = "完成 · 候选切换"
                metrics = result.metrics
                flip = _fmt_int(metrics.changed_pixels)
                share = _fmt_ratio(metrics.changed_image_ratio)
                delta = "%+d px" % metrics.area_delta_pixels
                if metrics.area_delta_ratio is not None:
                    delta += " (%+.1f%%)" % (metrics.area_delta_ratio * 100.0)
                iou = "%.3f" % metrics.iou
                tooltip = (
                    "参照：去掉该点后的反事实结果；新增 %s px，移除 %s px。\n"
                    "Δ面积以反事实前景为基准；IoU 描述整体重合。" % (
                        _fmt_int(metrics.added_pixels),
                        _fmt_int(metrics.removed_pixels)))
            elif key in self._pending_keys:
                status = "计算中"
                flip = share = delta = iou = "—"
                tooltip = "正在后台重放实际提示链路。"
            else:
                status = "未计算"
                flip = share = delta = iou = "—"
                tooltip = "点击“计算选中点影响”后生成，不自动运行重型分析。"
            values = (
                point_display_name(points, point_index),
                "颜色" if point.source == "color" else "人工",
                center,
                status,
                flip,
                share,
                delta,
                iou,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, point_index)
                if column in (4, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setToolTip(tooltip)
                table.setItem(row, column, item)
        visible_indices = [point_index for point_index, _point in visible]
        if visible_indices:
            if selected not in visible_indices:
                selected = visible_indices[0]
            row = visible_indices.index(selected)
            table.selectRow(row)
            self._last_selected_point = selected
        else:
            self._last_selected_point = -1
        table.blockSignals(False)
        self._on_point_selected()
        self._fill_box_only_summary()
        self._update_analysis_buttons()

    def _fill_box_only_summary(self):
        result = self._results.get("box_only")
        if result is None:
            self.box_only_btn.setToolTip(
                "保留当前框、去掉全部修正点，重跑一次 decoder。")
            self.box_only_result.setText(
                "框-only：尚未计算。用于观察全部修正点合在一起带来的变化。")
            self.box_only_view_btn.setEnabled(False)
            return
        self.box_only_view_btn.setEnabled(True)
        metrics = result.metrics
        self.box_only_btn.setToolTip(
            "框-only 对比：翻转 %s px，占画面 %s，IoU %.3f。" % (
                _fmt_int(metrics.changed_pixels),
                _fmt_ratio(metrics.changed_image_ratio),
                metrics.iou))
        self.box_only_result.setText(
            "框-only：翻转 %s px（占画面 %s），新增 %s / 移除 %s，"
            "Δ面积 %+d px，IoU %.3f；反事实候选 %d%s。" % (
                _fmt_int(metrics.changed_pixels),
                _fmt_ratio(metrics.changed_image_ratio),
                _fmt_int(metrics.added_pixels),
                _fmt_int(metrics.removed_pixels),
                metrics.area_delta_pixels,
                metrics.iou,
                result.counterfactual_candidate + 1,
                "（候选切换）" if result.candidate_changed else ""))

    def _selected_point_index(self):
        row = self.point_table.currentRow()
        if row < 0:
            return -1
        item = self.point_table.item(row, 0)
        if item is None:
            return -1
        value = item.data(Qt.UserRole)
        return -1 if value is None else int(value)

    def _on_point_selected(self):
        point_index = self._selected_point_index()
        points = self.window_ref._history.current.points
        if point_index < 0 or point_index >= len(points):
            self.point_detail.setPlainText("选择一个点查看局部约束和反事实指标。")
            self._update_effect_views(self._results.get("box_only"))
            self._update_analysis_buttons()
            return
        self._last_selected_point = point_index
        self._update_point_detail(point_index)
        selected_result = self._results.get("point_loo:%d" % point_index)
        if selected_result is None:
            selected_result = self._results.get("box_only")
        self._update_effect_views(selected_result)
        self._update_analysis_buttons()

    def _update_point_detail(self, row):
        window = self.window_ref
        points = window._history.current.points
        point = points[row]
        name = point_display_name(points, row)
        current_mask, _result = self._current_masks()
        lines = [
            "%s　来源：%s　坐标：(%.1f, %.1f) 原图像素" % (
                name, "颜色辅助" if point.source == "color" else "人工",
                point.x, point.y),
        ]
        if current_mask is not None:
            try:
                local = point_local_metrics(
                    current_mask, point, window._original_hw)
                lines.append(
                    "当前局部观察：中心%s；支持率 %.1f%%；边界接近度 %.1f%%；"
                    "采样半径 %d SAM 网格像素。" % (
                        "符合" if local.center_ok else "不符合",
                        local.support * 100.0,
                        local.boundary_proximity * 100.0,
                        local.radius))
            except (ValueError, TypeError):
                lines.append("当前局部观察不可用。")
        else:
            lines.append("当前尚未生成有效蒙版，局部观察不可用。")
        result = self._results.get("point_loo:%d" % row)
        if result is None:
            lines.append("")
            lines.append(
                "尚未计算反事实影响。局部观察不回答“去掉这个点后会怎样变化”。")
        else:
            metrics = result.metrics
            lines.extend([
                "",
                "实际链路重放：从最近无 mask_input 节点开始，去掉该点后重建每一步输入。",
                "当前候选 %d；反事实自动选择候选 %d%s。" % (
                    result.current_candidate + 1,
                    result.counterfactual_candidate + 1,
                    "（发生候选切换）" if result.candidate_changed else ""),
                "翻转像素 %s（占画面 %s）；新增 %s / 移除 %s。" % (
                    _fmt_int(metrics.changed_pixels),
                    _fmt_ratio(metrics.changed_image_ratio),
                    _fmt_int(metrics.added_pixels),
                    _fmt_int(metrics.removed_pixels)),
                "Δ面积 %+d px%s；IoU %.3f；Dice %.3f。" % (
                    metrics.area_delta_pixels,
                    (" (相对反事实 %+.1f%%)" %
                     (metrics.area_delta_ratio * 100.0)
                     if metrics.area_delta_ratio is not None else ""),
                    metrics.iou, metrics.dice),
                "分析网格：%d×%d（SAM 重采样尺寸）。" % (
                    result.current_mask.shape[1], result.current_mask.shape[0]),
                "反事实→当前：边界 P95 %s SAM 像素；连通域 %d→%d；孔洞 %d→%d。" % (
                    _fmt_optional_float(metrics.boundary_p95, 2),
                    metrics.before_components, metrics.after_components,
                    metrics.before_holes, metrics.after_holes),
                "耗时 %.2fs；decoder %d 次；缓存命中 %d 次。" % (
                    result.seconds, result.decodes, result.cache_hits),
            ])
        self.point_detail.setPlainText("\n".join(lines))

    def _show_box_only_views(self):
        result = self._results.get("box_only")
        if result is None:
            self.effect_status.setText("框-only 结果尚未计算")
            return
        self._update_effect_views(result)
        self.point_tabs.setCurrentIndex(1)

    def _update_effect_views(self, result):
        if result is None:
            self._displayed_result = None
            self.heat_label.set_placeholder("选择已计算的点查看差异热图")
            self.ab_current.set_placeholder("当前结果不可用")
            self.ab_reference.set_placeholder("尚未计算")
            self.ab_current_title.setText("当前结果")
            self.ab_reference_title.setText("去点后的反事实结果")
            self.heat_legend.setText(
                "绿色：保留该点后新增；红色：去掉该点后仍保留、但当前缺失。"
                "颜色只表示变化方向，不代表好坏。")
            return
        if result is self._displayed_result:
            return
        preview = self.window_ref._preview_rgb
        if preview is None:
            return
        current_preview = mask_to_preview(
            result.current_mask, preview.shape[:2])
        reference_preview = mask_to_preview(
            result.counterfactual_mask, preview.shape[:2])
        try:
            overlay = build_diff_overlay(
                preview, current_preview, reference_preview)
            current_alpha = current_preview.astype(np.float32)
            reference_alpha = reference_preview.astype(np.float32)
            current_cutout = pipeline.compose_cutout_preview(
                preview, current_alpha, False)
            reference_cutout = pipeline.compose_cutout_preview(
                preview, reference_alpha, False)
        except (ValueError, TypeError):
            self.heat_label.set_placeholder("当前无法生成差异视图")
            return
        self._displayed_result = result
        self.heat_label.set_rgb(overlay)
        self.ab_current.set_rgb(current_cutout)
        self.ab_reference.set_rgb(reference_cutout)
        self.ab_current_title.setText(
            "当前结果 · 候选 %d" % (result.current_candidate + 1))
        if result.plan.kind == "box_only":
            suffix = ""
            if result.candidate_changed:
                suffix = " · 候选已切换"
            self.ab_reference_title.setText(
                "框-only（无修正点）· 自动候选 %d%s" % (
                    result.counterfactual_candidate + 1, suffix))
            self.heat_legend.setText(
                "绿色：当前结果相对框-only新增；红色：框-only存在、"
                "但当前结果缺失。颜色只表示变化方向，不代表好坏。")
            return
        suffix = ""
        if result.candidate_changed:
            suffix = " · 候选已切换"
        self.ab_reference_title.setText(
            "去掉该点后 · 自动候选 %d%s" % (
                result.counterfactual_candidate + 1, suffix))
        self.heat_legend.setText(
            "绿色：保留该点后新增；红色：去掉该点后仍保留、但当前缺失。"
            "颜色只表示变化方向，不代表好坏。")

    def _fill_candidates(self):
        result = self.window_ref._result if self.window_ref._result_valid() else None
        if result is None:
            self.candidate_table.setRowCount(0)
            self.candidate_note.setText("等待框选或点提示生成 SAM 候选。")
            return
        window = self.window_ref
        points = window._history.current.points
        current_mask, _result = self._current_masks()
        self.candidate_note.setText(
            "当前选择候选 %d/%d；下表仅展示现有输出，不触发新的解码。" % (
                window._candidate + 1, result.count))
        table = self.candidate_table
        table.setRowCount(result.count)
        for index in range(result.count):
            try:
                mask = np.asarray(result.mask_logits(index)) > 0.0
                score = result.score(index)
            except (IndexError, AttributeError):
                continue
            constraint = (
                "%.0f%%" % (point_constraint_rate(
                    mask, points, window._original_hw) * 100.0)
                if points else "—")
            if current_mask is None:
                difference = "—"
            elif index == window._candidate:
                difference = "当前"
            else:
                changed = int(np.logical_xor(mask, current_mask).sum())
                union = int(np.logical_or(mask, current_mask).sum())
                intersection = int(np.logical_and(mask, current_mask).sum())
                iou = 1.0 if union == 0 else intersection / float(union)
                difference = "%s px · IoU %.3f" % (_fmt_int(changed), iou)
            values = (
                "候选 %d" % (index + 1),
                "%.3f" % score,
                constraint,
                "%.2f%%" % (float(mask.mean()) * 100.0),
                difference,
                "当前" if index == window._candidate else "可切换",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (1, 2, 3):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if index == window._candidate:
                    item.setBackground(QColor("#e6f1fb"))
                table.setItem(index, column, item)

    def _fill_color(self):
        window = self.window_ref
        snap = window._history.current
        current_count = len(window._color_samples)
        recent = list(getattr(window, "_last_color_samples", []) or [])
        color_points = [
            (index, point) for index, point in enumerate(snap.points)
            if point.source == "color"
        ]
        self.color_summary.setText(
            "当前采样 %d/8%s；最近一次样本 %d 个；颜色辅助点 %d 个。" % (
                current_count,
                "（采样中）" if window._color_assist_active else "（未在采样）",
                len(recent), len(color_points)))
        samples = window._color_samples or recent
        self.color_samples_table.setRowCount(len(samples))
        for index, sample in enumerate(samples):
            values = np.asarray(sample).reshape(-1)
            rgb = [int(v) for v in values[:3]]
            while len(rgb) < 3:
                rgb.append(0)
            self.color_samples_table.setItem(
                index, 0, QTableWidgetItem(str(index + 1)))
            self.color_samples_table.setItem(
                index, 1, QTableWidgetItem("(%d, %d, %d)" % tuple(rgb)))
            swatch = QTableWidgetItem("")
            swatch.setBackground(QColor(*rgb))
            self.color_samples_table.setItem(index, 2, swatch)
        current_mask, _result = self._current_masks()
        self.color_points_table.setRowCount(len(color_points))
        for row, (point_index, point) in enumerate(color_points):
            center = "—"
            if current_mask is not None:
                try:
                    local = point_local_metrics(
                        current_mask, point, window._original_hw)
                    center = "符合" if local.center_ok else "不符合"
                except (ValueError, TypeError):
                    pass
            values = (
                point_display_name(snap.points, point_index),
                "(%.1f, %.1f)" % (point.x, point.y),
                "正点",
                center,
            )
            for column, value in enumerate(values):
                self.color_points_table.setItem(
                    row, column, QTableWidgetItem(value))

    def _fill_feedback(self):
        window = self.window_ref
        draft = window._candidate_draft or {}
        feedback = draft.get("feedback", {})
        worst = draft.get("worst")
        selected = draft.get("selected", window._candidate)
        rating = feedback.get(str(selected), {}).get("rating")
        self.feedback_summary.setText(
            "反馈草稿 %d 项；当前草稿候选 %d；评分 %s；最差标记 %s。" % (
                len(feedback),
                int(selected) + 1,
                str(rating) if rating else "未评分",
                ("候选 %d" % (int(worst) + 1))
                if worst is not None else "未标记"))
        self.feedback_table.setRowCount(len(feedback))
        for row, key in enumerate(sorted(feedback, key=lambda value: int(value))):
            item = feedback[key]
            values = (
                "候选 %d" % (int(key) + 1),
                str(item.get("rating") or "未评分"),
                "、".join(item.get("tags", [])) or "无",
                item.get("note", "") or "无",
            )
            for column, value in enumerate(values):
                self.feedback_table.setItem(
                    row, column, QTableWidgetItem(value))
        if window._profile is None:
            self.feedback_note.setText(
                "未选择工程包：草稿只保留在内存中，不会写入磁盘。")
        elif not feedback:
            self.feedback_note.setText(
                "已选择工程包；尚未在工作台保存候选反馈。")
        else:
            self.feedback_note.setText(
                "已选择工程包；保存反馈会写入 %s。" % window._profile_path)
        self.feedback_note.setToolTip(str(window._profile_path or ""))

    def _fill_profile(self):
        window = self.window_ref
        profile = window._profile
        if profile is None:
            self.profile_impact.setText(
                "对本次蒙版：工程包未参与 SAM 解码；当前没有使用工程包。")
            self.profile_detail.setPlainText(
                "未选择工程包。\n\n"
                "工程包可以保存反馈、问题标签、颜色样本和确认示例，"
                "但本次实现不会用这些记录重排候选或改写提示。")
            return
        self.profile_impact.setText(
            "对本次蒙版：不参与 SAM 解码（工程包尚未接入推理）。"
            "这里不把未接入的作用伪造成数值 0。")
        ratings = {}
        tags = {}
        worst = 0
        for record in profile.feedback:
            if record.rating:
                ratings[int(record.rating)] = ratings.get(int(record.rating), 0) + 1
            for tag in record.tags:
                tags[tag] = tags.get(tag, 0) + 1
            if record.worst is not None:
                worst += 1
        rating_text = "、".join(
            "%d 分 %d 次" % (key, ratings[key]) for key in sorted(ratings)) or "无"
        tag_text = "、".join(
            "%s %d 次" % (key, tags[key])
            for key in sorted(tags, key=lambda k: (-tags[k], k))[:8]) or "无"
        lines = [
            "名称：%s" % profile.name,
            "路径：%s" % (window._profile_path or "尚未保存"),
            "确认示例：%d 个" % len(profile.examples),
            "候选反馈：%d 条" % len(profile.feedback),
            "评分分布：%s" % rating_text,
            "最差标记：%d 条" % worst,
            "问题标签：%s" % tag_text,
            "颜色样本资产：%d 个" % len(profile.color_samples_rgb),
            "",
            "工作流影响：评分、标签和样本当前只用于记录；"
            "候选选择会改变预览和导出，但选择动作本身不是工程包学习。",
        ]
        self.profile_detail.setPlainText("\n".join(lines))

    def _fill_export(self):
        window = self.window_ref
        output = str(window.out_dir)
        self.export_summary.setText(
            "导出精修：开启（原尺寸 matting + 引导上采样 + 去污）；"
            "输出目录：%s" % output)
        self.export_summary.setToolTip(output)
        summary = getattr(window, "_last_export_summary", None)
        if not summary:
            text = (
                "尚未导出当前图片。\n\n"
                "这里不会根据当前预览推测精修后的边界、透明度或质量；"
                "只有实际导出完成后才显示真实输出。")
        else:
            lines = [
                "最近导出时间：%s" % summary.get("time", "未知"),
                "耗时：%.2f 秒" % float(summary.get("seconds", 0.0)),
                "前景：%s" % summary.get("fg_path", ""),
                "蒙版：%s" % summary.get("mask_path", ""),
                "反馈草稿：导出成功后已清除。",
            ]
            text = "\n".join(lines)
        self.export_detail.setPlainText(text)

    # ------------------------------------------------------------------
    # 分析请求与事件
    # ------------------------------------------------------------------
    def _request_selected_point(self):
        point_index = self._selected_point_index()
        if point_index < 0:
            self.effect_status.setText("请先选择一个点")
            return
        try:
            self.window_ref.request_point_effect(point_index)
        except (ValueError, IndexError, RuntimeError) as exc:
            self.effect_status.setText("无法分析：%s" % exc)

    def _request_all_points(self):
        try:
            self.window_ref.request_all_point_effects()
        except (ValueError, IndexError, RuntimeError) as exc:
            self.effect_status.setText("无法分析：%s" % exc)

    def _request_box_only(self):
        try:
            self.window_ref.request_box_only_effect()
        except (ValueError, IndexError, RuntimeError) as exc:
            self.effect_status.setText("无法分析：%s" % exc)

    def _cancel_analysis(self):
        self.window_ref.cancel_effect_analysis()

    def _update_analysis_buttons(self):
        window = self.window_ref
        valid = bool(window._result_valid())
        has_points = bool(window._history.current.points)
        has_box = window._history.current.box_xyxy is not None
        point_index = self._selected_point_index()
        self.selected_point_btn.setEnabled(
            valid and point_index >= 0 and not self._analysis_active)
        self.all_points_btn.setEnabled(valid and has_points and not self._analysis_active)
        self.box_only_btn.setEnabled(
            valid and has_box and has_points and not self._analysis_active)
        self.cancel_effect_btn.setEnabled(self._analysis_active)

    def on_analysis_started(self, request_id, plans):
        self._analysis_active = True
        self._analysis_request_id = int(request_id)
        self._analysis_total = len(plans)
        self._analysis_completed = 0
        self._pending_keys = {
            "point_loo:%d" % plan.target_point_index
            if plan.target_point_index is not None else plan.kind
            for plan in plans
        }
        for key in self._pending_keys:
            self._results.pop(key, None)
        self.effect_progress.setRange(0, max(1, self._analysis_total))
        self.effect_progress.setValue(0)
        self.effect_progress.setVisible(True)
        self.effect_status.setText(
            "已开始 %d 项分析；正式编辑会被优先处理。" % self._analysis_total)
        self._fill_points()
        self._update_analysis_buttons()

    def on_effect_progress(self, event):
        if event.request_id != self._analysis_request_id:
            return
        self._analysis_completed = int(event.completed)
        self._analysis_total = max(1, int(event.total))
        self.effect_progress.setRange(0, self._analysis_total)
        self.effect_progress.setValue(self._analysis_completed)
        self.effect_status.setText(event.message)

    def on_effect_ready(self, event):
        if event.request_id != self._analysis_request_id:
            return
        result = event.result
        key = result.plan.key
        self._results[key] = result
        self._pending_keys.discard(key)
        self._analysis_completed = int(event.completed)
        self._analysis_total = max(1, int(event.total))
        self.effect_progress.setRange(0, self._analysis_total)
        self.effect_progress.setValue(self._analysis_completed)
        self.effect_status.setText(
            "已完成 %d/%d 项；最后一项 %s。" % (
                self._analysis_completed, self._analysis_total,
                result.plan.label))
        self._fill_points()
        if result.plan.kind == "box_only":
            current = result
        else:
            row = self._last_selected_point
            current = self._results.get("point_loo:%d" % row) if row >= 0 else None
        if current is not None:
            self._update_effect_views(current)
        self._update_analysis_buttons()

    def on_effects_finished(self, event):
        if event.request_id != self._analysis_request_id:
            return
        self._analysis_active = False
        self._pending_keys.clear()
        self.effect_progress.setRange(0, max(1, event.total))
        self.effect_progress.setValue(event.total)
        self.effect_progress.setVisible(False)
        self.effect_status.setText(
            "分析完成：%d 项，总耗时 %.2fs。" % (
                event.total, event.seconds))
        self._update_analysis_buttons()

    def on_effect_failed(self, request_id, message):
        if request_id == self._analysis_request_id:
            self._analysis_active = False
            self._pending_keys.clear()
            self.effect_progress.setVisible(False)
            self.effect_status.setText("分析失败：%s" % message)
            self._fill_points()
            self._update_analysis_buttons()
            return
        self.effect_status.setText("反事实分析失败：%s" % message)

    def on_analysis_cancelled(self, message="分析已取消"):
        self._analysis_active = False
        self._pending_keys.clear()
        self._analysis_request_id = None
        self.effect_progress.setRange(0, 1)
        self.effect_progress.setValue(0)
        self.effect_progress.setVisible(False)
        self.effect_status.setText(message)
        self._fill_points()
        self._update_analysis_buttons()

    def on_state_changed(self):
        self._results.clear()
        self._pending_keys.clear()
        self._displayed_result = None
        self._analysis_active = False
        self._analysis_request_id = None
        self._analysis_completed = 0
        self._analysis_total = 0
        self.effect_progress.setRange(0, 1)
        self.effect_progress.setValue(0)
        self.effect_progress.setVisible(False)
        self.effect_status.setText("状态已更新，请重新计算反事实影响")
        self.heat_label.set_placeholder("选择已计算的点查看差异热图")
        self.ab_current.set_placeholder("当前结果不可用")
        self.ab_reference.set_placeholder("尚未计算")
        self.point_detail.setPlainText("状态已更新；旧分析结果已清除。")
        self.refresh()

    def closeEvent(self, event):
        if self._analysis_active:
            self.window_ref.cancel_effect_analysis()
            self._analysis_active = False
        super().closeEvent(event)
