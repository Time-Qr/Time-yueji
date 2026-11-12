# -*- coding: utf-8 -*-
"""多候选对比、细节预览和阶段性反馈。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)


class ZoomGraphicsView(QGraphicsView):
    def wheelEvent(self, event):
        self.scale(1.2 if event.angleDelta().y() > 0 else 0.83,
                   1.2 if event.angleDelta().y() > 0 else 0.83)
        event.accept()


class CandidateImageLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class CandidatePreviewDialog(QDialog):
    def __init__(self, image, title, original=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("细节预览 — %s" % title)
        self.resize(1100, 800)
        self._image = image
        self._original = original
        self._view = ZoomGraphicsView(self)
        self._view.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._scene = QGraphicsScene(self._view)
        self._item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        self._scene.addItem(self._item)
        self._view.setScene(self._scene)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("滚轮：缩放　左键拖动：平移"))
        layout.addWidget(self._view)
        if original is not None:
            self.compare = QCheckBox("对比原图")
            self.compare.toggled.connect(self._toggle_original)
            layout.addWidget(self.compare)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        fit = buttons.addButton("适应窗口", QDialogButtonBox.ActionRole)
        fit.clicked.connect(self._fit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._fit()

    def _toggle_original(self, checked):
        self._item.setPixmap(QPixmap.fromImage(self._original if checked else self._image))
        self._fit()

    def _fit(self):
        self._view.resetTransform()
        self._view.fitInView(self._item, Qt.KeepAspectRatio)


class CandidateWorkbench(QDialog):
    def __init__(self, candidates, image_name, on_feedback, initial=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("候选结果工作台 — %s" % image_name)
        self.resize(1100, 760)
        self.setMinimumSize(720, 520)
        self._candidates = candidates
        self._on_feedback = on_feedback
        self._draft = dict(initial or {})
        self._selected = int(self._draft.get("selected", 0))
        self._cards = []
        self._build()
        self._select_card(min(self._selected, max(0, len(self._cards) - 1)))

    def _build(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("可单独保存任意候选；关闭工作台会保留草稿，直到导出。"))
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        grid = QGridLayout()
        for index, candidate in enumerate(self._candidates):
            box = QGroupBox(candidate["title"])
            box.setCheckable(True)
            box.clicked.connect(lambda _=False, i=index: self._select_card(i))
            self._cards.append(box)
            box_layout = QVBoxLayout(box)
            image = CandidateImageLabel()
            image.setAlignment(Qt.AlignCenter)
            image.clicked.connect(lambda i=index: self._select_card(i))
            image.setPixmap(QPixmap.fromImage(candidate["image"]).scaled(
                280, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            box_layout.addWidget(image)
            box_layout.addWidget(QLabel("SAM 评分：%.3f" % candidate["score"]))
            choose = QPushButton("选为最佳")
            choose.clicked.connect(lambda _=False, i=index: self._choose(i))
            box_layout.addWidget(choose)
            worst = QPushButton("标为最差")
            worst.clicked.connect(lambda _=False, i=index: self._mark_worst(i))
            box_layout.addWidget(worst)
            preview = QPushButton("放大查看细节")
            preview.clicked.connect(lambda _=False, i=index: self._preview(i))
            box_layout.addWidget(preview)
            grid.addWidget(box, index // 3, index % 3)
        content_layout.addLayout(grid)
        feedback = QGroupBox("当前选择的反馈")
        feedback_layout = QVBoxLayout(feedback)
        self.rating = QComboBox()
        self.rating.addItems(["未评分", "1 - 不可用", "2 - 较差", "3 - 可用", "4 - 良好", "5 - 最佳"])
        self.tag_presets = QComboBox()
        self.tag_presets.addItems(["选择预设标签", "漏细枝", "漏叶片", "保留花盆",
                                   "背景残留", "边缘过硬", "边缘透明", "主体断裂"])
        self.tag_presets.currentIndexChanged.connect(self._add_preset_tag)
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("可输入新标签，用逗号分隔")
        self.note = QTextEdit()
        self.note.setPlaceholderText("可选备注：为什么选它，或哪里需要改进")
        feedback_layout.addWidget(QLabel("评分"))
        feedback_layout.addWidget(self.rating)
        feedback_layout.addWidget(QLabel("问题标签"))
        feedback_layout.addWidget(self.tag_presets)
        feedback_layout.addWidget(self.tags_edit)
        feedback_layout.addWidget(QLabel("备注"))
        feedback_layout.addWidget(self.note)
        content_layout.addWidget(feedback)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox()
        save = buttons.addButton("保存当前候选", QDialogButtonBox.ActionRole)
        save_close = buttons.addButton("保存并关闭", QDialogButtonBox.AcceptRole)
        close = buttons.addButton("关闭（保留草稿）", QDialogButtonBox.RejectRole)
        save.clicked.connect(self._save_draft)
        save_close.clicked.connect(self._save_and_close)
        close.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _choose(self, index):
        self._select_card(index)

    def _select_card(self, index):
        self._selected = int(index)
        for i, card in enumerate(self._cards):
            card.setChecked(i == self._selected)
            card.setStyleSheet(
                "QGroupBox { border: 2px solid %s; border-radius: 4px; }" %
                ("#2b78c5" if i == self._selected else "#bdbdbd"))
        self._load_feedback(self._selected)

    def _mark_worst(self, index):
        self._select_card(index)
        self._draft["worst"] = int(index)
        self._save_draft()

    def _load_feedback(self, index):
        item = self._draft.get("feedback", {}).get(str(index), {})
        self.rating.setCurrentIndex(int(item.get("rating") or 0))
        self.tags_edit.setText(",".join(item.get("tags", [])))
        self.note.setPlainText(item.get("note", ""))

    def _add_preset_tag(self, index):
        if index <= 0:
            return
        tags = [x.strip() for x in self.tags_edit.text().split(",") if x.strip()]
        tag = self.tag_presets.itemText(index)
        if tag not in tags:
            tags.append(tag)
            self.tags_edit.setText(",".join(tags))
        self.tag_presets.setCurrentIndex(0)

    def _preview(self, index):
        candidate = self._candidates[int(index)]
        CandidatePreviewDialog(candidate["image"], candidate["title"],
                               candidate.get("original"), self).exec()

    def _save_draft(self):
        rating = self.rating.currentIndex()
        tags = [x.strip() for x in self.tags_edit.text().split(",") if x.strip()]
        self._draft.setdefault("feedback", {})[str(self._selected)] = {
            "rating": rating if rating else None,
            "tags": tags,
            "note": self.note.toPlainText().strip(),
        }
        self._draft["selected"] = self._selected
        self._on_feedback(self._draft, False)

    def _save_and_close(self):
        self._save_draft()
        self._on_feedback(self._draft, True)
        self.accept()
