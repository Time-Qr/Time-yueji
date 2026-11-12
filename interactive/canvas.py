# -*- coding: utf-8 -*-
"""图片画布：缩放、平移、框选和正负点标记。"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsScene,
    QGraphicsView,
)


MODE_BOX = "box"
MODE_POSITIVE = "positive"
MODE_NEGATIVE = "negative"
MODE_PAN = "pan"
MODE_COLOR_SAMPLE = "color_sample"


def numpy_rgb_to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888).copy()


class ImageCanvas(QGraphicsView):
    boxSelected = Signal(float, float, float, float)
    boxAdjusted = Signal(float, float, float, float)
    pointSelected = Signal(float, float, int)
    colorSampled = Signal(float, float)
    zoomChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pix_item)
        self._box_item = None
        self._rubber_item = None
        self._point_items = []
        self._point_label_items = []
        self._handle_items = {}
        self._selection_scene_rect = None
        self._resize_handle = None
        self._resize_original_rect = None
        self._mode = MODE_BOX
        self._preview_hw = (0, 0)
        self._original_hw = (0, 0)
        self._drag_start = None
        self._pan_last = None
        self._space_pan = False
        self._fit_pending = False
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#f3f1ec")))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setWhatsThis(
            "图片画布：框选模式拖出目标框；正点加入前景；负点排除误选；"
            "平移模式拖动画布；滚轮缩放；按住空格可临时平移。")

    def reset_document(self):
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pix_item)
        self._box_item = None
        self._rubber_item = None
        self._point_items = []
        self._point_label_items = []
        self._handle_items = {}
        self._selection_scene_rect = None
        self._resize_handle = None
        self._resize_original_rect = None
        self._preview_hw = (0, 0)
        self._original_hw = (0, 0)
        self.resetTransform()
        self.setSceneRect(QRectF())

    def set_image(self, qimage, original_hw):
        self._scene.clear()
        self._pix_item = self._scene.addPixmap(QPixmap.fromImage(qimage))
        self._pix_item.setTransformationMode(Qt.SmoothTransformation)
        self._preview_hw = (int(qimage.height()), int(qimage.width()))
        self._original_hw = (int(original_hw[0]), int(original_hw[1]))
        self._scene.setSceneRect(QRectF(0, 0, qimage.width(), qimage.height()))
        self._box_item = None
        self._rubber_item = None
        self._point_items = []
        self._handle_items = {}
        self._selection_scene_rect = None
        self._resize_handle = None
        self._resize_original_rect = None
        self._fit_pending = True
        self.fit_to_view()

    @property
    def has_image(self):
        return self._pix_item is not None and self._pix_item.pixmap().width() > 0

    def set_mode(self, mode):
        self._mode = mode
        if mode == MODE_PAN:
            self.setCursor(QCursor(Qt.OpenHandCursor))
            self.setDragMode(QGraphicsView.NoDrag)
        elif mode == MODE_BOX:
            self.setCursor(QCursor(Qt.CrossCursor))
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setCursor(QCursor(Qt.PointingHandCursor))
            self.setDragMode(QGraphicsView.NoDrag)

    def update_image(self, qimage):
        """仅更新画面，不改变缩放、平移和当前选区。"""
        if self._pix_item is None:
            return
        same_size = (qimage.width(), qimage.height()) == self._preview_hw[::-1]
        self._pix_item.setPixmap(QPixmap.fromImage(qimage))
        if not same_size:
            self._preview_hw = (int(qimage.height()), int(qimage.width()))
            self._scene.setSceneRect(QRectF(0, 0, qimage.width(), qimage.height()))

    def fit_to_view(self):
        if not self.has_image:
            return
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)
        self._fit_pending = True
        self.zoomChanged.emit()

    def _scene_to_original(self, point):
        if not self.has_image:
            return None
        preview_h, preview_w = self._preview_hw
        original_h, original_w = self._original_hw
        x = min(max(point.x(), 0.0), preview_w) * original_w / float(preview_w)
        y = min(max(point.y(), 0.0), preview_h) * original_h / float(preview_h)
        return x, y

    def _original_to_scene(self, x, y):
        preview_h, preview_w = self._preview_hw
        original_h, original_w = self._original_hw
        return QPointF(
            x * preview_w / float(original_w),
            y * preview_h / float(original_h),
        )

    def clear_selection(self):
        if self._box_item is not None:
            self._scene.removeItem(self._box_item)
            self._box_item = None
        for item in self._point_items:
            self._scene.removeItem(item)
        self._point_items = []
        for item in self._point_label_items:
            self._scene.removeItem(item)
        self._point_label_items = []
        self._clear_handles()
        self._selection_scene_rect = None
        self._resize_handle = None
        self._resize_original_rect = None
        self._remove_rubber()

    def set_selection(self, box_xyxy=None, points=(), active=False):
        self.clear_selection()
        if box_xyxy is not None:
            x0, y0, x1, y1 = box_xyxy
            p0 = self._original_to_scene(x0, y0)
            p1 = self._original_to_scene(x1, y1)
            rect = QRectF(p0, p1).normalized()
            self._selection_scene_rect = rect
            pen = QPen(QColor("#f0a000") if active else QColor("#d18b00"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine if active else Qt.SolidLine)
            self._box_item = self._scene.addRect(rect, pen, QBrush(Qt.NoBrush))
            self._box_item.setZValue(20)
            self._draw_handles(rect)
        positive_index = 0
        negative_index = 0
        for point in points:
            scene_pt = self._original_to_scene(point.x, point.y)
            color = QColor("#168a55") if point.label == 1 else QColor("#d83b3b")
            if point.label == 1:
                positive_index += 1
                label = "正%d" % positive_index
            else:
                negative_index += 1
                label = "负%d" % negative_index
            item = QGraphicsEllipseItem(-5, -5, 10, 10)
            item.setPos(scene_pt)
            item.setBrush(QBrush(color))
            item.setPen(QPen(QColor("white"), 1.5))
            item.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            item.setZValue(30)
            self._scene.addItem(item)
            self._point_items.append(item)
            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(color))
            font = text.font()
            font.setBold(True)
            text.setFont(font)
            text.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
            text.setPos(scene_pt + QPointF(7, -14))
            text.setZValue(31)
            self._scene.addItem(text)
            self._point_label_items.append(text)

    @staticmethod
    def _handle_positions(rect):
        cx = (rect.left() + rect.right()) * 0.5
        cy = (rect.top() + rect.bottom()) * 0.5
        return {
            "nw": QPointF(rect.left(), rect.top()),
            "n": QPointF(cx, rect.top()),
            "ne": QPointF(rect.right(), rect.top()),
            "e": QPointF(rect.right(), cy),
            "se": QPointF(rect.right(), rect.bottom()),
            "s": QPointF(cx, rect.bottom()),
            "sw": QPointF(rect.left(), rect.bottom()),
            "w": QPointF(rect.left(), cy),
        }

    def _clear_handles(self):
        for item in self._handle_items.values():
            self._scene.removeItem(item)
        self._handle_items = {}

    def _draw_handles(self, rect):
        self._clear_handles()
        for name, point in self._handle_positions(rect).items():
            item = QGraphicsRectItem(-4, -4, 8, 8)
            item.setPos(point)
            item.setBrush(QBrush(QColor("white")))
            item.setPen(QPen(QColor("#b36a00"), 1.5))
            item.setFlag(QGraphicsRectItem.ItemIgnoresTransformations, True)
            item.setAcceptedMouseButtons(Qt.NoButton)
            item.setZValue(32)
            self._scene.addItem(item)
            self._handle_items[name] = item

    def _hit_handle(self, scene_pos):
        if self._selection_scene_rect is None:
            return None
        rect = self._selection_scene_rect
        scale = max(abs(self.transform().m11()), 1e-6)
        tolerance = 8.0 / scale
        x, y = scene_pos.x(), scene_pos.y()

        near_left = abs(x - rect.left()) <= tolerance and (
            rect.top() - tolerance <= y <= rect.bottom() + tolerance)
        near_right = abs(x - rect.right()) <= tolerance and (
            rect.top() - tolerance <= y <= rect.bottom() + tolerance)
        near_top = abs(y - rect.top()) <= tolerance and (
            rect.left() - tolerance <= x <= rect.right() + tolerance)
        near_bottom = abs(y - rect.bottom()) <= tolerance and (
            rect.left() - tolerance <= x <= rect.right() + tolerance)

        # 角优先，其次整条边都可拖，不再要求必须正好命中白色小手柄。
        if near_left and near_top:
            return "nw"
        if near_right and near_top:
            return "ne"
        if near_right and near_bottom:
            return "se"
        if near_left and near_bottom:
            return "sw"
        if near_top:
            return "n"
        if near_bottom:
            return "s"
        if near_left:
            return "w"
        if near_right:
            return "e"
        return None

    @staticmethod
    def _handle_cursor(handle):
        table = {
            "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
            "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
            "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
            "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
        }
        return QCursor(table.get(handle, Qt.CrossCursor))

    def _update_handle_positions(self):
        if self._selection_scene_rect is None:
            return
        for name, point in self._handle_positions(self._selection_scene_rect).items():
            item = self._handle_items.get(name)
            if item is not None:
                item.setPos(point)

    def _start_resize(self, handle, scene_pos):
        self._resize_handle = handle
        self._resize_original_rect = QRectF(self._selection_scene_rect)
        self._remove_rubber()
        self.setCursor(self._handle_cursor(handle))
        self._resize_to(scene_pos)

    def _resize_to(self, scene_pos):
        if self._resize_handle is None or self._resize_original_rect is None:
            return
        rect = QRectF(self._resize_original_rect)
        handle = self._resize_handle
        x = scene_pos.x()
        y = scene_pos.y()
        if "w" in handle:
            rect.setLeft(x)
        if "e" in handle:
            rect.setRight(x)
        if "n" in handle:
            rect.setTop(y)
        if "s" in handle:
            rect.setBottom(y)
        pw, ph = self._preview_hw
        rect = rect.normalized().intersected(QRectF(0, 0, pw, ph))
        min_size = 4.0 / max(abs(self.transform().m11()), 1e-6)
        if rect.width() < min_size or rect.height() < min_size:
            return
        self._selection_scene_rect = rect
        if self._box_item is not None:
            self._box_item.setRect(rect)
        self._update_handle_positions()

    def _finish_resize(self, scene_pos):
        if self._resize_handle is None:
            return
        self._resize_to(scene_pos)
        rect = QRectF(self._selection_scene_rect)
        self._resize_handle = None
        self._resize_original_rect = None
        p0 = self._scene_to_original(rect.topLeft())
        p1 = self._scene_to_original(rect.bottomRight())
        if p0 is None or p1 is None:
            return
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        self.setCursor(QCursor(Qt.CrossCursor))
        self.boxAdjusted.emit(x0, y0, x1, y1)

    def _remove_rubber(self):
        if self._rubber_item is not None:
            self._scene.removeItem(self._rubber_item)
            self._rubber_item = None

    def _start_box(self, scene_pos):
        self._remove_rubber()
        pen = QPen(QColor("#f0a000"), 2)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        self._rubber_item = self._scene.addRect(
            QRectF(scene_pos, scene_pos), pen, QBrush(QColor(240, 160, 0, 35)))
        self._rubber_item.setZValue(25)
        self._drag_start = scene_pos

    def _update_box(self, scene_pos):
        if self._rubber_item is None or self._drag_start is None:
            return
        self._rubber_item.setRect(
            QRectF(self._drag_start, scene_pos).normalized())

    def _finish_box(self, scene_pos):
        if self._rubber_item is None or self._drag_start is None:
            return
        rect = QRectF(self._drag_start, scene_pos).normalized()
        self._remove_rubber()
        self._drag_start = None
        if rect.width() * self.transform().m11() < 4 or rect.height() * self.transform().m11() < 4:
            return
        p0 = self._scene_to_original(rect.topLeft())
        p1 = self._scene_to_original(rect.bottomRight())
        if p0 is None or p1 is None:
            return
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        self.boxSelected.emit(x0, y0, x1, y1)

    def _emit_point(self, scene_pos, label):
        pt = self._scene_to_original(scene_pos)
        if pt is not None:
            self.pointSelected.emit(pt[0], pt[1], label)

    def wheelEvent(self, event):
        if not self.has_image:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current = self.transform().m11()
        if current * factor < 0.03 or current * factor > 80:
            return
        self.scale(factor, factor)
        self._fit_pending = False
        self.zoomChanged.emit()

    def mousePressEvent(self, event):
        if not self.has_image:
            return super().mousePressEvent(event)
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and
                (self._mode == MODE_PAN or self._space_pan)):
            self._pan_last = event.position().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton:
            if self._mode == MODE_BOX:
                handle = self._hit_handle(scene_pos)
                if handle is not None:
                    self._start_resize(handle, scene_pos)
                else:
                    self._start_box(scene_pos)
                event.accept()
                return
            if self._mode == MODE_POSITIVE:
                self._emit_point(scene_pos, 1)
                event.accept()
                return
            if self._mode == MODE_NEGATIVE:
                self._emit_point(scene_pos, 0)
                event.accept()
                return
            if self._mode == MODE_COLOR_SAMPLE:
                pt = self._scene_to_original(scene_pos)
                if pt is not None:
                    self.colorSampled.emit(pt[0], pt[1])
                event.accept()
                return
        if event.button() == Qt.RightButton and self._mode != MODE_PAN:
            self._emit_point(scene_pos, 0)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            delta = event.position().toPoint() - self._pan_last
            self._pan_last = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            self._fit_pending = False
            event.accept()
            return
        if self._drag_start is not None:
            self._update_box(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if self._resize_handle is not None:
            self._resize_to(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if self._mode == MODE_BOX:
            handle = self._hit_handle(self.mapToScene(event.position().toPoint()))
            self.setCursor(self._handle_cursor(handle) if handle else QCursor(Qt.CrossCursor))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(QCursor(
                Qt.OpenHandCursor if self._mode == MODE_PAN else Qt.CrossCursor))
            event.accept()
            return
        if self._resize_handle is not None and event.button() == Qt.LeftButton:
            self._finish_resize(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if self._drag_start is not None and event.button() == Qt.LeftButton:
            self._finish_box(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.setCursor(QCursor(Qt.OpenHandCursor))
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self.setCursor(QCursor(
                Qt.OpenHandCursor if self._mode == MODE_PAN else Qt.CrossCursor))
            event.accept()
            return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_pending:
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)
