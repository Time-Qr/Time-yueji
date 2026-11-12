from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF

from interactive.canvas import ImageCanvas, numpy_rgb_to_qimage
from interactive.session import PromptPoint


def test_canvas_resize_handles_expand_existing_box(qtbot):
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    canvas.show()
    rgb = np.zeros((80, 100, 3), np.uint8)
    canvas.set_image(numpy_rgb_to_qimage(rgb), (80, 100))
    canvas.set_selection((10, 10, 50, 40))
    assert len(canvas._handle_items) == 8
    assert canvas._hit_handle(QPointF(50, 40)) == "se"
    # 底边中间偏左、没有白色小手柄的位置也必须能拖。
    assert canvas._hit_handle(QPointF(20, 40)) == "s"
    assert canvas._hit_handle(QPointF(50, 20)) == "e"
    with qtbot.waitSignal(canvas.boxAdjusted, timeout=1000) as blocker:
        canvas._start_resize("se", QPointF(50, 40))
        canvas._resize_to(QPointF(80, 60))
        canvas._finish_resize(QPointF(80, 60))
    x0, y0, x1, y1 = blocker.args
    assert round(x0) == 10 and round(y0) == 10
    assert round(x1) == 80 and round(y1) == 60


def test_canvas_coordinates_and_selection_preserved_on_image_update(qtbot):
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    canvas.show()
    rgb = np.zeros((80, 100, 3), np.uint8)
    rgb[..., 0] = 120
    canvas.set_image(numpy_rgb_to_qimage(rgb), (800, 1000))

    pt = canvas._scene_to_original(QPointF(50, 40))
    assert pt == (500.0, 400.0)
    canvas.set_selection((100, 100, 500, 500), [PromptPoint(200, 200, 1)])
    assert canvas._box_item is not None
    assert len(canvas._point_items) == 1
    assert len(canvas._point_label_items) == 1

    canvas.update_image(numpy_rgb_to_qimage(rgb))
    assert len(canvas._point_items) == 1
    assert canvas._box_item is not None


def test_canvas_coordinates_use_width_for_x_and_height_for_y(qtbot):
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    canvas.show()
    rgb = np.zeros((120, 200, 3), np.uint8)
    canvas.set_image(numpy_rgb_to_qimage(rgb), (6000, 4000))

    original = canvas._scene_to_original(QPointF(100, 60))
    assert original == (2000.0, 3000.0)

    scene = canvas._original_to_scene(2000, 3000)
    assert scene == QPointF(100, 60)


def test_canvas_numbers_positive_and_negative_points(qtbot):
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    canvas.show()
    rgb = np.zeros((80, 100, 3), np.uint8)
    canvas.set_image(numpy_rgb_to_qimage(rgb), (80, 100))
    canvas.set_selection(
        points=[
            PromptPoint(10, 10, 1),
            PromptPoint(20, 20, 1),
            PromptPoint(30, 30, 0),
        ])
    assert [item.text() for item in canvas._point_label_items] == ["正1", "正2", "负1"]
