from __future__ import annotations

import numpy as np

from interactive.color_assist import color_similarity, suggest_points


def test_color_similarity_prefers_sampled_color():
    rgb = np.array([[[20, 150, 30], [180, 30, 30]]], dtype=np.uint8)
    score = color_similarity(rgb, [[20, 150, 30]])
    assert score[0, 0] > score[0, 1]


def test_suggest_points_is_limited_and_stays_inside_box():
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    rgb[20:80, 20:100] = (20, 150, 30)
    points = suggest_points(
        rgb, [[20, 150, 30]], (20, 20, 100, 80),
        max_points=3, min_spacing=20)
    assert 1 <= len(points) <= 3
    assert all(20 <= x < 100 and 20 <= y < 80 for x, y, _ in points)
