from __future__ import annotations

import numpy as np
from PIL import Image

import generate_masks as gm
from interactive import pipeline


def test_compose_preview_and_fit_size():
    rgb = np.full((40, 60, 3), 100, np.uint8)
    preview = pipeline.make_preview_rgb(rgb, 30)
    assert preview.shape == (20, 30, 3)
    alpha = np.zeros((20, 30), np.float32)
    alpha[5:15, 10:20] = 1
    out = pipeline.compose_cutout_preview(preview, alpha)
    assert out.shape == (20, 30, 3)
    assert np.array_equal(out[7, 15], preview[7, 15])
    assert not np.array_equal(out[0, 0], preview[0, 0])


def test_sam_logits_are_treated_as_object_mask_not_probability():
    rgb = np.full((48, 64, 3), 110, np.uint8)
    logits = np.full((48, 64), -10.0, np.float32)
    # 很小的正 logit 在 sigmoid 下只有 0.5 左右，但它仍然属于对象内部；
    # 不能被当成半透明像素。
    logits[8:40, 10:54] = 0.01
    preview = pipeline.preview_alpha(logits, (24, 32))
    assert float(preview[12, 16]) > 0.99
    assert float(preview[0, 0]) < 0.01
    alpha, _info = pipeline.refine_sam_alpha(
        rgb, logits, work_size=64, use_matte=False)
    assert float(alpha[24, 32]) > 0.99
    assert float(alpha[0, 0]) < 0.01


def test_refine_does_not_use_unsafe_auto_cleaners(monkeypatch):
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 1] = 120
    logits = np.full((48, 64), -10.0, np.float32)
    logits[10:38, 12:52] = 10.0
    for name in ("clean_mask", "drop_specks", "tile_detail_alpha",
                 "bridge_gaps", "fill_enclosed_holes", "snap_flat_alpha"):
        monkeypatch.setattr(gm, name,
                            lambda *a, _n=name, **k: (_ for _ in ()).throw(
                                AssertionError("interactive pipeline called " + _n)))
    alpha, info = pipeline.refine_sam_alpha(rgb, logits, work_size=128, use_matte=False)
    assert alpha.shape == rgb.shape[:2]
    assert alpha.dtype == np.float32
    assert 0 < float(alpha[24, 32]) <= 1.0
    assert float(alpha[0, 0]) < 0.01
    assert info.coverage > 0.1


def test_export_interactive_outputs_png_and_mask_and_numbers_duplicates(tmp_path):
    src = tmp_path / "中文 源图.png"
    rgb = np.full((32, 40, 3), 180, np.uint8)
    rgb[..., 2] = 80
    exif = Image.Exif()
    exif[274] = 1
    Image.fromarray(rgb, "RGB").save(src, exif=exif)
    loaded, exif, icc, src_alpha = gm.load_image(str(src))
    logits = np.full((32, 40), -10.0, np.float32)
    logits[6:26, 8:32] = 10.0

    first = pipeline.export_interactive(
        str(src), str(tmp_path / "out"), loaded, exif, icc, src_alpha,
        logits, work_size=64, use_matte=False)
    second = pipeline.export_interactive(
        str(src), str(tmp_path / "out"), loaded, exif, icc, src_alpha,
        logits, work_size=64, use_matte=False)

    assert first["fg_path"] != second["fg_path"]
    assert first["mask_path"] != second["mask_path"]
    assert second["fg_path"].endswith("_fg_002.png")
    assert second["mask_path"].endswith("_fg_002_mask.png")
    with Image.open(first["fg_path"]) as im:
        assert im.mode == "RGBA"
        assert im.size == (40, 32)
        assert im.getexif().get(274) == 1
        alpha = np.array(im.getchannel("A"))
        assert alpha[15, 20] > 200
        assert alpha[0, 0] == 0
    with Image.open(first["mask_path"]) as im:
        m = np.array(im)
        assert set(np.unique(m)).issubset({0, 255})
        assert m[15, 20] == 255
        assert m[0, 0] == 0
    with Image.open(second["fg_path"]) as im:
        assert im.size == (40, 32)
