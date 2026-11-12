# -*- coding: utf-8 -*-
"""SAM 提示结果到快速预览、原尺寸精修和公共导出的桥接层。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

import generate_masks as gm


class EmptyMaskError(RuntimeError):
    pass


@dataclass(frozen=True)
class RefineInfo:
    work_hw: tuple[int, int]
    coverage: float


def fit_size(width, height, longest_side):
    if max(width, height) <= longest_side:
        return int(width), int(height)
    scale = longest_side / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def make_preview_rgb(rgb, longest_side=1800):
    h, w = rgb.shape[:2]
    pw, ph = fit_size(w, h, longest_side)
    if (pw, ph) == (w, h):
        return np.ascontiguousarray(rgb)
    return np.ascontiguousarray(
        cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA))


def preview_alpha(logits, preview_hw):
    # SAM logits 不是标定透明度，只作为对象蒙版使用。缩小时用双线性插值
    # 给边界留 1px 抗锯齿，内部保持完全不透明。
    binary = (np.asarray(logits, np.float32) > 0.0).astype(np.float32)
    ph, pw = preview_hw
    if binary.shape[:2] != (ph, pw):
        binary = cv2.resize(binary, (pw, ph), interpolation=cv2.INTER_LINEAR)
    return np.clip(binary, 0.0, 1.0).astype(np.float32)


def compose_cutout_preview(rgb_preview, alpha_preview, show_original=False):
    """棋盘格 + 当前蒙版的预览；只在主线程做显示合成。"""
    if alpha_preview is None or show_original:
        return np.ascontiguousarray(rgb_preview)
    h, w = rgb_preview.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 18) + (xx // 18)) % 2 == 0, 232, 205).astype(np.float32)
    checker = np.repeat(checker[..., None], 3, axis=2)
    a = np.clip(np.asarray(alpha_preview, np.float32), 0.0, 1.0)[..., None]
    out = rgb_preview.astype(np.float32) * a + checker * (1.0 - a)
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def refine_sam_alpha(rgb, logits, work_size=1536, use_matte=True, contrast=0.10):
    """SAM 只决定对象语义；这里用非学习 matting/引导滤波恢复边缘。

    SAM 的 logits 在物体内部也可能只有很小的正数，不能当 alpha 概率。
    先将它二值化为对象蒙版，再只在实际边界附近做 matting。
    """
    h, w = rgb.shape[:2]
    if logits.ndim != 2 or logits.size == 0:
        raise ValueError("SAM logits 形状异常")
    binary = (np.asarray(logits, np.float32) > 0.0).astype(np.float32)
    if binary.mean() <= 1e-7:
        raise EmptyMaskError("蒙版为空，请重新框选目标或增加正点")
    ww, hh = fit_size(w, h, max(1, int(work_size)))
    work = rgb if (ww, hh) == (w, h) else cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA)
    a_small = cv2.resize(binary, (ww, hh), interpolation=cv2.INTER_LINEAR)
    a_small = np.clip(a_small, 0.0, 1.0).astype(np.float32)

    if use_matte:
        a8 = (a_small * 255.0).round().astype(np.uint8)
        band = max(2, int(round(max(hh, ww) / 512)))
        a_matte = gm.closed_form_matte(work, a8, band=band)
        # 内部沿用二值对象蒙版，边界交给 matting；避免 SAM 的偏低 logits
        # 把整株植物变半透明。
        a_small = np.clip(0.15 * a_small + 0.85 * a_matte, 0.0, 1.0)

    scale = max(h, w) / float(max(hh, ww))
    alpha_full = gm.upsample_alpha(
        a_small, rgb, scale, s=2, contrast=max(0.0, float(contrast)))
    alpha_full = np.clip(alpha_full, 0.0, 1.0)
    coverage = float((alpha_full > 0.5).mean())
    if coverage <= 1e-7:
        raise EmptyMaskError("蒙版为空，请重新框选目标或增加正点")
    return alpha_full.astype(np.float32), RefineInfo((hh, ww), coverage)


def unique_output_paths(out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem + "_fg")
    mask = base + "_mask.png"
    fg = base + ".png"
    n = 2
    while os.path.exists(fg) or os.path.exists(mask) or os.path.exists(base + "_camera.json"):
        fg = "%s_%03d.png" % (base, n)
        mask = "%s_%03d_mask.png" % (base, n)
        n += 1
    return fg, mask


def export_interactive(source_path, out_dir, rgb, exif, icc, src_alpha,
                       selected_logits, work_size=1536, use_matte=True):
    """导出当前提示 revision 的结果，返回公共导出函数的结果。"""
    alpha, info = refine_sam_alpha(
        rgb, selected_logits, work_size=work_size, use_matte=use_matte)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    fg_path, mask_path = unique_output_paths(out_dir, stem)
    options = gm.ExportOptions(
        out_type="both",
        suffix="_fg",
        ext=".png",
        expand=0.0,
        feather=0.0,
        decon=True,
        work_size=max(1, int(work_size)),
        mask_binary=True,
        alpha_threshold=0.5,
        bg_fill="bleed",
        png_level=6,
        keep_exif=True,
        camera_json=False,
        drop_mask_specks=False,
    )
    result = gm.export_alpha(
        source_path, out_dir, rgb, exif, icc, src_alpha, alpha, options,
        fg_path=fg_path, mask_path=mask_path)
    result["refine"] = info
    return result
