#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_masks.py — 批量抠图（去背景）增强版 v2
=====================================================

相对原版的改进（详见 README_优化说明.md）：
  1. 边缘：模型粗分割 → 高分辨引导滤波上采样 + 封闭形式 alpha matting，
     不再使用「二值化 + 开运算 + 与软 alpha 相与」这类会吃掉边缘的操作。
  2. 背景：连通域按“相对显著度 + 距离”清理，保留孤立的叶片/顶芽；
     色彩去污（unmix）消除边缘背景色残留。
  3. 格式：新增 DNG / CR2 / NEF / ARW / RAF / RW2 / ORF / PEF / SRW 等 RAW
     （rawpy 解码），以及 16 位 PNG/TIFF、带 alpha 的 PNG、WebP、BMP。
  4. 细节：不做腐蚀；只填「不与图像边界连通」的封闭小孔，
     主体内部的叶片、茎干、泥沙纹理全部保留。

用法示例：
    python generate_masks.py --input input --output output
    python generate_masks.py --quality best --type both --workers 4
    python generate_masks.py --input 测试用图片 --output out --suffix _cut

"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass

import numpy as np
import cv2

# --------------------------------------------------------------------------
# 可选依赖
# --------------------------------------------------------------------------
try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:                                    # pragma: no cover
    Image = None
    _HAS_PIL = False

try:
    import pymatting
    _HAS_PYMATTING = True
except Exception:
    _HAS_PYMATTING = False

try:
    import piexif
    _HAS_PIEXIF = True
except Exception:
    _HAS_PIEXIF = False

try:
    import exifread
    _HAS_EXIFREAD = True
except Exception:
    _HAS_EXIFREAD = False

try:
    import rawpy
    _HAS_RAWPY = True
except Exception:
    _HAS_RAWPY = False


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
RAW_EXTS = {
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf",
    ".pef", ".srw", ".3fr", ".fff", ".iiq", ".kdc", ".mef", ".mos",
    ".mrw", ".x3f", ".raw", ".erf", ".dcr", ".rwl",
}
IMG_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".tif", ".tiff",
    ".webp", ".ppm", ".pgm", ".tga", ".heic", ".heif", ".avif",
}
SUPPORTED_EXTS = RAW_EXTS | IMG_EXTS

MODEL_PREFERENCE = [
    "birefnet-general",
    "birefnet-general-lite",
    "isnet-general-use",
    "u2net",
]

QUALITY_PRESETS = {
    # name:      (work_size, matting, gf_subsample)
    "fast":       (1024, False, 4),
    "balanced":   (1536, True, 2),
    "best":       (2048, True, 1),
}


# --------------------------------------------------------------------------
# 图像读取
# --------------------------------------------------------------------------
def _normalize_to_u8(arr):
    """把任意数值范围的数组稳健地映射到 0-255 uint8。"""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi - lo < 1e-6:
        return np.zeros(a.shape, np.uint8)
    a = (a - lo) * (255.0 / (hi - lo))
    return np.clip(a, 0, 255).astype(np.uint8)


def load_image(path):
    """读取图像 -> (rgb uint8 HxWx3, exif bytes|None, icc bytes|None, src_alpha|None)

    * RAW：rawpy 解码为 sRGB 8bit；
    * 16 位 PNG/TIFF：按 0.5%~99.5% 分位拉伸；
    * 带 alpha 的 PNG/WebP：返回原 alpha，最终与生成 alpha 相乘；
    * EXIF 方向：先按 Orientation 摆正，保存时把 Orientation 置 1。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in RAW_EXTS:
        if not _HAS_RAWPY:
            raise RuntimeError("处理 RAW 需要 rawpy：pip install rawpy")
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
                output_color=rawpy.ColorSpace.sRGB,
                user_flip=None,
            )
        info = camera_info(path)
        exif = _exif_bytes_from_info(info, rgb.shape[1], rgb.shape[0])
        return np.ascontiguousarray(rgb), exif, None, None

    if not _HAS_PIL:
        raise RuntimeError("需要 Pillow：pip install pillow")

    with Image.open(path) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        src_alpha = None

        # 所有非 RAW 分支都先按 EXIF 方向摆正，保证预览、提示坐标与导出像素
        # 使用同一个坐标系。16 位和带透明通道的图片以前在这里提前返回，
        # 会让 GUI 坐标和最终输出方向不一致。
        im = ImageOps.exif_transpose(im)

        # 16 位 / 浮点灰度、多页 TIFF 等
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = _normalize_to_u8(np.array(im))
            rgb = np.dstack([arr] * 3)
            return rgb, exif, icc, None

        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            arr = np.array(im)
            src_alpha = arr[:, :, 3]
            rgb = arr[:, :, :3]
            return np.ascontiguousarray(rgb), exif, icc, src_alpha

        rgb = np.array(im.convert("RGB"))
        return np.ascontiguousarray(rgb), exif, icc, None


def _tag_value(tags, *keys):
    for k in keys:
        v = tags.get(k)
        if v is not None:
            return v
    return None


def _tag_float(tags, *keys):
    v = _tag_value(tags, *keys)
    if v is None:
        return None
    try:
        val = v.values[0]
        num, den = getattr(val, "num", None), getattr(val, "den", None)
        if num is not None and den:
            return float(num) / float(den)
        return float(val)
    except Exception:
        return None


def _raw_tags(path):
    if not _HAS_EXIFREAD:
        return {}
    try:
        with open(path, "rb") as f:
            return exifread.process_file(f, details=False)
    except Exception:
        return {}


def camera_info(path, exif_bytes=None):
    """统一抽取相机内参信息（JPEG EXIF / RAW TIFF 标签）。

    三维重建拍完照片后最需要的是焦距、机型与画幅信息；这里既用于写回图片，
    也用于生成 *_camera.json 侧车文件。
    """
    ext = os.path.splitext(path)[1].lower()
    info = {"make": None, "model": None, "lens": None, "datetime": None,
            "focal_mm": None, "focal35_mm": None, "fnumber": None, "iso": None,
            "orientation": 1}

    if ext in RAW_EXTS:
        tags = _raw_tags(path)
        if not tags:
            return info
        info["make"] = str(tags["Image Make"]) if "Image Make" in tags else None
        info["model"] = str(tags["Image Model"]) if "Image Model" in tags else None
        lens = _tag_value(tags, "EXIF LensModel", "EXIF LensSpecification")
        if lens is not None:
            info["lens"] = str(lens)
        dt = _tag_value(tags, "EXIF DateTimeOriginal", "Image DateTime")
        info["datetime"] = str(dt) if dt is not None else None
        info["focal_mm"] = _tag_float(tags, "EXIF FocalLength", "Image FocalLength")
        info["focal35_mm"] = _tag_float(tags, "EXIF FocalLengthIn35mmFilm")
        info["fnumber"] = _tag_float(tags, "EXIF FNumber", "Image FNumber")
        iso = _tag_value(tags, "EXIF ISOSpeedRatings", "Image ISOSpeedRatings")
        if iso is not None:
            try:
                info["iso"] = int(iso.values[0])
            except Exception:
                pass
        return info

    if exif_bytes and _HAS_PIEXIF:
        try:
            d = piexif.load(exif_bytes)
        except Exception:
            return info
        z, e = d.get("0th", {}), d.get("Exif", {})

        def _s(v):
            if v is None:
                return None
            try:
                return v.decode("utf-8", "ignore").strip("\x00 ").strip()
            except Exception:
                return str(v).strip("\x00 ").strip()

        def _r(v):
            if v is None:
                return None
            try:
                if isinstance(v, tuple) and len(v) == 2:
                    return float(v[0]) / float(v[1]) if v[1] else None
                return float(v)
            except Exception:
                return None

        info["make"] = _s(z.get(piexif.ImageIFD.Make))
        info["model"] = _s(z.get(piexif.ImageIFD.Model))
        info["lens"] = _s(e.get(piexif.ExifIFD.LensModel))
        info["datetime"] = _s(e.get(piexif.ExifIFD.DateTimeOriginal)) or _s(z.get(piexif.ImageIFD.DateTime))
        info["focal_mm"] = _r(e.get(piexif.ExifIFD.FocalLength))
        info["focal35_mm"] = _r(e.get(piexif.ExifIFD.FocalLengthIn35mmFilm))
        info["fnumber"] = _r(e.get(piexif.ExifIFD.FNumber))
        try:
            iso = e.get(piexif.ExifIFD.ISOSpeedRatings)
            info["iso"] = int(iso) if iso is not None else None
        except Exception:
            pass
        try:
            info["orientation"] = int(z.get(piexif.ImageIFD.Orientation, 1) or 1)
        except Exception:
            info["orientation"] = 1
    return info


def _exif_bytes_from_info(info, width=None, height=None):
    """用 camera_info 组装一份 EXIF（主要给 RAW 用：DNG 没有可直接复制的 EXIF 段）。"""
    if not _HAS_PIEXIF:
        return None

    def _rat(x):
        return (int(round(float(x) * 10000)), 10000)

    zeroth, exif_ifd = {}, {}
    if info.get("make"):
        zeroth[piexif.ImageIFD.Make] = str(info["make"]).encode()
    if info.get("model"):
        zeroth[piexif.ImageIFD.Model] = str(info["model"]).encode()
    zeroth[piexif.ImageIFD.Orientation] = 1
    if width and height:
        exif_ifd[piexif.ExifIFD.PixelXDimension] = int(width)
        exif_ifd[piexif.ExifIFD.PixelYDimension] = int(height)
    if info.get("datetime"):
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = str(info["datetime"]).encode()
    if info.get("focal_mm"):
        exif_ifd[piexif.ExifIFD.FocalLength] = _rat(info["focal_mm"])
    if info.get("focal35_mm"):
        exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(info["focal35_mm"]))
    if info.get("fnumber"):
        exif_ifd[piexif.ExifIFD.FNumber] = _rat(info["fnumber"])
    if info.get("iso"):
        exif_ifd[piexif.ExifIFD.ISOSpeedRatings] = int(info["iso"])
    if info.get("lens"):
        exif_ifd[piexif.ExifIFD.LensModel] = str(info["lens"]).encode()
    if not zeroth and not exif_ifd:
        return None
    try:
        return piexif.dump({"0th": zeroth, "Exif": exif_ifd, "1st": {}, "GPS": {}, "Interop": {}})
    except Exception:
        return None


def intrinsics_json(info, width, height):
    """估算像素焦距（供 Metashape/COLMAP 等直接使用）。"""
    out = dict(info)
    out["width"] = int(width)
    out["height"] = int(height)
    out["cx"] = width / 2.0
    out["cy"] = height / 2.0
    out["focal_px"] = None
    out["sensor_width_mm"] = 36.0
    if info.get("focal35_mm"):
        # 35mm 等效焦距：f_px = width * f35 / 36
        out["focal_px"] = round(width * float(info["focal35_mm"]) / 36.0, 3)
    elif info.get("focal_mm") and info.get("focal35_mm"):
        pass
    return out


def _exif_with_orientation_reset(exif_bytes):
    """EXIF 方向已烧进像素，写回时把 Orientation 置 1，避免看图软件二次旋转。"""
    if not exif_bytes or not _HAS_PIEXIF:
        return exif_bytes
    try:
        data = piexif.load(exif_bytes)
        data["0th"][piexif.ImageIFD.Orientation] = 1
        return piexif.dump(data)
    except Exception:
        return exif_bytes


# --------------------------------------------------------------------------
# 抠图模型
# --------------------------------------------------------------------------
_SESSION = None
_SESSION_NAME = None


def resolve_model(name, model_dir):
    """auto：按“效果优先”顺序挑一个本地已存在的模型。"""
    if name and name != "auto":
        return name
    for m in MODEL_PREFERENCE:
        if os.path.exists(os.path.join(model_dir, m + ".onnx")):
            return m
    return "isnet-general-use"


class OnnxAlphaModel:
    """直接加载 BiRefNet 风格 ONNX（输入 input_image 1x3x1024x1024）。

    这样既能用 rembg 官方模型，也能用 HuggingFace(onnx-community) 导出的模型，
    而且省掉 rembg 的联网校验。
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path, threads=0, mem_arena=True, letterbox=False):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = bool(mem_arena)
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.sess = ort.InferenceSession(path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = (int(shape[2]), int(shape[3])) if len(shape) == 4 else (1024, 1024)
        self.letterbox = letterbox
        # BiRefNet_HR-matting 这类导出的输出名是 alpha，且已经过 sigmoid（0~1）；
        # 官方 general 版输出 logits（output_image），需要自己 sigmoid。
        self.activated = self.sess.get_outputs()[0].name.lower() == "alpha"

    def _prep(self, rgb):
        H, W = self.size
        h, w = rgb.shape[:2]
        if self.letterbox:
            # 等比缩放 + 居中补边，避免把主体压扁
            r = min(W / float(w), H / float(h))
            nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
            canvas = np.zeros((H, W, 3), np.uint8)
            ox, oy = (W - nw) // 2, (H - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized
            meta = (ox, oy, nw, nh)
            src = canvas
        else:
            src = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            meta = (0, 0, W, H)
        x = src.astype(np.float32) * (1.0 / 255.0)
        x = (x - self.MEAN) / self.STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), meta

    def predict(self, rgb, norm=None):
        """norm=None 按本张结果 min-max 拉伸（粗分割用，记为 last_range）；
        norm=(mn,mx) 沿用同一区间（分块精修用，保证和粗结果同一把尺子）；
        norm=False 直接用 sigmoid 原值。"""
        h, w = rgb.shape[:2]
        x, (ox, oy, nw, nh) = self._prep(rgb)
        out = self.sess.run(None, {self.input_name: x})[0][0, 0]
        if not self.activated:
            out = 1.0 / (1.0 + np.exp(-out))
        out = np.clip(out, 0.0, 1.0)
        if norm is False:
            mn, mx = 0.0, 1.0
        elif norm is None:
            mn, mx = float(out.min()), float(out.max())
            self.last_range = (mn, mx)
        else:
            mn, mx = float(norm[0]), float(norm[1])
        out = (out - mn) / (mx - mn + 1e-8)
        mask = (np.clip(out, 0, 1) * 255.0).astype(np.uint8)
        if self.letterbox:
            mask = mask[oy:oy + nh, ox:ox + nw]
        if (mask.shape[1], mask.shape[0]) != (w, h):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask


def _make_session(name, threads=0, mem_arena=True):
    """直接构造 rembg 的 session，以便控制线程数与 ONNX 内存池（大图批处理更省内存）。"""
    import onnxruntime as ort
    from rembg.session_factory import sessions_class

    so = ort.SessionOptions()
    if threads and threads > 0:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = bool(mem_arena)
    for sc in sessions_class:
        if sc.name() == name:
            return sc(name, so)
    from rembg import new_session
    return new_session(name)


def get_session(model, model_dir, threads=0, mem_arena=True):
    global _SESSION, _SESSION_NAME
    if _SESSION is not None and _SESSION_NAME == model:
        return _SESSION
    os.environ["U2NET_HOME"] = os.path.abspath(model_dir)
    os.environ["REMBG_HOME"] = os.path.abspath(model_dir)

    candidates = [model]
    if model in MODEL_PREFERENCE:
        candidates += [m for m in MODEL_PREFERENCE if m != model]
    last_err = None
    for name in candidates:
        local = os.path.join(os.path.abspath(model_dir), name + ".onnx")
        try:
            if os.path.exists(local) and _is_birefnet_onnx(local):
                _SESSION = OnnxAlphaModel(local, threads, mem_arena)
            else:
                _SESSION = _make_session(name, threads, mem_arena)
            _SESSION_NAME = name
            if name != model:
                print("[提示] 模型 %s 不可用（%s），已自动改用 %s" % (model, last_err, name))
            return _SESSION
        except Exception as e:                      # 模型文件损坏 / 下载失败
            last_err = "%s: %s" % (type(e).__name__, str(e)[:120])
            continue
    raise RuntimeError("无法加载任何抠图模型：%s" % last_err)


def _is_birefnet_onnx(path):
    """判断本地 onnx 是否是 BiRefNet 风格（输入名 input_image / image，4 维）。"""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        return inp.name in ("input_image", "image") and len(inp.shape) == 4
    except Exception:
        return False


def model_alpha(rgb, session, norm=None):
    """模型输出 0-255 的 alpha（灰度）。支持本地 ONNX 与 rembg 会话。"""
    if isinstance(session, OnnxAlphaModel):
        return session.predict(rgb, norm=norm)
    from rembg import remove
    mask = remove(rgb, session=session, only_mask=True)
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


# --------------------------------------------------------------------------
# 蒙版清理
# --------------------------------------------------------------------------
def clean_mask(alpha,
               rgb=None,
               min_area_frac=2e-5,
               significant_frac=0.01,
               max_dist_frac=0.6,
               fill_frac=0.02):
    """去掉背景杂点、补齐主体内部的封闭小孔。

    规则：
      * 只保留“面积够大”或“离主体足够近”的连通域 —— 既不会留下远处杂点，
        也不会误删与主体分离的小叶片（这类叶片通常就在主体附近）；
      * 只填“不与图像边界连通”的孔洞 —— 叶隙、枝条间的真实背景缝隙保持透明。
    """
    h, w = alpha.shape
    m = (alpha > 127).astype(np.uint8)
    total = int(m.sum())
    if total == 0:
        return alpha

    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8, cv2.CV_32S)
    areas = stats[:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas[1:]))
    diag = float(np.hypot(h, w))
    min_area = max(16.0, min_area_frac * alpha.size)
    keep = np.zeros(n, bool)
    keep[main] = True
    for i in range(1, n):
        if i == main:
            continue
        if areas[i] < min_area:
            continue
        d = float(np.hypot(cents[i][0] - cents[main][0], cents[i][1] - cents[main][1]))
        if areas[i] >= significant_frac * areas[main] and d < max_dist_frac * diag:
            keep[i] = True
    alpha = np.where(keep[lab], alpha, 0).astype(np.uint8)

    # 填封闭孔洞
    inv = (alpha <= 127).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, 4, cv2.CV_32S)
    subj = max(1, int((alpha > 127).sum()))
    limit = max(16.0, fill_frac * subj)
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    for i in range(1, n2):
        x, y, ww, hh, area = stats2[i]
        if x <= 0 or y <= 0 or x + ww >= w or y + hh >= h:
            continue                                  # 连通到图像边界 -> 真实背景
        if area > limit:
            continue
        if rgb is not None:
            # 如果孔洞颜色和周边背景几乎一样，说明那是叶片之间的真实缝隙，保持透明
            hole = (lab2 == i)
            ring = cv2.dilate(hole.astype(np.uint8), k_ring).astype(bool) & ~hole
            ring &= (alpha <= 127)
            if ring.sum() >= 30:
                d = np.linalg.norm(rgb[hole].mean(axis=0) - rgb[ring].mean(axis=0))
                if d < 32.0:
                    continue
        alpha[lab2 == i] = 255

    # 去掉清理后仍残留的极小碎块
    m2 = (alpha > 127).astype(np.uint8)
    n3, lab3, stats3, _ = cv2.connectedComponentsWithStats(m2, 8, cv2.CV_32S)
    if n3 > 1:
        a3 = stats3[:, cv2.CC_STAT_AREA]
        biggest = int(a3[1:].max())
        drop = np.zeros(n3, bool)
        for i in range(1, n3):
            if a3[i] < max(12.0, 0.002 * biggest):
                drop[i] = True
        if drop.any():
            alpha = np.where(drop[lab3], 0, alpha).astype(np.uint8)

    return alpha


# --------------------------------------------------------------------------
# Alpha matting / 边缘精修
# --------------------------------------------------------------------------
def closed_form_matte(rgb, alpha_u8, band=3):
    """以模型 alpha 作 trimap，跑封闭形式 matting，恢复半透明的真实边缘。"""
    if not _HAS_PYMATTING:
        return alpha_u8.astype(np.float32) / 255.0
    a = alpha_u8.astype(np.float32) / 255.0
    fg = (a > 0.95).astype(np.uint8)
    bg = (a < 0.05).astype(np.uint8)
    unk = ((1 - fg) & (1 - bg)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    unk = cv2.dilate(unk, k) & (1 - cv2.erode(fg, k))
    trimap = np.where(fg > 0, 1.0, np.where(bg > 0, 0.0, 0.5)).astype(np.float64)
    trimap[unk > 0] = 0.5
    if (trimap == 0.5).sum() == 0 or (trimap == 1.0).sum() == 0:
        return a
    try:
        out = pymatting.estimate_alpha_cf(rgb.astype(np.float64) / 255.0, trimap)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        if os.environ.get("REMBG_DEBUG"):
            print("[警告] matting 失败，回退模型 alpha：%s" % e, flush=True)
        return a


def sharpen_alpha(a, width=0.20):
    """把上采样后偏宽的过渡带收窄成 1~2px 的清晰边缘（smoothstep 对比度拉伸）。

    width 是过渡带半宽：越小边缘越锐，过大则会出现锯齿；0 表示不处理。
    """
    if width <= 0:
        return a
    lo, hi = 0.5 - width, 0.5 + width
    x = np.clip((a - lo) * (1.0 / (hi - lo)), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def _inv3(C):
    a00 = C[..., 0, 0]; a01 = C[..., 0, 1]; a02 = C[..., 0, 2]
    a11 = C[..., 1, 1]; a12 = C[..., 1, 2]; a22 = C[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    inv = np.empty_like(C)
    inv[..., 0, 0] = c00 / det
    inv[..., 0, 1] = inv[..., 1, 0] = c01 / det
    inv[..., 0, 2] = inv[..., 2, 0] = c02 / det
    inv[..., 1, 1] = (a00 * a22 - a02 * a02) / det
    inv[..., 1, 2] = inv[..., 2, 1] = (a02 * a01 - a00 * a12) / det
    inv[..., 2, 2] = (a00 * a11 - a01 * a01) / det
    return inv


def guided_filter_fast(I, p, r, eps=1e-4, s=2):
    """He 等人快速引导滤波：I 为彩色引导图，p 为待滤波的 alpha。"""
    h, w = p.shape
    if s > 1 and min(h, w) > 4 * s:
        hs, ws = max(1, h // s), max(1, w // s)
        I_l = cv2.resize(I, (ws, hs), interpolation=cv2.INTER_AREA)
        p_l = cv2.resize(p, (ws, hs), interpolation=cv2.INTER_AREA)
        r_l = max(1, int(round(r / s)))
    else:
        I_l, p_l, r_l, s = I, p, r, 1

    mI = _box(I_l, r_l)
    mp = _box(p_l, r_l)
    cov_Ip = _box(I_l * p_l[..., None], r_l) - mI * mp[..., None]
    mII = np.stack([_box(I_l[..., i] * I_l[..., j], r_l) for i in range(3) for j in range(i, 3)], -1)
    C = np.empty(I_l.shape[:2] + (3, 3), np.float32)
    for i, j, k in ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4), (2, 2, 5)):
        v = mII[..., k] - mI[..., i] * mI[..., j]
        C[..., i, j] = v
        C[..., j, i] = v
    for i in range(3):
        C[..., i, i] += eps
    a = np.einsum("...ij,...j->...i", _inv3(C), cov_Ip)
    b = mp - np.einsum("...i,...i->...", a, mI)

    if s > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.einsum("...i,...i->...", _box(a, r), I) + _box(b, r)


def upsample_alpha(alpha_low, rgb_full, scale, radius=None, eps=1e-4, s=2, tile=1536, contrast=0.0):
    """把低分辨率 alpha 边缘对齐地放大到原图分辨率（分块处理，控制内存）。"""
    H, W = rgb_full.shape[:2]
    if alpha_low.shape[:2] == (H, W):
        return np.clip(alpha_low.astype(np.float32), 0.0, 1.0)
    up = cv2.resize(alpha_low.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    np.clip(up, 0.0, 1.0, out=up)
    if radius is None:
        radius = int(np.clip(round(scale * 1.6), 4, 32))
    out = np.empty_like(up)
    pad = radius + 4
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1 = min(y0 + tile, H); x1 = min(x0 + tile, W)
            ys = max(0, y0 - pad); xs = max(0, x0 - pad)
            ye = min(H, y1 + pad); xe = min(W, x1 + pad)
            # 只在本块内转浮点，避免整张大图产生 float32 副本
            sub_I = np.ascontiguousarray(rgb_full[ys:ye, xs:xe], dtype=np.float32)
            sub_I *= (1.0 / 255.0)
            sub_p = np.ascontiguousarray(up[ys:ye, xs:xe])
            q = guided_filter_fast(sub_I, sub_p, radius, eps, s)
            q = q[y0 - ys:y0 - ys + (y1 - y0), x0 - xs:x0 - xs + (x1 - x0)]
            np.clip(q, 0.0, 1.0, out=q)
            if contrast and contrast > 0:
                q = sharpen_alpha(q, contrast)
            out[y0:y1, x0:x1] = q
    np.clip(out, 0.0, 1.0, out=out)
    return out


def decontaminate(rgb, alpha, radius, strength=1.0, tile=1024, dst=None):
    """色彩去污：把半透明像素里混进来的背景色“反解”出去，消除彩色边缘。

    分块处理，48MP 级别的大图也不会吃满内存；dst 可直接指向输出 RGBA 的 RGB 部分。
    """
    H, W = rgb.shape[:2]
    out = dst if dst is not None else np.empty_like(rgb)
    pad = radius + 4
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1 = min(y0 + tile, H); x1 = min(x0 + tile, W)
            ys = max(0, y0 - pad); xs = max(0, x0 - pad)
            ye = min(H, y1 + pad); xe = min(W, x1 + pad)

            a = np.ascontiguousarray(alpha[ys:ye, xs:xe])
            a = a.astype(np.float32) * (1.0 / 255.0) if a.dtype == np.uint8 else a.astype(np.float32)
            img = np.ascontiguousarray(rgb[ys:ye, xs:xe], dtype=np.float32)
            img *= (1.0 / 255.0)
            wb = 1.0 - a

            den_b = _box(wb, radius)
            B = np.stack([_box(img[..., c] * wb, radius) / np.maximum(den_b, 1e-3) for c in range(3)], -1)
            den_f = _box(a, radius)
            F = np.stack([_box(img[..., c] * a, radius) / np.maximum(den_f, 1e-3) for c in range(3)], -1)

            aa = np.clip(a, 0.15, 1.0)[..., None]
            unmix = np.clip((img - (1.0 - aa) * B) / aa, 0.0, 1.0)

            trust = (np.abs(F - B).sum(-1) > 0.25)
            wgt = np.clip((0.98 - a) / 0.45, 0.0, 1.0) * np.clip(a / 0.05, 0.0, 1.0) \
                  * trust.astype(np.float32) * strength
            wgt = wgt[..., None]
            res = img * (1.0 - wgt) + unmix * wgt
            res = np.clip(res, 0.0, 1.0) * 255.0
            out[y0:y1, x0:x1] = res[y0 - ys:y0 - ys + (y1 - y0),
                                    x0 - xs:x0 - xs + (x1 - x0)].round().astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# 透明区 RGB 填充（决定 PNG 体积的关键一步）
# --------------------------------------------------------------------------
def bridge_gaps(alpha, rgb=None, max_width=24, color_tol=45.0):
    """把窄缝（≤max_width 像素）里、颜色又和旁边前景接近的地方补成前景。

    细枝在 alpha 上卡在 0.4~0.6，二值化时正好断开。真正的叶子间隙通常几十上百像素宽，
    用这么小的闭运算补不到，所以不会把叶片之间的黑缝糊上。
    """
    H, W = alpha.shape
    ds = 1
    if max(H, W) > 3000:
        ds = 2
    a_s = alpha[::ds, ::ds] if ds > 1 else alpha
    bw = (a_s >= 0.5).astype(np.uint8)
    if bw.all() or not bw.any():
        return alpha
    k = int(max_width // 2) * 2 + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se)
    add = (closed > 0) & (bw == 0)
    if not add.any():
        return alpha
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    add &= dist <= max_width / 2.0
    if rgb is not None and add.any():
        r_s = rgb[::ds, ::ds] if ds > 1 else rgb
        fg = bw.astype(np.float32)
        den = cv2.blur(fg, (k, k))
        nc = np.stack([cv2.blur(r_s[:, :, c].astype(np.float32) * fg, (k, k)) for c in range(3)], -1)
        near = nc / np.maximum(den, 1e-3)[..., None]
        dd = np.abs(near - r_s.astype(np.float32)).mean(2)
        add &= dd <= color_tol
    if not add.any():
        return alpha
    if ds > 1:
        add = cv2.resize(add.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    out = np.array(alpha, copy=True)
    out[add] = 1.0
    return out


def fill_enclosed_holes(alpha, ds=2, max_frac=0.25):
    """把主体内部封闭的洞全部补实 —— 宁可变"胖"，也不要出现破洞。

    只补“不接触画面边缘”的背景块，所以主体外部的背景不会被吃进来。
    """
    H, W = alpha.shape
    bw = (alpha >= 0.5).astype(np.uint8)
    if bw.all() or not bw.any():
        return alpha
    s = ds if max(H, W) > 1500 else 1
    small = cv2.resize(bw, (max(2, W // s), max(2, H // s)), interpolation=cv2.INTER_NEAREST)
    inv = (1 - small).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    if n <= 1:
        return alpha
    border = np.zeros(n, bool)
    for arr in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
        border[np.unique(arr)] = True
    fill = np.zeros(n, bool)
    lim = max_frac * small.size
    for i in range(1, n):
        if not border[i] and st[i, cv2.CC_STAT_AREA] <= lim:
            fill[i] = True
    if not fill.any():
        return alpha
    fm = fill[lab].astype(np.uint8)
    if s > 1:
        fm = cv2.resize(fm, (W, H), interpolation=cv2.INTER_NEAREST)
    out = np.maximum(alpha, fm.astype(np.float32))
    return out


def drop_specks(mask, min_frac=1e-5, min_px=64):
    """删掉面积过小的孤立碎块（多半是模型犹豫抖出来的噪点），主体不受影响。"""
    if mask.ndim != 2:
        return mask
    thr = 127 if mask.max() > 1 else 0.5
    bw = (mask >= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 2:
        return mask
    lim = max(min_px, int(min_frac * mask.size))
    kill = np.zeros(n, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < lim:
            kill[i] = 1
    if not kill.any():
        return mask
    out = mask.copy()
    out[kill[lab] > 0] = 0
    return out


def fill_transparent(rgba, a8, target=256, band=32):
    """把 alpha==0 区域的 RGB 填掉，这是 PNG 体积能不能降下来的关键。

    原透明区里是去污公式 F=(C-(1-a)B)/a 在 a→0 时除爆得到的逐像素噪声，
    PNG 的 deflate 对噪声几乎零压缩 —— 一张 50MP 的图因此能顶到 47MB。

    做法：离主体 band 像素以内用 push-pull 扩散出的主体颜色（边缘不发黑），
    更远处统一填主体边缘的平均色（大块平坦，压缩率最高）。
    只改 alpha==0 的像素，主体上一个像素都不动。
    """
    H, W = a8.shape
    hole = a8 == 0
    if not hole.any() or hole.all():
        return rgba
    s = max(1, int(round(max(H, W) / float(target))))    # 扩散场只用 256px 的工作分辨率
    sh, sw = max(4, H // s), max(4, W // s)
    cov = cv2.resize((~hole).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.resize(np.ascontiguousarray(rgba[:, :, :3]), (sw, sh),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    pyr = [(small * cov[..., None], cov)]
    while min(pyr[-1][1].shape) > 4:
        c, wgt = pyr[-1]
        pyr.append((cv2.resize(c, (wgt.shape[1] // 2, wgt.shape[0] // 2), interpolation=cv2.INTER_AREA),
                    cv2.resize(wgt, (wgt.shape[1] // 2, wgt.shape[0] // 2), interpolation=cv2.INTER_AREA)))
    c, wgt = pyr[-1]
    c = c / np.maximum(wgt[..., None], 1e-4)
    for cc, ww in reversed(pyr[:-1]):
        up = cv2.resize(c, (cc.shape[1], cc.shape[0]), interpolation=cv2.INTER_LINEAR)
        c = cc + up * np.maximum(1.0 - ww, 0.0)[..., None]
    full = cv2.resize(np.clip(c, 0.0, 255.0).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)

    fg = (a8 > 0).astype(np.uint8)
    ring = fg & (1 - cv2.erode(fg, np.ones((9, 9), np.uint8)))
    mean = rgba[:, :, :3][ring.astype(bool)].reshape(-1, 3).mean(0) if ring.any()         else np.array([128.0, 128.0, 128.0], np.float32)
    mean = np.clip(mean, 0, 255).astype(np.uint8)

    # 只在离主体 band 像素内用扩散色，其余地方用统一色（可压缩性好得多）
    ds = max(2, W // 8), max(2, H // 8)
    fg_s = cv2.resize(fg, ds, interpolation=cv2.INTER_AREA) > 0
    dist = cv2.distanceTransform((~fg_s).astype(np.uint8), cv2.DIST_L2, 5)
    near_s = (dist <= max(1.0, band / 8.0)).astype(np.uint8)
    near = cv2.resize(near_s, (W, H), interpolation=cv2.INTER_NEAREST) > 0

    np.copyto(full, np.broadcast_to(mean, full.shape), where=(hole & ~near)[..., None])
    np.copyto(rgba[:, :, :3], full, where=hole[..., None])
    return rgba


# --------------------------------------------------------------------------
# 单张处理
# --------------------------------------------------------------------------
def _fit(w, h, max_side):
    s = min(1.0, float(max_side) / float(max(w, h)))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def _tile_origins(start, end, size, step):
    if end - start <= size:
        return [max(start, end - size)]
    out = list(range(start, end - size + 1, step))
    if out[-1] != end - size:
        out.append(end - size)
    return out


def edge_band(alpha, band=64, ds=4):
    """以 alpha 的 0.5 等值线为中心，band 像素内权重 1→0 的羽化带。

    精修结果只在边界附近生效：远离边界的大片区域一律沿用粗蒙版的判断，
    否则小块模型“各说各话”会把整片区域平均成灰色。
    """
    H, W = alpha.shape
    sh, sw = max(2, H // ds), max(2, W // ds)
    s = cv2.resize(alpha.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    fg = (s >= 0.5).astype(np.uint8)
    din = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
    d = np.where(fg > 0, din, dout) * float(ds)
    w = np.clip(1.0 - d / max(1.0, float(band)), 0.0, 1.0).astype(np.float32)
    return cv2.resize(w, (W, H), interpolation=cv2.INTER_LINEAR)


def regional_decision(alpha, thr=0.5, ds=16, sigma=3.0):
    """把 alpha 缩到 1/16 再平滑阈值化 —— 让一个物体整体做一个决定。

    全分辨率的 0.5 阈值会把“平均 0.5、忽高忽低”的区域切成一半白一半黑的碎片，
    缩到 1/16 之后这些问题就被平均掉了。
    """
    H, W = alpha.shape
    sh, sw = max(2, H // ds), max(2, W // ds)
    small = cv2.resize(alpha.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    hard = (up >= float(thr)).astype(np.uint8)
    hard = cv2.medianBlur(hard, 5)          # 去掉零星碎点
    return hard.astype(np.float32)


def snap_flat_alpha(alpha, band=16, ds=4, sigma=12.0, thr=0.5):
    """离边缘超过 band 像素的地方只允许 0 或 1。

    真实边界只有 1~3 像素的过渡，几千像素宽的“半透明”几乎都是模型在犹豫。
    远处用大尺度平滑后的结果做“整块裁决”（大 sigma 高斯），
    这样同一块区域（比如花盆）只会整体算前景或整体算背景，不会一半白一半黑；
    边界附近仍保留 alpha 的抗锯齿过渡。
    """
    w = edge_band(alpha, band=band, ds=ds)
    hard = regional_decision(alpha, thr=thr)
    return np.clip(alpha * w + hard * (1.0 - w), 0.0, 1.0)


def tile_detail_alpha(rgb, session, coarse_full, tile=3072, overlap=768,
                      max_tiles=12, verbose=False, norm=None, band=24, max_hole=0.0,
                      bg_tol=42.0):
    """分块精修：把主体区域切成 tile×tile 的小块按原生分辨率推理，再无缝拼回。

    原图 8000px 直接喂 1024 的模型等于 8 倍下采样，叶片之间 100px 的缝隙会被抹平；
    切成 3072 的块只下采样 3 倍，缝隙就能被正确识别成背景。
    """
    H, W = rgb.shape[:2]
    ys, xs = np.where(coarse_full > 0.5)
    if ys.size == 0:
        return coarse_full
    pad = int(0.04 * max(H, W))
    y0 = max(0, int(ys.min()) - pad); y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad); x1 = min(W, int(xs.max()) + pad + 1)

    step = max(64, tile - overlap)
    ys_ = _tile_origins(y0, y1, tile, step)
    xs_ = _tile_origins(x0, x1, tile, step)
    if len(ys_) * len(xs_) > max_tiles:
        return coarse_full

    # 累加区 = 全部分块的并集：分块可能比主体包围盒大，也可能被图像边界裁短，
    # 只有把累加区取成并集，块的形状和写入位置才永远对得上。
    ay0 = ys_[0]; ay1 = min(H, ys_[-1] + tile)
    ax0 = xs_[0]; ax1 = min(W, xs_[-1] + tile)
    bh, bw = ay1 - ay0, ax1 - ax0
    acc = np.zeros((bh, bw), np.float32)
    wsum = np.zeros((bh, bw), np.float32)

    _wins = {}

    def _make_window(th, tw):
        """按分块的真实尺寸生成羽化窗（图像比 tile 小时块会被裁短）。"""
        key = (th, tw)
        win = _wins.get(key)
        if win is not None:
            return win
        win = np.ones((th, tw), np.float32)
        oh = min(overlap, th // 3)          # 最多各占 1/3，中间留出平台区
        ow = min(overlap, tw // 3)
        if oh > 0:
            ry = np.linspace(0.0, 1.0, oh + 2, dtype=np.float32)[1:-1]
            win[:oh, :] *= ry[:, None]
            win[-oh:, :] *= ry[::-1][:, None]
        if ow > 0:
            rx = np.linspace(0.0, 1.0, ow + 2, dtype=np.float32)[1:-1]
            win[:, :ow] *= rx[None, :]
            win[:, -ow:] *= rx[::-1][None, :]
        _wins[key] = win
        return win

    n = 0
    t0 = time.time()
    for ty in ys_:
        th = min(tile, H - ty)
        for tx in xs_:
            tw = min(tile, W - tx)
            if th <= 0 or tw <= 0:
                continue
            crop = rgb[ty:ty + th, tx:tx + tw]
            a = model_alpha(crop, session, norm=norm).astype(np.float32) / 255.0
            if a.shape[:2] != (th, tw):
                a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_LINEAR)
            w = _make_window(th, tw)
            acc[ty - ay0:ty - ay0 + th, tx - ax0:tx - ax0 + tw] += a * w
            wsum[ty - ay0:ty - ay0 + th, tx - ax0:tx - ax0 + tw] += w
            n += 1
    tiled = np.array(coarse_full, copy=True)
    covered = wsum > 1e-6
    sub = tiled[ay0:ay1, ax0:ax1]
    sub[covered] = acc[covered] / wsum[covered]
    tiled[ay0:ay1, ax0:ax1] = sub
    tiled = np.clip(tiled, 0.0, 1.0)

    # 1) 边界 ±band 内采纳精修：远处保持粗判断，避免整片变灰
    w = edge_band(coarse_full, band=band)
    out = coarse_full * (1.0 - w) + tiled * w

    # 2) 但“叶子之间的黑缝”是粗结果误并进前景的**小洞**，精修在高分辨率下看得很清楚。
    #    允许挖掉这类小面积区域；花盆那种大面积区域不在其列（会破坏整体一致性）。
    demote = (tiled < 0.3) & (coarse_full > 0.5) if max_hole > 0 else None
    if demote is not None and demote.any():
        n_lab, lab, st, _ = cv2.connectedComponentsWithStats(demote.astype(np.uint8), 8)
        lim = max(256, int(max_hole * coarse_full.size))
        ok = np.zeros(n_lab, bool)
        for i in range(1, n_lab):
            ok[i] = st[i, cv2.CC_STAT_AREA] < lim
        # 还要求这块的颜色接近真正的背景（叶子缝里是黑布），
        # 否则会把花盆边缘那类“模型看走眼”的区域也挖出个缺口
        if rgb is not None and ok.any():
            bgpix = rgb[coarse_full < 0.1]
            if bgpix.size >= 30:
                bgm = np.median(bgpix.reshape(-1, 3)[::7].astype(np.float32), axis=0)
                for i in range(1, n_lab):
                    if not ok[i]:
                        continue
                    x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
                    wd, ht = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
                    sub = rgb[y:y + ht, x:x + wd]
                    ml = lab[y:y + ht, x:x + wd] == i
                    if ml.sum() < 16:
                        continue
                    c = sub[ml].reshape(-1, 3).astype(np.float32).mean(0)
                    if np.abs(c - bgm).max() > bg_tol:
                        ok[i] = False
        demote = ok[lab]
        if demote.any():
            out = np.where(demote, np.minimum(out, tiled), out)
    if verbose:
        print("      detail tiles: %d  %.1fs" % (n, time.time() - t0), flush=True)
    return np.clip(out, 0.0, 1.0)


def compute_alpha(rgb, session, work_size, use_matte, gf_s=2, contrast=0.2,
                  detail=False, tile=3072, verbose=False, detail_band=64, snap=True,
                  alpha_threshold=0.5, bridge=24, carve_holes=0.0, fill_holes=True):
    """模型粗分割 -> 清理 -> matting -> 引导滤波上采样，返回原图尺寸的 float alpha。"""
    h, w = rgb.shape[:2]
    t = {}

    t0 = time.time()
    ww, hh = _fit(w, h, work_size)
    small = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA) if (ww, hh) != (w, h) else rgb
    a_small = model_alpha(small, session)
    norm_range = getattr(session, "last_range", None)   # 分块精修沿用同一拉伸区间
    t["model"] = time.time() - t0

    t0 = time.time()
    a_small = clean_mask(a_small, rgb=small)
    a = a_small.astype(np.float32) / 255.0
    t["clean"] = time.time() - t0

    if use_matte:
        t0 = time.time()
        band = max(2, int(round(max(hh, ww) / 512)))
        a_matte = closed_form_matte(small, a_small, band=band)
        # matting 只在 uncertain 区可信，其它位置保留原 alpha 的形态
        a = np.clip(0.15 * a + 0.85 * a_matte, 0.0, 1.0)
        t["matte"] = time.time() - t0

    t0 = time.time()
    scale = max(h, w) / float(max(hh, ww))
    if detail and max(h, w) > work_size:
        # 先把粗蒙版放大到原图，再分块按原生分辨率重算，叶片缝隙不会被抹平
        coarse = upsample_alpha(a, rgb, scale, s=gf_s, contrast=0.0)
        a_full = tile_detail_alpha(rgb, session, coarse, tile=tile, verbose=verbose,
                                   norm=norm_range, band=detail_band,
                                   max_hole=carve_holes)
        a_full = sharpen_alpha(a_full, contrast)
    else:
        a_full = upsample_alpha(a, rgb, scale, s=gf_s, contrast=contrast)
    if snap:
        # 大片“半透明”统一钉成黑白，只保留边缘那一两像素的过渡
        a_full = snap_flat_alpha(a_full, thr=alpha_threshold)
    if bridge > 0:
        # 细枝断口：只补窄缝且颜色接近前景的地方
        a_full = bridge_gaps(a_full, rgb, max_width=bridge)
    if fill_holes:
        # 主体内部任何封闭的洞一律补实：宁可多留一点背景，也不要破洞
        a_full = fill_enclosed_holes(a_full)
    t["upsample"] = time.time() - t0
    if verbose:
        print("      timing:", {k: round(v, 1) for k, v in t.items()}, flush=True)
    return a_full


@dataclass(frozen=True)
class ExportOptions:
    """CLI 与交互界面共用的导出参数。"""
    out_type: str = "both"
    suffix: str = "_fg"
    ext: str = ".png"
    expand: float = 0.0
    feather: float = 0.0
    decon: bool = True
    work_size: int = 1536
    mask_binary: bool = True
    alpha_threshold: float = 0.5
    bg_fill: str = "bleed"
    png_level: int = 6
    keep_exif: bool = True
    camera_json: bool = False
    drop_mask_specks: bool = True


def _temporary_target(target, suffix=None):
    """在同目录创建临时文件，成功保存后再原子替换目标。"""
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    ext = suffix if suffix is not None else (os.path.splitext(target)[1] or ".tmp")
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=ext, dir=directory)
    os.close(fd)
    return tmp


def _save_image_atomic(img, target, save_format, **save_kw):
    tmp = _temporary_target(target)
    try:
        img.save(tmp, format=save_format, **save_kw)
        if not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
            raise OSError("图像写入结果为空：%s" % target)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _save_mask_atomic(mask, target):
    tmp = _temporary_target(target, suffix=".png")
    try:
        Image.fromarray(mask, mode="L").save(tmp, format="PNG")
        if not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
            raise OSError("蒙版写入结果为空：%s" % target)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _save_json_atomic(payload, target):
    tmp = _temporary_target(target, suffix=".json")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def export_alpha(source_path, out_dir, rgb, exif, icc, src_alpha, alpha, options,
                 fg_path=None, mask_path=None):
    """把给定 alpha 精修收尾并写出结果。

    这是 CLI 和交互界面共用的唯一导出入口，保证透明 PNG、蒙版、EXIF/ICC、
    DNG 相机信息与源 alpha 的处理完全一致。返回输出路径与覆盖率统计。
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    ext = options.ext if options.ext.startswith(".") else "." + options.ext
    if options.out_type not in ("fg", "mask", "both"):
        raise ValueError("out_type 必须是 fg、mask 或 both")
    if fg_path is None:
        fg_path = os.path.join(out_dir, stem + options.suffix + ext)
    if mask_path is None:
        mask_path = os.path.join(out_dir, stem + options.suffix + "_mask.png")

    for target in (fg_path, mask_path):
        if target and os.path.abspath(target) == os.path.abspath(source_path):
            raise ValueError("输出路径不能与原图相同：%s" % target)

    alpha = np.asarray(alpha, dtype=np.float32)
    if options.expand:
        r = int(round(options.expand))
        if r > 0:
            alpha = cv2.dilate(
                alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        elif r < 0:
            alpha = cv2.erode(
                alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * -r + 1,) * 2))
    if options.feather > 0:
        k = 2 * int(round(options.feather)) + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    if src_alpha is not None:
        alpha = alpha * (src_alpha.astype(np.float32) / 255.0)

    a8 = (alpha * 255.0).round().astype(np.uint8)
    H, W = a8.shape
    cover = float((a8 > 127).mean())
    del alpha

    fg_written = None
    mask_written = None

    if options.out_type in ("fg", "both"):
        info = camera_info(source_path, exif)
        info["orientation"] = 1                # 方向已烘焙进像素
        rgba = np.empty((H, W, 4), np.uint8)
        if options.decon:
            decontaminate(rgb, a8,
                          radius=max(4, int(round(2 * max(H, W) / max(1, options.work_size)))),
                          dst=rgba[:, :, :3])
        else:
            rgba[:, :, :3] = rgb
        rgba[:, :, 3] = a8
        if options.bg_fill == "bleed":
            fill_transparent(rgba, a8)
        elif options.bg_fill == "black":
            rgba[a8 == 0, :3] = 0
        img = Image.frombuffer("RGBA", (W, H), rgba, "raw", "RGBA", 0, 1)
        del rgba
        save_kw = {}
        if fg_path.lower().endswith(".png"):
            save_kw["compress_level"] = int(options.png_level)
        ex = _exif_with_orientation_reset(exif)
        if ex and options.keep_exif:
            save_kw["exif"] = ex
        if icc:
            save_kw["icc_profile"] = icc

        if fg_path.lower().endswith(".png"):
            try:
                from PIL.PngImagePlugin import PngInfo
                meta = PngInfo()
                for k, v in (("Make", info.get("make")), ("Model", info.get("model")),
                             ("LensModel", info.get("lens")),
                             ("DateTimeOriginal", info.get("datetime")),
                             ("FocalLength", info.get("focal_mm")),
                             ("FocalLengthIn35mmFilm", info.get("focal35_mm")),
                             ("FNumber", info.get("fnumber")), ("ISO", info.get("iso"))):
                    if v not in (None, ""):
                        meta.add_text(k, str(v))
                save_kw["pnginfo"] = meta
            except Exception:
                pass
        save_format = Image.registered_extensions().get(os.path.splitext(fg_path)[1].lower(), "PNG")
        _save_image_atomic(img, fg_path, save_format, **save_kw)
        fg_written = fg_path

        if options.camera_json:
            try:
                payload = intrinsics_json(info, W, H)
                payload["source"] = os.path.basename(source_path)
                camera_path = os.path.splitext(fg_path)[0] + "_camera.json"
                _save_json_atomic(payload, camera_path)
            except Exception as e:
                raise OSError("相机参数写入失败：%s" % e)

    if options.out_type in ("mask", "both"):
        m = a8
        if options.mask_binary:
            m = (a8 >= options.alpha_threshold * 255.0).astype(np.uint8) * 255
        if options.drop_mask_specks:
            m = drop_specks(m)
        _save_mask_atomic(m, mask_path)
        mask_written = mask_path

    return {"cover": cover, "w": W, "h": H, "fg_path": fg_written, "mask_path": mask_written}


def process_one(path, out_dir, args, session, model_name):
    """处理一张图，返回 (ok, message, stats)。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    ext = args.ext if args.ext.startswith(".") else "." + args.ext
    fg_path = os.path.join(out_dir, stem + args.suffix + ext)
    mask_path = os.path.join(out_dir, stem + args.suffix + "_mask.png")
    # 只有“本次要产出的文件都齐了”才算跳过：以前 fg 在、蒙版缺时会漏掉蒙版
    need = []
    if args.type in ("fg", "both"):
        need.append(fg_path)
    if args.type in ("mask", "both"):
        need.append(mask_path)
    if need and args.skip_existing and not args.overwrite and all(os.path.exists(p) for p in need):
        return True, "skip(已存在)", {}

    t0 = time.time()
    rgb, exif, icc, src_alpha = load_image(path)
    if rgb is None or rgb.size == 0:
        return False, "读取失败", {}
    H, W = rgb.shape[:2]

    alpha = compute_alpha(rgb, session, args.work_size, args.matte,
                          gf_s=args.gf_subsample, contrast=args.alpha_contrast,
                          detail=args.detail, tile=args.tile_size,
                          verbose=args.verbose, detail_band=args.detail_band,
                          snap=args.alpha_snap, alpha_threshold=args.alpha_threshold,
                          bridge=args.bridge, carve_holes=args.carve_holes)

    options = ExportOptions(
        out_type=args.type,
        suffix=args.suffix,
        ext=ext,
        expand=args.expand,
        feather=args.feather,
        decon=args.decon,
        work_size=max(1, args.work_size),
        mask_binary=args.mask_binary,
        alpha_threshold=args.alpha_threshold,
        bg_fill=args.bg_fill,
        png_level=args.png_level,
        keep_exif=args.keep_exif,
        camera_json=args.camera_json,
        drop_mask_specks=True,
    )
    result = export_alpha(path, out_dir, rgb, exif, icc, src_alpha, alpha, options,
                          fg_path=fg_path if args.type in ("fg", "both") else None,
                          mask_path=mask_path if args.type in ("mask", "both") else None)

    dt = time.time() - t0
    return True, "%.1fs" % dt, {"cover": result["cover"], "w": W, "h": H}


# --------------------------------------------------------------------------
# 批处理
# --------------------------------------------------------------------------
_WORKER_ARGS = None


def _init_worker(args, model_name):
    global _WORKER_ARGS
    _WORKER_ARGS = (args, model_name)
    if args.workers > 1:
        cv2.setNumThreads(1)


def _worker(path):
    args, model_name = _WORKER_ARGS
    session = get_session(model_name, args.model_dir, args.threads, args.mem_arena)
    try:
        return process_one(path, args.output, args, session, model_name)
    except Exception as e:
        return False, "错误: %s" % e, {"trace": traceback.format_exc()}
    finally:
        import gc
        gc.collect()


def collect_files(root, recursive):
    if os.path.isfile(root):
        return [root]
    out = []
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                    out.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            fp = os.path.join(root, f)
            if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                out.append(fp)
    return sorted(out)


def build_parser():
    p = argparse.ArgumentParser(
        description="批量抠图：去背景、保留主体细节、输出透明 PNG（支持 DNG/RAW）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="input", help="输入图片或目录")
    p.add_argument("--output", default="output", help="输出目录")
    p.add_argument("--type", choices=["fg", "mask", "both"], default="fg",
                   help="fg=透明前景 PNG；mask=蒙版；both=两者")
    p.add_argument("--ext", default=".png", help="前景图输出扩展名")
    p.add_argument("--suffix", default="_fg", help="输出文件名后缀")
    p.add_argument("--model", default="auto",
                   help="auto|birefnet-general|birefnet-general-lite|isnet-general-use|u2net")
    p.add_argument("--model-dir", default=None, help="模型缓存目录（默认 U2NET_HOME 或脚本目录）")
    p.add_argument("--quality", choices=list(QUALITY_PRESETS), default="balanced",
                   help="fast/balanced/best，控制工作分辨率与是否 matting")
    p.add_argument("--work-size", type=int, default=None, help="模型工作分辨率（长边像素）")
    p.add_argument("--matte", dest="matte", action="store_true", default=None,
                   help="启用封闭形式 matting（默认按 quality 预设）")
    p.add_argument("--no-matte", dest="matte", action="store_false")
    p.add_argument("--decon", dest="decon", action="store_true", default=True,
                   help="边缘色彩去污（默认开）")
    p.add_argument("--no-decon", dest="decon", action="store_false")
    p.add_argument("--gf-subsample", type=int, default=None, help="引导滤波下采样倍数 1/2/4")
    p.add_argument("--detail", dest="detail", action="store_true", default=True,
                   help="分块精修：按原生分辨率重算主体区域，叶片间缝隙不再被抹平（默认开，约 1~3 分钟/张）")
    p.add_argument("--no-detail", dest="detail", action="store_false",
                   help="关闭分块精修，速度快但叶片间暗缝会被填平")
    p.add_argument("--carve-holes", dest="carve_holes", type=float, default=0.0,
                   help="允许精修在主体内部挖掉的“小洞”面积上限（占画面比例），0=关闭")
    p.add_argument("--bridge", type=int, default=24,
                   help="细枝断口桥接的缝宽上限（像素），0=关闭")
    p.add_argument("--alpha-threshold", type=float, default=0.5,
                   help="大块区域的整体裁决阈值：调低更容易把花盆等连带物体保留，调高更严格")
    p.add_argument("--detail-band", type=int, default=24,
                   help="分块精修只在这个像素范围内影响边界，远离边界沿用粗判断")
    p.add_argument("--alpha-snap", dest="alpha_snap", action="store_true", default=True,
                   help="把远离边缘的大片半透明钉成 0/1（默认开）")
    p.add_argument("--no-alpha-snap", dest="alpha_snap", action="store_false")
    p.add_argument("--tile-size", type=int, default=3072,
                   help="分块精修的块大小（像素，越小越准但越慢）")
    p.add_argument("--alpha-contrast", type=float, default=0.20,
                   help="边缘过渡带收窄强度（0 关闭；0.1~0.3 较合适）")
    p.add_argument("--expand", type=float, default=0.0, help="边缘外扩像素（负数为内缩）")
    p.add_argument("--feather", type=float, default=0.0, help="额外羽化半径（像素）")
    p.add_argument("--mask-binary", dest="mask_binary", action="store_true", default=True,
                   help="mask 输出纯黑/纯白（默认开）")
    p.add_argument("--mask-soft", dest="mask_binary", action="store_false",
                   help="mask 保留软边（抗锯齿），合成时更自然")
    p.add_argument("--bg-fill", choices=["bleed", "black", "keep"], default="bleed",
                   help="透明区的 RGB 怎么处理：bleed=用主体颜色扩散填满(默认,最小体积)、"
                        "black=填 0、keep=保留去污噪声(文件最大)")
    p.add_argument("--png-level", type=int, default=6, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   help="PNG 压缩级别 0-9（默认 6；9 更小但慢 3~4 倍）")
    p.add_argument("--keep-exif", dest="keep_exif", action="store_true", default=True,
                   help="把原图 EXIF（含焦距等内参）写回输出 PNG")
    p.add_argument("--no-exif", dest="keep_exif", action="store_false")
    p.add_argument("--camera-json", dest="camera_json", action="store_true", default=False,
                   help="额外输出 *_camera.json（焦距/像素焦距/主点，方便三维重建）")
    p.add_argument("--no-camera-json", dest="camera_json", action="store_false")
    p.add_argument("--recursive", action="store_true", help="递归子目录")
    p.add_argument("--workers", type=int, default=1,
                   help="并行进程数（内存 32GB 建议 2~3；单进程最稳）")
    p.add_argument("--recycle", type=int, default=6,
                   help="并行时每处理 N 张就重建进程池（防止内存碎片累积）")
    p.add_argument("--threads", type=int, default=0,
                   help="每个进程的推理线程数（0=auto，并行时建议 2~4）")
    p.add_argument("--no-mem-arena", dest="mem_arena", action="store_false", default=True,
                   help="关闭 ONNX 内存池，进一步降低峰值内存")
    p.add_argument("--skip-existing", action="store_true", help="已存在则跳过")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有文件（配合 --skip-existing）")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model_dir is None:
        # 优先用项目自带的 models\（整个文件夹可打包发给别人）
        local_models = os.path.join(script_dir, "models")
        has_local = os.path.isdir(local_models) and any(
            f.lower().endswith(".onnx") for f in os.listdir(local_models))
        if has_local:
            args.model_dir = local_models
        elif os.environ.get("U2NET_HOME"):
            args.model_dir = os.environ["U2NET_HOME"]
        else:
            args.model_dir = local_models
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    work, matte, gf_s = QUALITY_PRESETS[args.quality]
    if args.work_size is not None:
        work = args.work_size
    if args.matte is None:
        matte = matte
    if args.gf_subsample is None:
        gf_s = gf_s
    args.work_size, args.matte, args.gf_subsample = work, matte, gf_s
    if args.quality == "fast":
        args.detail = False          # fast 预设以速度优先

    if args.type in ("fg", "both"):
        e = args.ext.lower()
        if e not in (".png", ".tif", ".tiff", ".webp"):
            print("[提示] 透明通道需要 PNG/TIFF/WebP，--ext 已改为 .png")
            args.ext = ".png"

    files = collect_files(args.input, args.recursive)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("输入目录 '%s' 中没有找到支持的图片（%s ...）" % (args.input, ", ".join(sorted(IMG_EXTS)[:6])))
        return 1

    model_name = resolve_model(args.model, args.model_dir)
    print("=" * 64)
    print("抠图任务：%d 张 | 模型 %s | 质量 %s (work=%d, matte=%s, 分块精修=%s)" %
          (len(files), model_name, args.quality, args.work_size, args.matte, args.detail))
    print("模型目录：%s" % os.path.abspath(args.model_dir))
    print("输入：%s\n输出：%s" % (os.path.abspath(args.input), os.path.abspath(args.output)))
    print("=" * 64)

    t0 = time.time()
    ok = fail = 0
    if args.threads <= 0:
        cpu = os.cpu_count() or 4
        args.threads = max(1, min(6, cpu // max(1, args.workers)))
    if args.workers > 1:
        cv2.setNumThreads(1)
    session = get_session(model_name, args.model_dir, args.threads, args.mem_arena)
    print("线程：每进程 %d，进程数 %d" % (args.threads, args.workers))

    def _run(i, path):
        name = os.path.basename(path)
        print("[%d/%d] %s ..." % (i, len(files), name), end=" ", flush=True)
        good, msg, stats = False, "", {}
        for attempt in (1, 2):
            try:
                good, msg, stats = process_one(path, args.output, args, session, model_name)
                break
            except MemoryError:
                import gc
                gc.collect()
                msg = "内存不足，已重试一次" if attempt == 1 else "错误: 内存不足"
                good = False
            except Exception as e:
                good, msg, stats = False, "错误: %s" % e, {}
                if args.verbose:
                    traceback.print_exc()
                break
        print(("✓ " if good else "✗ ") + msg, flush=True)
        return good

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # 每处理 recycle 张就重建进程池：Python/ONNX 的碎片化内存不会一直累积
        chunk = max(1, args.recycle)
        done = 0
        for s in range(0, len(files), chunk):
            batch = files[s:s + chunk]
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                     initargs=(args, model_name)) as ex:
                futs = {ex.submit(_worker, p): p for p in batch}
                for fut in as_completed(futs):
                    p = futs[fut]
                    done += 1
                    try:
                        good, msg, _ = fut.result()
                    except Exception as e:
                        good, msg = False, "错误: %s" % e
                    print("[%d/%d] %s ... %s %s" % (done, len(files), os.path.basename(p),
                                                    "✓" if good else "✗", msg), flush=True)
                    ok += int(good); fail += int(not good)
    else:
        for i, p in enumerate(files, 1):
            good = _run(i, p)
            ok += int(good); fail += int(not good)

    print("-" * 64)
    print("完成：成功 %d，失败 %d，总耗时 %.1fs -> %s" %
          (ok, fail, time.time() - t0, os.path.abspath(args.output)))
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    main()
