# -*- coding: utf-8 -*-
"""检查交互模式依赖和 SAM ONNX 模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_modules() -> bool:
    modules = (
        ("numpy", "numpy"),
        ("cv2", "opencv-python-headless"),
        ("PIL", "Pillow"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
        ("onnxruntime", "onnxruntime"),
        ("pymatting", "PyMatting"),
        ("rawpy", "rawpy"),
        ("exifread", "ExifRead"),
        ("piexif", "piexif"),
        ("PySide6", "PySide6-Essentials"),
    )
    missing = []
    for module, package in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing.append("%s (%s): %s" % (module, package, exc))
    if missing:
        print("[错误] 依赖不完整：")
        for item in missing:
            print("  - %s" % item)
        return False
    print("[依赖] 检查通过")
    return True


def check_models() -> bool:
    manifest_path = MODELS / "sam_vit_b_manifest.json"
    if not manifest_path.is_file():
        print("[错误] 缺少 models\\sam_vit_b_manifest.json")
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for section in ("encoder", "decoder"):
            entry = manifest[section]
            path = MODELS / entry["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(entry["bytes"]):
                raise RuntimeError("%s 文件大小不匹配" % entry["file"])
            if sha256(path) != entry["sha256"]:
                raise RuntimeError("%s SHA256 不匹配" % entry["file"])
    except Exception as exc:
        print("[错误] SAM 模型检查失败：%s" % exc)
        print("      请先运行：python download_models.py")
        return False
    print("[SAM] encoder、decoder 和 manifest 检查通过")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Time-yueji 交互模式自检")
    parser.parse_args()
    ok = check_modules() and check_models()
    if ok:
        (ROOT / ".installed").write_text("ok\n", encoding="utf-8")
        print("[完成] 可以运行交互抠图。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
