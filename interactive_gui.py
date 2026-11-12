#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互抠图桌面入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description="交互式框选抠图（SAM + 现有边缘精修）")
    p.add_argument("path", nargs="?", default=None,
                   help="可选：启动时打开的图片文件夹或单张图片")
    p.add_argument("--models", default=None, help="模型目录（默认项目 models）")
    p.add_argument("--output", default=None, help="输出目录（默认 output_interactive）")
    return p


def main():
    args = build_parser().parse_args()
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        raise SystemExit(
            "缺少 PySide6。请先运行「首次安装_双击我.bat」。\n%s" % exc)
    from interactive.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("交互抠图")
    app.setStyle("Fusion")
    window = MainWindow(models_dir=args.models, out_dir=args.output)
    window.show()
    if args.path:
        from PySide6.QtCore import QTimer
        path = Path(args.path)
        QTimer.singleShot(0, lambda: window.open_path(path))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
