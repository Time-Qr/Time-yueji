# -*- coding: utf-8 -*-
"""Download and verify the SAM ViT-B ONNX artifacts used by interactive mode."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RELEASE_BASE = (
    "https://github.com/Time-Qr/Time-yueji/releases/download/v1.1.0/"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, expected: str) -> None:
    target = MODELS / name
    if target.is_file() and sha256(target) == expected:
        print("[模型] 已存在并通过校验：%s" % name)
        return

    temporary = target.with_suffix(target.suffix + ".part")
    url = RELEASE_BASE + name
    print("[模型] 下载：%s" % url)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Time-yueji"})
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with temporary.open("wb") as stream:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    stream.write(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write(
                            "\r    %.1f / %.1f MB"
                            % (done / 1048576, total / 1048576)
                        )
                        sys.stdout.flush()
        print()
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise

    actual = sha256(target)
    if actual != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            "%s SHA256 校验失败：期望 %s，实际 %s" % (name, expected, actual)
        )
    print("[模型] 校验通过：%s" % name)


def main() -> int:
    manifest_path = MODELS / "sam_vit_b_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for section in ("encoder", "decoder"):
        entry = manifest[section]
        download(entry["file"], entry["sha256"])
    print("[完成] 交互模型已准备好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
