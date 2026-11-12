# Time-yueji

Windows 下的交互式图像抠图工具。使用 **SAM ViT-B ONNX** 根据框选、正点和负点选择目标，并通过 matting、引导滤波和去污生成透明 PNG 与蒙版。

本仓库只包含交互模式，不包含原项目的 BiRefNet 批量入口和批量模型。

## 功能

- 打开文件夹或添加多张图片
- 框选目标
- 正点加入目标区域
- 负点排除误选区域
- 多候选蒙版切换
- 颜色辅助选点
- 候选工作台、评分、标签和工程包
- 有效状态面板与按需反事实分析
- 支持 JPG、PNG、BMP、TIFF、WebP 以及 DNG、CR2、NEF、ARW、RAF、RW2、ORF、PEF 等 RAW
- 导出透明 PNG 和黑白蒙版，并保留可用的 EXIF 信息

## 安装

要求：

- Windows 10/11 64 位
- Python 3.10、3.11 或 3.12（64 位）
- 首次安装需要联网，模型约 360 MB

双击：

```text
install.bat
```

安装脚本会创建 `.venv`、安装依赖、下载 SAM encoder/decoder 并校验 SHA256。

## 使用

双击：

```text
run_interactive.bat
```

也可以把图片文件或图片文件夹拖到 `run_interactive.bat` 上。

默认输出目录：

```text
output_interactive\
```

常用快捷键：

| 快捷键 | 功能 |
|---|---|
| B | 框选 |
| P | 正点 |
| N | 负点 |
| H | 平移 |
| Ctrl+Z | 撤销 |
| Ctrl+S | 导出 |
| R | 重置 |
| F | 适应窗口 |
| O | 原图/抠图预览 |
| [ / ] | 切换候选蒙版 |

## 手动运行

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe setup_check.py
.venv\Scripts\python.exe interactive_gui.py
```

## 仓库内容

```text
interactive_gui.py       交互程序入口
interactive/              交互界面、SAM、后台线程和候选分析
generate_masks.py         交互模式使用的图像读取、精修和导出后端
models/                   SAM manifest 和第三方许可文件
tests/                    交互功能测试
install.bat               安装依赖和模型
run_interactive.bat       启动程序
download_models.py        下载并校验模型
setup_check.py            安装后自检
```

模型文件没有放入 Git 历史，而是作为 GitHub Release 资产发布，避免超过 GitHub 单文件大小限制。

## 第三方模型

SAM ViT-B ONNX 模型由 Meta 的 Segment Anything 官方 checkpoint 导出，相关来源、版本、校验值和 Apache-2.0 许可见：

```text
models\sam_vit_b_manifest.json
models\SAM_LICENSE.txt
```

## 测试

```bat
.venv\Scripts\python.exe -m pytest tests
```
