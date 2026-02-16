#!/usr/bin/env python3
"""
SHARP 批量生成 3D 高斯模型
基于 Apple SHARP 单图生成3D高斯模型工作流参数
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sharp.models import PredictorParams, create_predictor
from sharp.utils.io import convert_focallength, get_supported_image_extensions
from sharp.utils.gaussians import save_ply, unproject_gaussians

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger(__name__)

# ── 工作流参数（对应 ComfyUI 截图） ─────────────────────────
FOCAL_LENGTH_MM = 50.0      # focal_length_mm
LONGEST_SIDE    = 2048       # 图层工具 - 按宽高比缩放 V2
INTERNAL_SHAPE  = (1536, 1536)  # SHARP 内部推理分辨率（固定）
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = SCRIPT_DIR / "input"
MODEL_DIR  = SCRIPT_DIR / "model"
OUTPUT_DIR = SCRIPT_DIR / "output"


def find_model():
    """在 model 文件夹中查找 .pt 模型文件"""
    pts = list(MODEL_DIR.glob("*.pt"))
    if not pts:
        LOG.error("model/ 文件夹中未找到 .pt 模型文件")
        sys.exit(1)
    return pts[0]


def find_images():
    """在 input 文件夹中查找所有支持的图片"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic',
            '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.HEIC'}
    images = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix in exts:
            images.append(f)
    return images


def preprocess(image_path):
    """读取并预处理图片：最长边缩放到 LONGEST_SIDE，lanczos"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > LONGEST_SIDE:
        if w >= h:
            new_w = LONGEST_SIDE
            new_h = int(h * LONGEST_SIDE / w)
        else:
            new_h = LONGEST_SIDE
            new_w = int(w * LONGEST_SIDE / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img)


@torch.no_grad()
def predict(predictor, image_np, f_px, device):
    """单张图片推理，返回 Gaussians3D"""
    image_pt = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized = F.interpolate(
        image_pt[None],
        size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
        mode="bilinear",
        align_corners=True,
    )

    gaussians_ndc = predictor(image_resized, disparity_factor)

    intrinsics = torch.tensor([
        [f_px, 0, width / 2, 0],
        [0, f_px, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).float().to(device)

    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_SHAPE[0] / width
    intrinsics_resized[1] *= INTERNAL_SHAPE[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, INTERNAL_SHAPE
    )
    return gaussians, height, width


def main():
    # 检查目录
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        LOG.error("请将图片放入 input/ 文件夹")
        sys.exit(1)

    images = find_images()
    if not images:
        LOG.error("input/ 文件夹中未找到支持的图片文件")
        sys.exit(1)

    # 加载模型
    model_path = find_model()
    LOG.info(f"模型: {model_path.name}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    LOG.info(f"设备: {device}")
    LOG.info(f"焦距: {FOCAL_LENGTH_MM}mm | 最长边: {LONGEST_SIDE}px")
    LOG.info(f"图片数: {len(images)}")
    LOG.info("-" * 50)

    state_dict = torch.load(model_path, weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)

    OUTPUT_DIR.mkdir(exist_ok=True)

    total_start = time.time()
    success = 0
    fail = 0

    for i, img_path in enumerate(images, 1):
        LOG.info(f"[{i}/{len(images)}] {img_path.name}")
        t0 = time.time()

        try:
            image_np = preprocess(img_path)
            h, w = image_np.shape[:2]
            f_px = convert_focallength(w, h, FOCAL_LENGTH_MM)

            gaussians, height, width = predict(predictor, image_np, f_px, torch.device(device))

            out_path = OUTPUT_DIR / f"{img_path.stem}.ply"
            save_ply(gaussians, f_px, (height, width), out_path)

            elapsed = time.time() - t0
            size_mb = out_path.stat().st_size / (1024 * 1024)
            LOG.info(f"  -> {out_path.name} | {size_mb:.1f}MB | {elapsed:.1f}s")
            success += 1

        except Exception as e:
            elapsed = time.time() - t0
            LOG.error(f"  -> 失败: {e} | {elapsed:.1f}s")
            fail += 1

    total_elapsed = time.time() - total_start
    LOG.info("-" * 50)
    LOG.info(f"完成: {success} 成功, {fail} 失败 | 总耗时 {total_elapsed:.1f}s")
    if success > 0:
        LOG.info(f"平均 {total_elapsed / success:.1f}s/张")
    LOG.info(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
