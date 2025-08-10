#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_check.py

随机抽取 50 个车辆结果，将其缩略图按网格绘制在一张图中，
并标注检测出的颜色和类型，用于快速目视检查。
"""

import random
from pathlib import Path
import cv2
import numpy as np

# 参数
OUT_DIR = Path("out")
THUMB_DIR = OUT_DIR / "thumbs"
LABEL_DIR = OUT_DIR / "labels"
OUTPUT_IMG = OUT_DIR / "visual_assess.png"

GRID_COLS = 10       # 每行显示多少张
MAX_SAMPLES = 50     # 最多显示多少张
THUMB_SIZE = 128     # 缩略图统一缩放大小 (正方形)

def load_label_info(txt_path):
    """读取标签文件，返回 (color, type)"""
    with open(txt_path, "r") as f:
        line = f.readline().strip()
        if not line:
            return "?", "?"
        parts = line.split()
        if len(parts) < 10:
            return "?", "?"
        color = parts[6]
        vtype = " ".join(parts[7:-2])  # 兼容"Pickup Truck"
        return color, vtype

def main():
    thumbs = sorted(THUMB_DIR.glob("*.png"))
    labels = sorted(LABEL_DIR.glob("*.txt"))

    # 建立 {stem: path} 索引
    label_map = {p.stem: p for p in labels}
    valid_samples = [t for t in thumbs if t.stem in label_map]

    if not valid_samples:
        print("No matching thumbs and labels found.")
        return

    # 随机抽样
    sample_files = random.sample(valid_samples, min(MAX_SAMPLES, len(valid_samples)))

    annotated_imgs = []
    for thumb_path in sample_files:
        color, vtype = load_label_info(label_map[thumb_path.stem])
        img = cv2.imread(str(thumb_path))
        if img is None:
            continue
        # 缩放到统一尺寸
        img = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE))

        # 在顶部绘制标签背景
        label_text = f"{color}, {vtype}"
        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (0, 0), (THUMB_SIZE, th + 4), (0, 0, 0), -1)
        cv2.putText(img, label_text, (2, th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        annotated_imgs.append(img)

    # 网格排列
    rows = (len(annotated_imgs) + GRID_COLS - 1) // GRID_COLS
    grid_img = np.zeros((rows * THUMB_SIZE, GRID_COLS * THUMB_SIZE, 3), dtype=np.uint8)

    for idx, img in enumerate(annotated_imgs):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        grid_img[r*THUMB_SIZE:(r+1)*THUMB_SIZE, c*THUMB_SIZE:(c+1)*THUMB_SIZE] = img

    cv2.imwrite(str(OUTPUT_IMG), grid_img)
    print(f"Saved visual assessment image to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
