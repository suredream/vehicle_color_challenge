#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_compare.py
Visual comparison of old vs new color/type from clip_summary.csv

Usage:
  PYTHONPATH=. python tools/validate_compare.py
"""

import pandas as pd
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Paths
ROOT = Path("/content/vehicle_color_challenge")
OUT_DIR = ROOT / "out"
THUMB_DIR = OUT_DIR / "thumbs"
SUMMARY_CSV = OUT_DIR / "clip_summary.csv"
OUTPUT_IMG = OUT_DIR / "visual_compare.png"

# Font (fallback to default if not found)
try:
    FONT = ImageFont.truetype("arial.ttf", 14)
except:
    FONT = ImageFont.load_default()

# Grid settings
COLS = 10
ROWS = 5
CELL_W = 100
CELL_H = 100
PADDING = 5
TEXT_H = 30  # space for 2 lines of text

def main():
    if not SUMMARY_CSV.exists():
        print(f"[validate_compare] No summary file found at {SUMMARY_CSV}")
        return

    df = pd.read_csv(SUMMARY_CSV)
    # Ensure we only keep changed results
    changed = df[(df["old_color"] != df["new_color"]) | (df["old_type"] != df["new_type"])]
    if changed.empty:
        print("[validate_compare] No changed records found.")
        return

    # Sample up to 50
    sample_df = changed.sample(min(50, len(changed)), random_state=42)

    # Prepare output canvas
    img_w = COLS * CELL_W
    img_h = ROWS * (CELL_H + TEXT_H)
    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, row in enumerate(sample_df.itertuples(index=False)):
        thumb_path = THUMB_DIR / f"{row.file}.png"
        if not thumb_path.exists():
            continue
        thumb = Image.open(thumb_path).convert("RGB")
        thumb = thumb.resize((CELL_W, CELL_H))

        col = idx % COLS
        row_idx = idx // COLS
        if row_idx >= ROWS:
            break

        x = col * CELL_W
        y = row_idx * (CELL_H + TEXT_H)

        canvas.paste(thumb, (x, y))

        # Annotate old/new values
        old_text = f"{row.old_color}/{row.old_type}"
        new_text = f"{row.new_color}/{row.new_type}"
        draw.text((x + 2, y + CELL_H + 2), old_text, fill="red", font=FONT)
        draw.text((x + 2, y + CELL_H + 15), new_text, fill="green", font=FONT)

    canvas.save(OUTPUT_IMG)
    print(f"[validate_compare] Saved visual comparison to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
