#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply OpenCLIP zero-shot classification to update color/type for generated labels.

Usage:
  PYTHONPATH=. python -m tools.apply_openclip [--overwrite-metrics]
"""

import csv
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

from lite_openclip import (
    classify_vehicle_with_openclip, VALID_COLORS, VALID_TYPES, model
)

OUT_DIR = Path("out")
THUMB_DIR = OUT_DIR / "thumbs"
LABEL_DIR = OUT_DIR / "labels"
SUMMARY_CSV = OUT_DIR / "clip_summary.csv"

def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) < 10:
        raise ValueError("Bad label format")
    cls, xc, yc, w, h, theta = parts[:6]
    color = parts[6]
    occ = parts[-2]
    shd = parts[-1]
    vtype = " ".join(parts[7:-2])  # handle "Pickup Truck"
    return {
        "cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "theta": theta,
        "color": color, "type": vtype, "occlusion": occ, "shadow": shd
    }

def format_label_line(rec):
    return (f"{rec['cls']} {rec['xc']} {rec['yc']} {rec['w']} {rec['h']} "
            f"{rec['theta']} {rec['color']} {rec['type']} {rec['occlusion']} {rec['shadow']}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite-metrics", action="store_true",
                    help="also overwrite occlusion/shadow to 0.0")
    args = ap.parse_args()

    if model is None:
        print("[apply_openclip] OpenCLIP model unavailable. No labels will be changed.")
        return

    thumbs = sorted(THUMB_DIR.glob("*.png"))
    labels = {p.stem: p for p in LABEL_DIR.glob("*.txt")}
    rows = []

    for t in tqdm(thumbs):
        stem = t.stem
        if stem not in labels:
            continue
        lbl = labels[stem]
        img = Image.open(t).convert("RGB")

        pred_color, pred_type = classify_vehicle_with_openclip(img)

        rec = parse_label_line(lbl.read_text())
        old_color, old_type = rec["color"], rec["type"]

        changed = False
        if pred_color in VALID_COLORS and pred_color != rec["color"]:
            rec["color"] = pred_color
            changed = True
        if pred_type in VALID_TYPES and pred_type != rec["type"]:
            rec["type"] = pred_type
            changed = True
        if args.overwrite_metrics:
            rec["occlusion"] = "0.0"
            rec["shadow"] = "0.0"

        if changed or args.overwrite_metrics:
            lbl.write_text(format_label_line(rec))
            rows.append({
                "file": stem,
                "old_color": old_color, "new_color": rec["color"],
                "old_type": old_type, "new_type": rec["type"]
            })

    if rows:
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file","old_color","new_color","old_type","new_type"])
            w.writeheader(); w.writerows(rows)
        print(f"[apply_openclip] Updated {len(rows)} files. Summary: {SUMMARY_CSV}")
    else:
        print("[apply_openclip] No changes made (predictions matched existing labels).")

if __name__ == "__main__":
    main()
