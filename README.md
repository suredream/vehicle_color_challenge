# Vehicle Color Detection Challenge — Heuristic Baseline (No Training)

This repository provides a **2-hour, training-free baseline** to generate the required challenge outputs:

- `<img-id>-<vehicle-idx>.png`: square thumbnail of each vehicle (upright)
- `<img-id>-<vehicle-idx>.txt`: text file per vehicle with the following line format:

```
<class> <x-center> <y-center> <width> <height> <theta> <color> <type> <occlusion> <shadow>
```

Where:
- The first 6 values are **copied from your label** (normalized le90 OBB): class, xc, yc, w, h, theta.
- `color` is one of **{Black, White, Gray, Silver, Blue, Red, Brown, Gold, Green, Tan, Orange, Yellow}**.
- `type` is one of **{Car, Van, Pickup Truck, Freight}**.
- `occlusion` and `shadow` are floats in **[0, 1]**.

> This baseline is **annotation-driven** (no detector inference at runtime). It uses deterministic geometry, HSV-based color estimation, and light heuristics for type/shadow/occlusion. It is intended to maximize deliverability within the time-box while keeping clean interfaces for future upgrades.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data Layout

```
data/
  images/
    <img-id>.png
    ...
  labels/
    <img-id>.txt   # rows: "<class> <xc> <yc> <w> <h> <theta>"
```

- Coordinates are **normalized [0,1]** relative to the image width/height.
- `theta` is assumed to be **degrees** (le90). If your labels use radians, convert them ahead of time.

## 3. Run

```bash
python main.py --images data/images --labels data/labels --out out --config configs/dev.yaml
```

Outputs:
```
out/
  thumbs/
    <img-id>-<idx>.png
  labels/
    <img-id>-<idx>.txt
```

## 4. Validation

```bash
python tools/validate_outputs.py out
```
This checks:
- one thumbnail per label and vice versa
- numeric ranges for normalized fields
- allowed enums for `color` and `type`

## 5. Configuration

Key knobs are in `configs/dev.yaml`:
- `thumbnail.size` and `thumbnail.pad`
- `gsd.meters_per_px` (default 0.3 for 30 cm GSD)
- `shadow.median_alpha` (adaptive thresholding on V channel)
- Type thresholds in meters (long-edge, aspect, area)

## 6. Method Overview

- **Geometry**: Convert normalized le90 OBB to pixel-space polygon; rotate-upright crop via perspective transform; square thumbnail.
- **Color**: HSV medians within the OBB mask mapped into the 12-class palette (tunable thresholds).
- **Type**: Heuristics from object scale (meters) and aspect ratio — quick and explainable.
- **Shadow**: Ratio of dark pixels under a local-adaptive threshold in the V channel.
- **Occlusion**: Mask-overlap proxy across vehicles within the same image.

## 7. Roadmap (Optional Upgrades)

- Replace heuristics with a **small classifier** (e.g., CLIP features + linear probe) for vehicle type.
- Refine shadow with illumination normalization and guided filtering.
- Improve occlusion via instance masks (SAM2/Mask2Former) and neighbor reasoning.
- Add **Great Expectations** or similar for data contracts; add unit tests.

---

**Authoring Notes**
- This baseline favors **determinism, simplicity, and speed**.
- All modules are small and isolated (`src/geom.py`, `src/color_map.py`, `src/heuristics.py`, `src/io_utils.py`, `src/pipeline.py`).

Happy building!
