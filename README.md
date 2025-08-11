# Vehicle Color Detection Challenge

This repository provides somethoughts to generate the required challenge outputs:

## 1. Codebase Environment

Open `vehicle_color_challenge.ipynb` in colab, run the cells in the order to prepare the environment

## 2. Approach overview: heuristics and openclip intergration

### **Heuristic Approach**

- **Advantages**
  
- Very fast, no extra training needed.
  
- Easy to implement and tune thresholds.
  
- Works offline without additional labeled data.
  
- **Disadvantages**
  
- Struggles with edge cases (e.g., shadows, glare, multi-colored vehicles).
  
- Limited adaptability to new environments without manual rule changes.
  
- Performance heavily depends on image conditions.
  

```
# processing
!python main.py --images /content/data --labels /content/data --out out --config configs/dev.yaml

# Validation checks:
# - one thumbnail per label and vice versa
# - numeric ranges for normalized fields
# - allowed enums for `color` and `type`

!python tools/validate_outputs.py out

# generate a layout png for easy visual check
!python tools/validate_check.py out

```

### **OpenCLIP Integration**

- The OpenCLIP model and prompts are zero-shot, meaning no fine-tuning is required.
  
- Classification accuracy can be improved by refining text prompts or using larger OpenCLIP variants.
  

```bash
# Conduct zero-shot classification using openclip:
%cd /content/vehicle_color_challenge
!python tools/apply_lite_openclip.py --overwrite-metrics

# Review changes in CSV:
head out/clip_summary.csv

# Generate comparison visualization:
# visiualize the results which are different from the Heuristic
%cd /content/vehicle_color_challenge
!python tools/validate_compare.py 


```

### light classifier with ReDet and ROI header

`src/ReDet_roi_classifier.py` implements a research-oriented pipeline for vehicle analysis using the MMRotate

library. It is designed to first detect vehicles with oriented bounding boxes using a pre-trained model (like ReDet) and then extract internal deep features from the model's Region of Interest (RoI) head.

The core idea is that these deep features are a richer, more robust representation of the vehicles than raw pixels, making them ideal for training lightweight downstream classifiers for tasks like color and type identification.

However, this script requires a full, correctly installed MMRotate environment and the corresponding model configuration and checkpoint files to run, which requires more time to handle and not fully implemented yet.

The ROI header approach remains a superior strategy for building a production-grade, high-accuracy computer vision system.

Please check `src/ReDet_roi_classifier.py` for more details.