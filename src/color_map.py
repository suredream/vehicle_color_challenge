import numpy as np
import cv2

PALETTE = [
    "Black","White","Gray","Silver","Blue","Red","Brown","Gold","Green","Tan","Orange","Yellow"
]

def hsv_to_color(h, s, v):
    # grayscale buckets first
    if v < 50 and s < 80:
        return "Black"
    if v > 210 and s < 40:
        return "White"
    if s < 45 and 50 <= v <= 210:
        return "Silver" if v >= 160 else "Gray"
    # hues
    if (h <= 10 or h >= 170) and s > 70:
        return "Red"
    if 10 < h <= 20 and s > 70:
        return "Orange"
    if 20 < h <= 35 and s > 70 and v > 170:
        return "Yellow"
    if 20 < h <= 35 and 20 <= s <= 80:
        return "Tan" if v > 120 else "Brown"
    if 10 < h <= 25 and v < 150:
        return "Brown"
    if 35 < h <= 85 and s > 60:
        return "Green"
    if 85 < h <= 125 and s > 60:
        return "Blue"
    # fallback
    if v < 80:
        return "Black"
    return "Gray"

def estimate_color(image_bgra, mask):
    bgr = image_bgra[:,:,:3] if image_bgra.shape[2]==4 else image_bgra
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = mask.astype(bool)
    if m.sum()==0:
        return "Gray"
    h = hsv[:,:,0][m].astype(np.uint8)
    s = hsv[:,:,1][m].astype(np.uint8)
    v = hsv[:,:,2][m].astype(np.uint8)
    H = int(np.median(h)); S = int(np.median(s)); V = int(np.median(v))
    return hsv_to_color(H,S,V)
