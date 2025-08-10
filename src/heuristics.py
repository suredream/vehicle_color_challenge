import numpy as np
import cv2

def infer_type_from_wh(w_px, h_px, meters_per_px=0.3,
                       freight_long_edge_m=9.0, freight_aspect_min=2.5,
                       van_long_edge_m=6.5, van_area_m2=10.0,
                       pickup_aspect_min=2.2, pickup_long_edge_min_m=4.5, pickup_long_edge_max_m=7.5):
    long_edge = max(w_px, h_px) * meters_per_px
    area_m2 = (w_px*meters_per_px)*(h_px*meters_per_px)
    aspect = max(w_px, h_px) / max(1.0, min(w_px, h_px))
    if long_edge >= freight_long_edge_m and aspect >= freight_aspect_min:
        return "Freight"
    if long_edge >= van_long_edge_m and area_m2 >= van_area_m2:
        return "Van"
    if aspect >= pickup_aspect_min and (pickup_long_edge_min_m <= long_edge < pickup_long_edge_max_m):
        return "Pickup Truck"
    return "Car"

def estimate_shadow(image_bgra, mask, median_alpha=0.7):
    bgr = image_bgra[:,:,:3] if image_bgra.shape[2]==4 else image_bgra
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    m = mask.astype(bool)
    if m.sum()==0:
        return 0.0
    v_vals = V[m].astype(np.float32)
    med = float(np.median(v_vals))
    thr = med * median_alpha
    return float(np.clip((v_vals < thr).mean(), 0.0, 1.0))

def estimate_occlusion(idx, masks):
    area = float((masks[idx]>0).sum())
    if area==0.0: return 0.0
    overlap = 0.0
    for j, m in enumerate(masks):
        if j==idx: continue
        inter = (masks[idx]>0) & (m>0)
        overlap += float(inter.sum())
    return float(np.clip(overlap/area, 0.0, 1.0))
