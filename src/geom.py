import math
import numpy as np
import cv2

def obb_to_polygon_pixels(xc, yc, w, h, theta_deg, W, H, normalized=True):
    """Convert normalized le90 OBB to pixel-space polygon (4x2).
    Assumes theta is in degrees (as in many label formats)."""
    if normalized:
        xc *= W; yc *= H; w *= W; h *= H
    theta = math.radians(theta_deg)
    dx = w/2.0; dy = h/2.0
    corners = np.array([[-dx, -dy],[ dx, -dy],[ dx,  dy],[-dx,  dy]], dtype=np.float32)
    rot = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta),  math.cos(theta)]], dtype=np.float32)
    rotated = corners @ rot.T
    translated = rotated + np.array([xc, yc], dtype=np.float32)
    return translated

def polygon_mask_from_shape(shape_hw, poly):
    H, W = shape_hw[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

def rotate_crop_square(image, poly, pad=4, out_size=160):
    """Rotate the polygon upright and return a square crop."""
    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    side = int(max(rect[1][0], rect[1][1]) + 2*pad)
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(image, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if out_size and out_size>0:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop
