from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

from .geom import obb_to_polygon_pixels, polygon_mask_from_shape, rotate_crop_square
from .color_map import estimate_color
from .heuristics import infer_type_from_wh, estimate_shadow, estimate_occlusion
from .io_utils import read_label_txt, write_vehicle_txt, save_thumbnail

def _clamp01(x): 
    return float(min(1.0, max(0.0, x)))

def process_image(img_path: Path, lbl_path: Path, out_dir: Path, cfg: dict):
    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    H, W = image.shape[:2]
    recs = read_label_txt(lbl_path)

    polys, masks = [], []
    for r in recs:
        poly = obb_to_polygon_pixels(r['xc'], r['yc'], r['w'], r['h'], r['theta'], W, H, normalized=True)
        mask = polygon_mask_from_shape(image.shape, poly)
        polys.append(poly); masks.append(mask)

    img_id = img_path.stem
    for i, r in enumerate(recs):
        poly = polys[i]; mask = masks[i]
        thumb = rotate_crop_square(image, poly, pad=int(cfg['thumbnail']['pad']), out_size=int(cfg['thumbnail']['size']))
        color = estimate_color(image, mask) if cfg['color']['enabled'] else 'Gray'
        w_px = r['w']*W; h_px = r['h']*H
        vtype = infer_type_from_wh(
            w_px, h_px, meters_per_px=float(cfg['gsd']['meters_per_px']),
            freight_long_edge_m=float(cfg['type']['freight_long_edge_m']),
            freight_aspect_min=float(cfg['type']['freight_aspect_min']),
            van_long_edge_m=float(cfg['type']['van_long_edge_m']),
            van_area_m2=float(cfg['type']['van_area_m2']),
            pickup_aspect_min=float(cfg['type']['pickup_aspect_min']),
            pickup_long_edge_min_m=float(cfg['type']['pickup_long_edge_min_m']),
            pickup_long_edge_max_m=float(cfg['type']['pickup_long_edge_max_m']),
        ) if cfg['type']['enabled'] else 'Car'
        shadow = estimate_shadow(image, mask, median_alpha=float(cfg['shadow']['median_alpha'])) if cfg['shadow']['enabled'] else 0.0
        occlusion = estimate_occlusion(i, masks) if cfg['occlusion']['enabled'] else 0.0

        thumb_path = out_dir / "thumbs" / f"{img_id}-{i}.png"
        txt_path = out_dir / "labels" / f"{img_id}-{i}.txt"
        save_thumbnail(thumb_path, thumb)

        r_out = dict(r)  # 复制一份用于输出
        r_out["xc"] = _clamp01(r_out["xc"])
        r_out["yc"] = _clamp01(r_out["yc"])
        r_out["w"]  = _clamp01(r_out["w"])
        r_out["h"]  = _clamp01(r_out["h"])
        write_vehicle_txt(txt_path, {
            **r_out, "color": color, "type": vtype, "occlusion": occlusion, "shadow": shadow
        })

def process_dataset(images_dir: Path, labels_dir: Path, out_dir: Path, cfg_path: Path):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thumbs").mkdir(exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)

    # match pairs
    imgs = sorted([p for p in images_dir.glob("*.png")])
    for img_path in tqdm(imgs):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        process_image(img_path, lbl_path, out_dir, cfg)
