from pathlib import Path

def read_label_txt(path: Path):
    recs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            cls, xc, yc, w, h, theta = line.strip().split()
            recs.append({
                "idx": i,
                "cls": int(cls),
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h),
                "theta": float(theta),
            })
    return recs

def write_vehicle_txt(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            f"{record['cls']} {record['xc']:.6f} {record['yc']:.6f} {record['w']:.6f} {record['h']:.6f} {record['theta']:.6f} "
            f"{record['color']} {record['type']} {record['occlusion']:.3f} {record['shadow']:.3f}\n"
        )

def save_thumbnail(path: Path, image):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

def list_images_and_labels(images_dir: Path, labels_dir: Path):
    imgs = sorted([p for p in images_dir.glob("*.png")])
    pairs = []
    for p in imgs:
        lbl = labels_dir / (p.stem + ".txt")
        if lbl.exists():
            pairs.append((p, lbl))
    return pairs
