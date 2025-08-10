import argparse
from pathlib import Path
from src.pipeline import process_dataset

def main():
    ap = argparse.ArgumentParser(description="Vehicle Color Detection Challenge - Heuristic Baseline")
    ap.add_argument("--images", type=Path, required=True, help="Directory with PNG images")
    ap.add_argument("--labels", type=Path, required=True, help="Directory with TXT labels")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for thumbs and txts")
    ap.add_argument("--config", type=Path, default=Path("configs/dev.yaml"), help="YAML config path")
    args = ap.parse_args()
    process_dataset(args.images, args.labels, args.out, args.config)

if __name__ == "__main__":
    main()
