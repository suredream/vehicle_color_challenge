import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, List

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[lite_clip] Using device: {DEVICE}")

try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("[lite_clip] CLIP model and processor loaded successfully.")
except Exception as e:
    print(f"[lite_clip] Error loading model: {e}")
    model = None
    processor = None

VALID_COLORS = [
    "Black", "White", "Gray", "Silver", "Blue", "Red",
    "Brown", "Gold", "Green", "Tan", "Orange", "Yellow"
]
VALID_TYPES = ["Car", "Van", "Pickup Truck", "Freight"]

def classify_vehicle_with_clip(
    vehicle_image: Image.Image,
    color_candidates: List[str] = VALID_COLORS,
    type_candidates: List[str] = VALID_TYPES
) -> Tuple[str, str]:
    """Zero-shot color & type classification for a vehicle thumbnail using CLIP.

    Returns (color, type). If model/processor failed to load, returns ("N/A", "N/A").
    """
    if not model or not processor:
        return "N/A", "N/A"

    # Build prompts
    color_prompts = [f"a photo of a {c.lower()} vehicle" for c in color_candidates]
    type_prompts = [f"a photo of a {t.lower()}" for t in type_candidates]

    # Preprocess
    inputs = processor(
        text=color_prompts + type_prompts,
        images=vehicle_image,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image  # [1, N]
    n_colors = len(color_candidates)
    color_logits = logits[:, :n_colors]
    type_logits = logits[:, n_colors:]

    best_color_idx = color_logits.argmax(dim=1).item()
    best_type_idx = type_logits.argmax(dim=1).item()

    return color_candidates[best_color_idx], type_candidates[best_type_idx]