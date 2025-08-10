import torch
from PIL import Image
from typing import Tuple, List

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[lite_clip] Using device: {DEVICE}")

# --- Try modern processor first, then fall back ---
processor = None
tokenizer = None
image_proc = None
model = None

try:
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("[lite_clip] Loaded CLIPModel + CLIPProcessor.")
except Exception as e:
    print(f"[lite_clip] Processor import failed: {e} -> falling back")
    try:
        # Older/newer APIs: CLIPImageProcessor (new) or CLIPFeatureExtractor (old)
        from transformers import CLIPModel, CLIPTokenizer
        try:
            from transformers import CLIPImageProcessor as _ImageProc
        except Exception:
            from transformers import CLIPFeatureExtractor as _ImageProc
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
        image_proc = _ImageProc.from_pretrained(MODEL_NAME)
        print("[lite_clip] Loaded CLIPModel + (Tokenizer, Image/Feature Processor).")
    except Exception as e2:
        print(f"[lite_clip] Fallback load failed: {e2}")
        model = None



VALID_COLORS = ["Black","White","Gray","Silver","Blue","Red","Brown","Gold","Green","Tan","Orange","Yellow"]
VALID_TYPES  = ["Car","Van","Pickup Truck","Freight"]

def _process_inputs(texts, pil_image):
    if processor is not None:
        return processor(text=texts, images=pil_image, return_tensors="pt", padding=True).to(DEVICE)
    # fallback path
    text_inputs  = tokenizer(texts, padding=True, return_tensors="pt")
    image_inputs = image_proc(images=pil_image, return_tensors="pt")
    # merge dicts and move to device
    merged = {**text_inputs, **image_inputs}
    return {k: v.to(DEVICE) for k, v in merged.items()}

def classify_vehicle_with_clip(
    vehicle_image: Image.Image,
    color_candidates: List[str] = VALID_COLORS,
    type_candidates: List[str]  = VALID_TYPES
) -> Tuple[str, str]:
    if model is None:
        return "N/A", "N/A"

    color_prompts = [f"a photo of a {c.lower()} vehicle" for c in color_candidates]
    type_prompts  = [f"a photo of a {t.lower()}"          for t in type_candidates]
    inputs = _process_inputs(color_prompts + type_prompts, vehicle_image)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image  # [1, N]
    n_colors = len(color_candidates)
    color_logits = logits[:, :n_colors]
    type_logits  = logits[:, n_colors:]
    best_color = color_candidates[color_logits.argmax(dim=1).item()]
    best_type  = type_candidates[type_logits.argmax(dim=1).item()]
    return best_color, best_type