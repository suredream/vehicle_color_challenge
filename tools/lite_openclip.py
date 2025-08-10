# src/lite_clip_openclip.py
import torch
from PIL import Image
from typing import Tuple, List
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"          # lightweight & widely available
PRETRAIN  = "openai"             # weights tag in open_clip
IMAGE_SIZE = 224                 # default preprocess size

print(f"[lite_clip_openclip] Using device: {DEVICE}")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAIN, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    print("[lite_clip_openclip] OpenCLIP model & preprocess loaded.")
except Exception as e:
    print(f"[lite_clip_openclip] Failed to load OpenCLIP: {e}")
    model = None
    preprocess = None
    tokenizer = None

VALID_COLORS = ["Black","White","Gray","Silver","Blue","Red","Brown","Gold","Green","Tan","Orange","Yellow"]
VALID_TYPES  = ["Car","Van","Pickup Truck","Freight"]

def _encode_text(prompts: List[str]) -> torch.Tensor:
    text = tokenizer(prompts)
    with torch.no_grad():
        te = model.encode_text(text.to(DEVICE))
        te = te / te.norm(dim=-1, keepdim=True)
    return te

def _encode_image(img: Image.Image) -> torch.Tensor:
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        ie = model.encode_image(img_t)
        ie = ie / ie.norm(dim=-1, keepdim=True)
    return ie

def classify_vehicle_with_openclip(
    vehicle_image: Image.Image,
    color_candidates: List[str] = VALID_COLORS,
    type_candidates:  List[str] = VALID_TYPES
) -> Tuple[str, str]:
    if model is None or preprocess is None or tokenizer is None:
        return "N/A", "N/A"

    # Build zero-shot prompts
    color_prompts = [f"a photo of a {c.lower()} vehicle" for c in color_candidates]
    type_prompts  = [f"a photo of a {t.lower()}"          for t in type_candidates]

    # Encode once
    img_emb   = _encode_image(vehicle_image.convert("RGB"))
    color_txt = _encode_text(color_prompts)
    type_txt  = _encode_text(type_prompts)

    # Cosine similarity via dot product on normalized embeddings
    with torch.no_grad():
        color_scores = img_emb @ color_txt.T   # [1, C]
        type_scores  = img_emb @ type_txt.T    # [1, T]

    best_color = color_candidates[int(color_scores.argmax(dim=1).item())]
    best_type  = type_candidates[int(type_scores.argmax(dim=1).item())]
    return best_color, best_type
