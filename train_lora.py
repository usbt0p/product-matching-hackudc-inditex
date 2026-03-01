# Shield 1: anti-fragmentation env var MUST be first, before torch import
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
import requests
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from dotenv import load_dotenv

from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRODUCTS_CACHE = Path("data/products_cache")
PRODUCTS_CACHE.mkdir(parents=True, exist_ok=True)

BUNDLE_IMG_DIR = Path("data/bundles")
TRAIN_CSV      = Path("data_csvs/bundles_product_match_train.csv")
PRODUCT_CSV    = Path("data_csvs/product_dataset.csv")
LORA_OUT_DIR   = Path("gr_lite_lora")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_ts(url: str) -> int:
    m = re.search(r"ts=(\d+)", url or "")
    return int(m.group(1)) if m else 0


def download_product_image(url: str, pid: str) -> Path | None:
    """Downloads a product image to the local cache; returns path or None."""
    dest = PRODUCTS_CACHE / f"{pid}.jpg"
    if dest.exists():
        return dest
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return dest
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BundleProductDataset(Dataset):
    """Returns (bundle_crop_tensor, product_img_tensor) pairs."""

    def __init__(self, processor, yolo_model, max_samples: int = 5000):
        df_train   = pd.read_csv(TRAIN_CSV)
        df_prod    = pd.read_csv(PRODUCT_CSV)
        url_map    = df_prod.set_index("product_asset_id")["product_image_url"].to_dict()

        # Collect valid pairs
        self.pairs = []
        for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Building pairs"):
            bid = row["bundle_asset_id"]
            pid = row["product_asset_id"]
            bundle_path = BUNDLE_IMG_DIR / f"{bid}.jpg"
            if not bundle_path.exists():
                continue
            url = url_map.get(pid)
            if not url:
                continue
            prod_path = download_product_image(url, str(pid))
            if prod_path is None:
                continue
            self.pairs.append((str(bundle_path), str(prod_path), bid))
            if len(self.pairs) >= max_samples:
                break

        print(f"Dataset ready: {len(self.pairs)} pairs")
        self.processor  = processor
        self.yolo_model = yolo_model
        self.fallback_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _get_best_yolo_crop(self, img_path: str, img: Image.Image) -> Image.Image:
        try:
            results = self.yolo_model.predict(img_path, verbose=False)
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                return img
            # Pick the largest box
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            best  = boxes[int(np.argmax(areas))]
            x1, y1, x2, y2 = map(int, best)
            pad = 15
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad); y2 = min(img.height, y2 + pad)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                return img.crop((x1, y1, x2, y2))
        except Exception:
            pass
        return img

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        bundle_path, prod_path, bid = self.pairs[idx]
        try:
            bundle_img = Image.open(bundle_path).convert("RGB")
            crop       = self._get_best_yolo_crop(bundle_path, bundle_img)
            prod_img   = Image.open(prod_path).convert("RGB")

            crop_t = self.processor(images=crop,     return_tensors="pt")["pixel_values"].squeeze(0)
            prod_t = self.processor(images=prod_img, return_tensors="pt")["pixel_values"].squeeze(0)
            return crop_t, prod_t
        except Exception:
            dummy = torch.zeros(3, 224, 224)
            return dummy, dummy


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

def apply_lora_to_vit(base_model):
    """Freeze base, enable gradient checkpointing, inject LoRA."""
    for param in base_model.parameters():
        param.requires_grad = False

    # Shield 2: gradient checkpointing
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Real module names discovered by inspecting the GR-Lite model graph:
        # attention: q_proj, k_proj, v_proj  |  mlp: up_proj, down_proj
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(base_model, config)
    lora_model.print_trainable_parameters()
    return lora_model.to("cuda")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lora_bulletproof(lora_model, dataloader, epochs: int = 4, accum_steps: int = 8):
    """Shield 3: Gradient accumulation + mixed-precision AMP."""
    optimizer   = torch.optim.AdamW(lora_model.parameters(), lr=2e-4)
    scaler      = GradScaler()
    temperature = 0.05

    lora_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for batch_idx, (crops, cats) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            crops = crops.to("cuda", non_blocking=True)
            cats  = cats.to("cuda",  non_blocking=True)

            with autocast():
                embs_crop = lora_model(pixel_values=crops).last_hidden_state.mean(dim=1)
                embs_cat  = lora_model(pixel_values=cats ).last_hidden_state.mean(dim=1)

                embs_crop = F.normalize(embs_crop, p=2, dim=-1)
                embs_cat  = F.normalize(embs_cat,  p=2, dim=-1)

                sim_matrix = torch.matmul(embs_crop, embs_cat.T) / temperature
                labels     = torch.arange(embs_crop.size(0), device="cuda")

                # Divide by accum_steps so the effective gradient = full-batch gradient
                loss = F.cross_entropy(sim_matrix, labels) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps  # rescale for display

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")
        gc.collect()

    return lora_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load GR-Lite base model
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    print("Loading GR-Lite base model...")
    config    = AutoConfig.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-lvd1689m", token=token, trust_remote_code=True
    )
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-lvd1689m", token=token, trust_remote_code=True
    )
    base_model = AutoModel.from_config(config, trust_remote_code=True)

    # Load pre-trained GR-Lite weights
    d        = torch.load("gr_lite.pt", map_location="cpu")
    gr_dict  = {}
    for k, v in d.items():
        if k.startswith("model.model."):
            gr_dict[k.replace("model.model.", "", 1)] = v
        elif k.startswith("model."):
            gr_dict[k.replace("model.", "", 1)] = v
        else:
            gr_dict[k] = v
    base_model.load_state_dict(gr_dict, strict=False)
    print("GR-Lite weights loaded.")

    # Apply LoRA
    lora_model = apply_lora_to_vit(base_model)

    # Load YOLO for cropping
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    v8_path    = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
    yolo_model = YOLO(v8_path)
    print("YOLO loaded for cropping.")

    # Build dataset and loader
    # Shield 3a: small batch_size=8 to guarantee no OOM
    dataset    = BundleProductDataset(processor, yolo_model, max_samples=5000)
    dataloader = DataLoader(
        dataset,
        batch_size=8,          # effective batch = 8 * accum_steps = 64
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,        # avoid single-sample batches breaking InfoNCE
    )

    if len(dataset) == 0:
        print("No valid pairs found. Check CDN connectivity or local data paths.")
        return

    # Train
    lora_model = train_lora_bulletproof(lora_model, dataloader, epochs=4, accum_steps=8)

    # Save LoRA adapter (only the delta weights, ~50 MB)
    LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(LORA_OUT_DIR))
    print(f"LoRA adapter saved to {LORA_OUT_DIR}/")
    print("Run inference with: model = PeftModel.from_pretrained(base_model, 'gr_lite_lora')")


if __name__ == "__main__":
    main()
