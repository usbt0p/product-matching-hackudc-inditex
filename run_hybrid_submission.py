import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

def load_gr_lite(device):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("No HF_TOKEN found in .env, skipping GR-Lite")
        return None, None
    
    try:
        config = AutoConfig.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', token=token.strip(), trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', token=token.strip(), trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        d = torch.load('gr_lite.pt', map_location='cpu')
        
        gr_dict = {}
        for k, v in d.items():
            if k.startswith('model.model.'):
                gr_dict[k.replace('model.model.', '', 1)] = v
            elif k.startswith('model.'):
                gr_dict[k.replace('model.', '', 1)] = v
            else:
                gr_dict[k] = v
                
        model.load_state_dict(gr_dict, strict=False)
        model = model.to(device)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"Failed to assemble GR-Lite: {e}")
        return None, None

def get_embeddings(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embs = outputs.pooler_output
        else:
            embs = outputs.last_hidden_state.mean(dim=1)
    
    return embs.cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load YOLOv8-Clothing
    print("\nLoading YOLOv8-Clothing model...")
    try:
        from huggingface_hub import hf_hub_download
        v8_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
        yolo_model = YOLO(v8_model_path)
    except Exception as e:
        print(f"Failed to load YOLOv8-Clothing: {e}")
        return
    
    # Load GR-Lite
    print("Loading GR-Lite model (DINOv3 backbone)...")
    gr_model, gr_processor = load_gr_lite(device)
    if not gr_model:
        return
    
    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    
    ordered_catalog_ids = df_products['product_asset_id'].tolist()
    test_bundle_ids = df_test['bundle_asset_id'].unique().tolist()
    
    # 1. Load Precomputed Catalog
    catalog_emb_path = "catalog_grlite_embeddings.npy"
    catalog_ids_path = "valid_grlite_ids.npy"
    
    if os.path.exists(catalog_emb_path) and os.path.exists(catalog_ids_path):
        print("\nLoading precomputed GR-Lite catalog embeddings...")
        catalog_embeddings = np.load(catalog_emb_path)
        valid_catalog_ids = np.load(catalog_ids_path, allow_pickle=True).tolist()
    else:
        print("\nGR-Lite catalog embeddings missing. Please run run_gr_lite.py first to cache them.")
        return

    # Normalize catalog embeddings
    catalog_norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    catalog_norms[catalog_norms == 0] = 1e-10
    normalized_catalog = catalog_embeddings / catalog_norms

    # 2. Process Test Bundles
    print("\nExtracting Hybrid Embeddings (Global + YOLO Crops) and Predicting...")
    
    submission_rows_top15 = []
    
    w_global = 0.4
    w_local = 0.6
    
    for i in tqdm(range(len(test_bundle_ids)), desc="Test Bundles"):
        bid = test_bundle_ids[i]
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
        # Global Embroidery
        global_emb = get_embeddings(gr_model, gr_processor, [img], device)[0]
        g_norm = np.linalg.norm(global_emb)
        if g_norm == 0: g_norm = 1e-10
        normalized_global = global_emb / g_norm
        sim_global = np.dot(normalized_catalog, normalized_global).flatten()
        
        # Local Embeddings
        sim_local_aggregated = np.zeros_like(sim_global)
        results = yolo_model.predict(img_path, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        valid_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                crop = img.crop((x1, y1, x2, y2))
                valid_crops.append(crop)
                
        if valid_crops:
            # Batch process crops to speed things up
            batch_size = 16
            all_crop_sims = []
            
            for j in range(0, len(valid_crops), batch_size):
                batch_crops = valid_crops[j:j+batch_size]
                crop_embs = get_embeddings(gr_model, gr_processor, batch_crops, device)
                
                for crop_emb in crop_embs:
                    c_norm = np.linalg.norm(crop_emb)
                    if c_norm == 0: c_norm = 1e-10
                    normalized_crop = crop_emb / c_norm
                    crop_sim = np.dot(normalized_catalog, normalized_crop).flatten()
                    sim_local_aggregated = np.maximum(sim_local_aggregated, crop_sim)
        else:
            sim_local_aggregated = sim_global
            
        # Combine
        final_scores = (w_global * sim_global) + (w_local * sim_local_aggregated)
        
        # Get Top 15
        top_indices = np.argsort(final_scores)[::-1][:15]
        
        for idx in top_indices:
            submission_rows_top15.append({"bundle_asset_id": bid, "product_asset_id": valid_catalog_ids[idx]})

    # 3. Save submissions
    df_sub_15 = pd.DataFrame(submission_rows_top15)
    df_sub_15.to_csv("submission_hybrid_grlite_top15.csv", index=False)
    
    print(f"\nSaved Hybrid Top-15 submission to submission_hybrid_grlite_top15.csv! ({len(df_sub_15)} rows)")

if __name__ == "__main__":
    main()
