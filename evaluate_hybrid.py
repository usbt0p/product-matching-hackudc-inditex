import os
import random
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
    
    # 1. Load YOLOv8-Clothing Model
    print("\nLoading YOLOv8-Clothing model...")
    try:
        from huggingface_hub import hf_hub_download
        v8_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
        yolo_model = YOLO(v8_model_path)
    except Exception as e:
        print(f"Failed to load YOLOv8-Clothing: {e}")
        yolo_model = None
    
    # 2. Load GR-Lite Model
    print("Loading GR-Lite model and processor...")
    gr_model, gr_processor = load_gr_lite(device)
    if not gr_model:
        return
        
    print("\nLoading datasets for Hybrid Train subset evaluation...")
    df_train = pd.read_csv('data_csvs/bundles_product_match_train.csv')
    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    
    all_products = df_products['product_asset_id'].unique().tolist()
    
    # 3. Setup Mini-Catalog (50 Bundles)
    random.seed(42)
    bundle_pool = df_train['bundle_asset_id'].unique().tolist()
    sample_bundles = random.sample(bundle_pool, 50)
    
    gt_df = df_train[df_train['bundle_asset_id'].isin(sample_bundles)]
    gt_products = gt_df['product_asset_id'].unique().tolist()
    
    distractors_needed = 1000 - len(gt_products)
    remaining_products = list(set(all_products) - set(gt_products))
    distractors = random.sample(remaining_products, distractors_needed)
    
    eval_catalog_ids = gt_products + distractors
    random.shuffle(eval_catalog_ids)
    
    print(f"Selected {len(sample_bundles)} bundles.")
    print(f"Built mini-catalog of size {len(eval_catalog_ids)} ({len(gt_products)} ground truths + {distractors_needed} distractors).")
    
    # 4. Embed the mini-catalog (Products)
    print("\nEmbedding Mini-Catalog...")
    catalog_embeddings = []
    valid_catalog_ids = []
    
    batch_size = 16
    batch_imgs = []
    batch_ids = []
    
    for pid in tqdm(eval_catalog_ids, desc="Catalog"):
        img_path = os.path.join("data", "products", f"{pid}.jpg")
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(img)
            batch_ids.append(pid)
        except Exception:
            pass
            
        if len(batch_imgs) >= batch_size:
            embs = get_embeddings(gr_model, gr_processor, batch_imgs, device)
            catalog_embeddings.extend(embs)
            valid_catalog_ids.extend(batch_ids)
            batch_imgs = []
            batch_ids = []
            
    if batch_imgs:
        embs = get_embeddings(gr_model, gr_processor, batch_imgs, device)
        catalog_embeddings.extend(embs)
        valid_catalog_ids.extend(batch_ids)
        
    catalog_embeddings = np.array(catalog_embeddings)
    # Normalize catalog embeddings
    catalog_norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    catalog_norms[catalog_norms == 0] = 1e-10
    normalized_catalog = catalog_embeddings / catalog_norms
    
    # 5. Hybrid Evaluation Loop
    print("\nExtracting Bundle Crop + Global Embeddings & Evaluating...")
    top1_correct = 0
    top5_correct = 0
    top15_correct = 0
    valid_eval_bundles = 0
    
    w_global = 0.4
    w_local = 0.6
    
    for idx, bid in enumerate(tqdm(sample_bundles, desc="Hybrid Bundles")):
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
        # 5a. Global Similarity
        global_emb = get_embeddings(gr_model, gr_processor, [img], device)[0]
        g_norm = np.linalg.norm(global_emb)
        if g_norm == 0: g_norm = 1e-10
        normalized_global = global_emb / g_norm
        sim_global = np.dot(normalized_catalog, normalized_global).flatten()
        
        # 5b. Local (Crop) Similarity via YOLOv8
        sim_local_aggregated = np.zeros_like(sim_global)
        
        # Disable NMS logically by taking all boxes without filtering
        results = yolo_model.predict(img_path, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        valid_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Add small padding, clamp to image bounds
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
            crop_embs = get_embeddings(gr_model, gr_processor, valid_crops, device)
            
            # For each crop, calculate similarity to catalog
            for crop_emb in crop_embs:
                c_norm = np.linalg.norm(crop_emb)
                if c_norm == 0: c_norm = 1e-10
                normalized_crop = crop_emb / c_norm
                
                crop_sims = np.dot(normalized_catalog, normalized_crop).flatten()
                
                # Aggregate crop similarities (e.g., take the max similarity for each product across all crops)
                sim_local_aggregated = np.maximum(sim_local_aggregated, crop_sims)
        else:
            # If no crops found, fallback entirely to global
            sim_local_aggregated = sim_global
            
        # 5c. Combine Scores
        final_scores = (w_global * sim_global) + (w_local * sim_local_aggregated)
        
        top_indices = np.argsort(final_scores)[::-1][:15]
        top_preds = [valid_catalog_ids[i] for i in top_indices]
        
        # Ground truth validation
        true_prods = gt_df[gt_df['bundle_asset_id'] == bid]['product_asset_id'].tolist()
        
        if any(p in top_preds[:1] for p in true_prods):
            top1_correct += 1
        if any(p in top_preds[:5] for p in true_prods):
            top5_correct += 1
        if any(p in top_preds[:15] for p in true_prods):
            top15_correct += 1
            
        valid_eval_bundles += 1
        
    print(f"\nHybrid Evaluation Results ({valid_eval_bundles} valid bundles):")
    print(f"Top-1 Accuracy:  {top1_correct / valid_eval_bundles * 100:.2f}%")
    print(f"Top-5 Accuracy:  {top5_correct / valid_eval_bundles * 100:.2f}%")
    print(f"Top-15 Accuracy: {top15_correct / valid_eval_bundles * 100:.2f}%")

if __name__ == "__main__":
    main()
