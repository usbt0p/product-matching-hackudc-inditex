import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO
import warnings
import json
from semantic_filtering import SemanticFilter

from train_mapper import ResidualDomainMapper

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Temporal Proximity Weighting
# ---------------------------------------------------------------------------

def extract_ts(url):
    """Extrae el timestamp en milisegundos de una URL de Inditex CDN."""
    if not isinstance(url, str):
        return 0
    match = re.search(r'ts=(\d+)', url)
    return int(match.group(1)) if match else 0


def build_timestamp_arrays(catalog_ids, df_products):
    """Crea un array de timestamps alineado con valid_catalog_ids."""
    df_products = df_products.copy()
    df_products['ts'] = df_products['product_image_url'].apply(extract_ts)
    ts_map = df_products.set_index('product_asset_id')['ts'].to_dict()
    return np.array([ts_map.get(pid, 0) for pid in catalog_ids], dtype=np.float64)


def apply_temporal_weighting(sims, bundle_ts, catalog_ts, sigma=2.592e9):
    """Multiplica las similitudes por un decaimiento gaussiano basado en la
    diferencia temporal entre el bundle y cada producto del catálogo.
    sigma ~= 1 mes en milisegundos (2 592 000 000 ms).
    """
    diffs = np.abs(catalog_ts - bundle_ts)
    base_weight = 0.5
    temporal_bonus = np.exp(-(diffs ** 2) / (2 * sigma ** 2))
    weights = base_weight + (1.0 - base_weight) * temporal_bonus
    return sims * weights


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
    
    # Load Domain Mapper (if available)
    domain_mapper = None
    mapper_path = "domain_mapper.pt"
    if os.path.exists(mapper_path):
        print("\nLoading Domain Mapper...")
        domain_mapper = ResidualDomainMapper(dim=1024).to(device)
        domain_mapper.proj.load_state_dict(torch.load(mapper_path, map_location=device, weights_only=True))
        domain_mapper.eval()
        print("Domain Mapper loaded successfully.")
    
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

    # Build temporal arrays ------------------------------------------------
    print("Building temporal timestamp arrays...")
    catalog_timestamps = build_timestamp_arrays(valid_catalog_ids, df_products)

    df_bundles = pd.read_csv('data_csvs/bundles_dataset.csv')
    df_bundles['ts'] = df_bundles['bundle_image_url'].apply(extract_ts)
    bundle_ts_map = df_bundles.set_index('bundle_asset_id')['ts'].to_dict()
    n_bundles_with_ts = (df_bundles['ts'] > 0).sum()
    n_catalog_with_ts = (catalog_timestamps > 0).sum()
    print(f"  Bundles con ts: {n_bundles_with_ts}/{len(df_bundles)}  |  "
          f"Productos con ts: {n_catalog_with_ts}/{len(catalog_timestamps)}")

    print("\nLoading Semantic Metadata...")
    sf = SemanticFilter()
    sf.precompute_metadata()
    
    macro_cache = {}
    if os.path.exists("test_dino_macro.json"):
        with open("test_dino_macro.json", "r") as f:
            macro_cache = json.load(f)

    # 2. Process Test Bundles
    print("\nExtracting Embeddings (Raw YOLO Crops + Global if < 3) and Round-Robin Predicting...")
    
    submission_rows_top15 = []
    
    # Configurable parameter: Add global if total images < N
    max_global_if_less_than = 3
    
    for i in tqdm(range(len(test_bundle_ids)), desc="Test Bundles"):
        bid = test_bundle_ids[i]
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
        bundle_section = sf.get_bundle_section(bid)
        macro_boxes = macro_cache.get(bid, [])
            
        # Get raw YOLO boxes (No NMS, No Area Filtering)
        results = yolo_model.predict(img_path, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        valid_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Add small padding, clamp to bounds
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                valid_crops.append(img.crop((x1, y1, x2, y2)))
        
        # Determine prediction sources
        prediction_sources = valid_crops.copy()
        
        # Determine prediction zones via semantic bounding-box intersections
        prediction_zones = sf.assign_zones_to_micro_crops(boxes, macro_boxes)
        
        # Clean up any missing valid crops filtered out dynamically
        # Since valid_crops filters by w>10 and h>10, we must construct prediction_zones perfectly paired.
        valid_zones = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                valid_zones.append(prediction_zones[i])
        
        prediction_zones = valid_zones.copy()
        
        # Add Global if fewer than max_global_if_less_than crops
        if len(valid_crops) < max_global_if_less_than:
            prediction_sources.append(img)
            prediction_zones.append("UNKNOWN")
            
        # Fallback if 0 sources (e.g. YOLO breaks)
        if not prediction_sources:
            prediction_sources = [img]
            prediction_zones = ["UNKNOWN"]
            
        # MANDATORY GLOBAL INCLUSION
        prediction_sources.append(img)
        prediction_zones.append("UNKNOWN")
            
        # Extract embeddings for all sources
        source_embs = []
        batch_size = 16
        for j in range(0, len(prediction_sources), batch_size):
            batch = prediction_sources[j:j+batch_size]
            embs = get_embeddings(gr_model, gr_processor, batch, device)
            source_embs.extend(embs)
            
        # Compute similarities and rank independently for each source
        sorted_lists = []
        for i, emb in enumerate(source_embs):
            # Apply Domain Mapper if available
            if domain_mapper is not None:
                with torch.no_grad():
                    # domain_mapper expects a tensor and returns a normalized tensor
                    t_emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    mapped_t_emb = domain_mapper(t_emb)
                    normalized_emb = mapped_t_emb.squeeze(0).cpu().numpy()
            else:
                norm = np.linalg.norm(emb)
                if norm == 0: norm = 1e-10
                normalized_emb = emb / norm
            
            sims = np.dot(normalized_catalog, normalized_emb).flatten()
            
            # Temporal Proximity Weighting
            current_bundle_ts = bundle_ts_map.get(bid, 0)
            if current_bundle_ts > 0:
                sims = apply_temporal_weighting(sims, current_bundle_ts, catalog_timestamps, sigma=2.592e9)
            
            source_zone = prediction_zones[i]
            sims = sf.apply_similarity_filters(sims, valid_catalog_ids, source_zone, bundle_section)
            
            top_indices = np.argsort(sims)[::-1][:60] # Just top 60 to save memory
            ranked_products = [valid_catalog_ids[idx] for idx in top_indices]
            sorted_lists.append(ranked_products)
            
        # Slot-Filling (Round-Robin) Interleaving
        top_preds = []
        seen = set()
        M = len(sorted_lists)
        indices = [0] * M
        
        while len(top_preds) < 15:
            added_in_round = False
            for src_idx in range(M):
                if len(top_preds) >= 15:
                    break
                    
                # Find the next unseen prediction for source i
                while indices[src_idx] < len(sorted_lists[src_idx]):
                    item = sorted_lists[src_idx][indices[src_idx]]
                    indices[src_idx] += 1
                    if item not in seen:
                        seen.add(item)
                        top_preds.append(item)
                        added_in_round = True
                        break
                        
            # Prevent infinite loop if we hit the end of our lists
            if not added_in_round:
                break
                
        # Append to submission
        for pred in top_preds:
            submission_rows_top15.append({"bundle_asset_id": bid, "product_asset_id": pred})

    # Save submissions
    df_sub_15 = pd.DataFrame(submission_rows_top15)
    df_sub_15.to_csv("submission_no_nms_semantic.csv", index=False)
    
    print(f"\nSaved Slot-Filling Semantic (No NMS) submission! Total rows: {len(df_sub_15)}")

if __name__ == "__main__":
    main()
