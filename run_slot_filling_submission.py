import os
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
            
        # Get raw YOLO boxes
        results = yolo_model.predict(img_path, verbose=False)
        boxes_data = []
        for box_res in results[0].boxes:
            box = box_res.xyxy[0].cpu().numpy()
            conf = float(box_res.conf[0].cpu().numpy())
            cls_id = int(box_res.cls[0].cpu().numpy())
            boxes_data.append({'box': box, 'conf': conf, 'cls_id': cls_id})
            
        # Sort by confidence descending
        boxes_data.sort(key=lambda x: x['conf'], reverse=True)
        
        # NMS function
        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            return iou

        filtered_boxes = []
        for b in boxes_data:
            box = b['box']
            conf = b['conf']
            
            # Area/Size filter (80x80)
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            if w < 80 or h < 80:
                continue
                
            # Confidence filter
            if conf < 0.3:
                continue
                
            # NMS over 0.6 IoU
            keep = True
            for keep_b in filtered_boxes:
                if compute_iou(box, keep_b['box']) > 0.6:
                    keep = False
                    break
            
            if keep:
                filtered_boxes.append(b)
        
        valid_crops = []
        valid_crops_zones = []
        
        grouped_boxes = {"shoe": []}
        other_boxes = []
        
        for b in filtered_boxes:
            box = b['box']
            cls_name = yolo_model.names[int(b['cls_id'])] if 'cls_id' in b else ""
            
            if "shoe" in cls_name.lower() or "boot" in cls_name.lower() or "sneaker" in cls_name.lower() or "footwear" in cls_name.lower():
                grouped_boxes["shoe"].append(box)
            else:
                other_boxes.append(box)
                
        def create_composite_crop(boxes_list, orig_img):
            if not boxes_list: return None
            indiv_crops = []
            pad = 15
            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(orig_img.width, x2 + pad); y2 = min(orig_img.height, y2 + pad)
                indiv_crops.append(orig_img.crop((x1, y1, x2, y2)))
                
            total_width = sum(c.width for c in indiv_crops)
            max_height = max(c.height for c in indiv_crops)
            composite = Image.new('RGB', (total_width, max_height), color='white')
            
            current_x = 0
            for c in indiv_crops:
                composite.paste(c, (current_x, 0))
                current_x += c.width
            return composite

        other_boxes_zones = sf.assign_zones_to_micro_crops(other_boxes, macro_boxes)

        for box, zone in zip(other_boxes, other_boxes_zones):
            x1, y1, x2, y2 = map(int, box)
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            valid_crops.append(img.crop((x1, y1, x2, y2)))
            valid_crops_zones.append(zone)
            
        for group_name, boxes_list in grouped_boxes.items():
            if boxes_list:
                composite = create_composite_crop(boxes_list, img)
                if composite:
                    valid_crops.append(composite)
                    valid_crops_zones.append("FEET")
        
        # Determine prediction sources
        prediction_sources = valid_crops.copy()
        prediction_zones = valid_crops_zones.copy()
        
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
            norm = np.linalg.norm(emb)
            if norm == 0: norm = 1e-10
            normalized_emb = emb / norm
            
            sims = np.dot(normalized_catalog, normalized_emb).flatten()
            
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
    df_sub_15.to_csv("submission_slot_filling_top15.csv", index=False)
    
    print(f"\nSaved Slot-Filling Top-15 submission! Total rows: {len(df_sub_15)}")

if __name__ == "__main__":
    main()
