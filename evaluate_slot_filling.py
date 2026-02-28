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
        
    print("\nLoading datasets for Slot-Filling Train subset evaluation...")
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
    
    # 5. Slot-Filling Evaluation Loop
    print("\nExtracting Bundle Crops & Evaluating with Slot-Filling (Round-Robin)...")
    top1_correct = 0
    top5_correct = 0
    top15_correct = 0
    valid_eval_bundles = 0
    
    # Configurable parameter from user: Add global if total images < N
    max_global_if_less_than = 3
    
    for idx, bid in enumerate(tqdm(sample_bundles, desc="Round-Robin Bundles")):
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
            
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

        # Group Semantic Categories (Shoes & Gloves)
        valid_crops = []
        grouped_boxes = {"shoe": [], "glove": []}
        other_boxes = []
        
        for b in filtered_boxes:
            box = b['box']
            conf = b['conf']
            cls_name = yolo_model.names[int(b['cls_id'])] if 'cls_id' in b else ""
            
            # YOLOv8-clothing labels might be 'Shoe' or 'shoe'
            if "shoe" in cls_name.lower() or "boot" in cls_name.lower() or "sneaker" in cls_name.lower() or "footwear" in cls_name.lower():
                grouped_boxes["shoe"].append(box)
            elif "glove" in cls_name.lower():
                grouped_boxes["glove"].append(box)
            else:
                other_boxes.append(box)
                
        def create_composite_crop(boxes_list, orig_img):
            if not boxes_list: return None
            # Get individual crops
            indiv_crops = []
            pad = 15
            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(orig_img.width, x2 + pad); y2 = min(orig_img.height, y2 + pad)
                indiv_crops.append(orig_img.crop((x1, y1, x2, y2)))
                
            # Paste side by side
            total_width = sum(c.width for c in indiv_crops)
            max_height = max(c.height for c in indiv_crops)
            composite = Image.new('RGB', (total_width, max_height), color='white')
            
            current_x = 0
            for c in indiv_crops:
                composite.paste(c, (current_x, 0))
                current_x += c.width
            return composite

        # Process standard items
        for box in other_boxes:
            x1, y1, x2, y2 = map(int, box)
            pad = 15
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad); y2 = min(img.height, y2 + pad)
            valid_crops.append(img.crop((x1, y1, x2, y2)))
            
        # Process grouped items
        for group_name, boxes_list in grouped_boxes.items():
            if boxes_list:
                composite = create_composite_crop(boxes_list, img)
                if composite:
                    valid_crops.append(composite)
        
        # Determine prediction sources
        prediction_sources = valid_crops.copy()
        
        # MANDATORY GLOBAL INCLUSION (Level 3 Architecture)
        # We ALWAYS include the global uncropped image as a slot-filling source.
        prediction_sources.append(img)
            
        # Fallback if 0 sources (e.g. YOLO predicts nothing)
        if not prediction_sources:
            prediction_sources = [img]
            
        # Extract embeddings for all sources
        source_embs = []
        batch_size = 16
        for j in range(0, len(prediction_sources), batch_size):
            batch = prediction_sources[j:j+batch_size]
            embs = get_embeddings(gr_model, gr_processor, batch, device)
            source_embs.extend(embs)
            
        # Compute similarities and rank independently for each source
        sorted_lists = []
        for emb in source_embs:
            norm = np.linalg.norm(emb)
            if norm == 0: norm = 1e-10
            normalized_emb = emb / norm
            
            sims = np.dot(normalized_catalog, normalized_emb).flatten()
            top_indices = np.argsort(sims)[::-1]
            ranked_products = [valid_catalog_ids[i] for i in top_indices]
            sorted_lists.append(ranked_products)
            
        # Slot-Filling (Round-Robin)
        top_preds = []
        seen = set()
        M = len(sorted_lists)
        indices = [0] * M
        
        while len(top_preds) < 15:
            added_in_round = False
            for i in range(M):
                if len(top_preds) >= 15:
                    break
                    
                # Find the next unseen prediction for source i
                while indices[i] < len(sorted_lists[i]):
                    item = sorted_lists[i][indices[i]]
                    indices[i] += 1
                    if item not in seen:
                        seen.add(item)
                        top_preds.append(item)
                        added_in_round = True
                        break
                        
            # If we somehow exhausted the catalog without reaching 15, avoid infinite loop
            if not added_in_round:
                break
                
        # Ground truth validation
        true_prods = gt_df[gt_df['bundle_asset_id'] == bid]['product_asset_id'].tolist()
        
        if any(p in top_preds[:1] for p in true_prods):
            top1_correct += 1
        if any(p in top_preds[:5] for p in true_prods):
            top5_correct += 1
        if any(p in top_preds[:15] for p in true_prods):
            top15_correct += 1
            
        valid_eval_bundles += 1
        
    print(f"\nSlot-Filling Evaluation Results ({valid_eval_bundles} valid bundles):")
    print(f"Top-1 Accuracy:  {top1_correct / valid_eval_bundles * 100:.2f}%")
    print(f"Top-5 Accuracy:  {top5_correct / valid_eval_bundles * 100:.2f}%")
    print(f"Top-15 Accuracy: {top15_correct / valid_eval_bundles * 100:.2f}%")

if __name__ == "__main__":
    main()
