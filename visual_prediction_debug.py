'''Create an image displaying bundle + detections + found products + ground truth products.

This is useful for debugging and understanding the model's performance, and where it falls short.
Also, a project with no "tangible" results is much less accessible to judges and curious people like you that are reading this.
'''

import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont 
import math
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, 
    AutoModelForObjectDetection, AutoConfig, AutoModel
)
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO as NativeYOLO

from semantic_filtering import SemanticFilter
from run_gr_lite import get_embeddings, load_gr_lite

import warnings
warnings.filterwarnings("ignore")

def create_mosaic(images_or_paths, pids, gt_pids, max_cols=5, img_size=(250, 350)):
    if not images_or_paths:
        return Image.new('RGB', img_size, color='white')
    n = len(images_or_paths)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    mosaic = Image.new('RGB', (cols * img_size[0], rows * img_size[1]), color='white')
    for i, (item, pid) in enumerate(zip(images_or_paths, pids)):
        try:
            if isinstance(item, str):
                if not os.path.exists(item): continue
                img = Image.open(item).convert('RGB')
            else:
                img = item.copy()
            img.thumbnail((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            bg = Image.new('RGB', img_size, color='white')
            offset = ((img_size[0] - img.width) // 2, (img_size[1] - img.height) // 2)
            bg.paste(img, offset)
            
            # Draw green borders for correct matches
            if pid in gt_pids:
                draw = ImageDraw.Draw(bg)
                draw.rectangle([0, 0, img_size[0]-1, img_size[1]-1], outline="lime", width=12)
                
            x = (i % cols) * img_size[0]
            y = (i // cols) * img_size[1]
            mosaic.paste(bg, (x, y))
        except Exception as e:
            pass
    return mosaic

# --- Configuration Parameters ---
CONFIDENCE_THRESHOLD = 0.3
USE_NMS = True
NMS_IOU_THRESHOLD = 0.6
# 0 = No Global, 1 = Only Global if 0 crops, 2 = Always Include Global
GLOBAL_MODE = 2
# --------------------------------

def main(out_dir="visual_debug_yolos_no_nms", detection_model="clothing", random_selection=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(out_dir, exist_ok=False)
    
    df_train = pd.read_csv('data_csvs/bundles_product_match_train.csv')
    train_bids = df_train['bundle_asset_id'].unique().tolist()
    if not random_selection: random.seed(42)
    sample_bids = random.sample(train_bids, 20)
    
    # 1. Load DINO for macro regions to avoid OOM by doing it first
    print("Extracting DINO macro regions for the 20 samples...")
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    macro_cache = {}
    for bid in sample_bids:
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert("RGB")
        inputs = dino_processor(images=img, text="head. upper body. lower body. feet.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.15, text_threshold=0.2, target_sizes=target_sizes
        )[0]
        
        bundle_macros = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            bundle_macros.append({
                "box": box.cpu().tolist(),
                "score": float(score),
                "zone": label
            })
        macro_cache[bid] = bundle_macros
        
    del dino_model
    torch.cuda.empty_cache()
    print("DINO macro extraction complete. Model offloaded.")

    print("\nLoading YOLO model...")
    yolos_processor = None
    yolos_model = None
    yolov8_model = None
    if detection_model == "fashionpedia":
        yolos_processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        yolos_model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia").to(device)
    elif detection_model == "clothing":
        from huggingface_hub import hf_hub_download
        v8_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
        yolov8_model = NativeYOLO(v8_model_path)
    elif detection_model == "slot_filling_router":
        raise NotImplementedError("Slot Filling Router not implemented yet")
    
    print("Loading GR-Lite model (DINOv3 backbone)...")
    gr_model, gr_processor = load_gr_lite(device)

    print("\nLoading precomputed GR-Lite catalog embeddings...")
    catalog_embeddings = np.load("catalog_grlite_embeddings.npy")
    valid_catalog_ids = np.load("valid_grlite_ids.npy", allow_pickle=True).tolist()
    catalog_norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    catalog_norms[catalog_norms == 0] = 1e-10
    normalized_catalog = catalog_embeddings / catalog_norms

    print("\nLoading Semantic Metadata...")
    sf = SemanticFilter()
    sf.precompute_metadata()
    
    # Font
    linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(linux_font_path, 35)
    except:
        font = ImageFont.load_default()

    for bid in tqdm(sample_bids, desc="Visualizing Train Bundles"):
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert("RGB")
        
        bundle_section = sf.get_bundle_section(bid)
        macro_boxes = macro_cache.get(bid, [])
        
        # --- Detection ---
        raw_preds = []

        if detection_model == "clothing" and yolov8_model:
            try:
                v8_res = yolov8_model.predict(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
                for box_t, score_t, cls_t in zip(
                    v8_res[0].boxes.xyxy,
                    v8_res[0].boxes.conf,
                    v8_res[0].boxes.cls,
                ):
                    raw_preds.append({
                        "box": box_t.cpu().numpy(),
                        "score": float(score_t),
                        "cls": yolov8_model.names[int(cls_t)],
                    })
            except Exception as e:
                print(f"YOLOv8 failed for {bid}: {e}")

        elif detection_model == "fashionpedia" and yolos_model and yolos_processor:
            allowed_cats = [
                'shirt', 'top', 't-shirt', 'sweatshirt', 'sweater', 'cardigan',
                'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress',
                'jumpsuit', 'glasses', 'hat', 'hair accessory', 'tie', 'glove',
                'belt', 'sock', 'shoe', 'bag', 'scarf', 'collar', 'lapel', 'buckle'
            ]
            try:
                inputs = yolos_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = yolos_model(**inputs)
                target_sizes = torch.tensor([img.size[::-1]])
                yolos_results = yolos_processor.post_process_object_detection(
                    outputs, threshold=0.01, target_sizes=target_sizes
                )[0]
                for score, label_t, box in zip(
                    yolos_results["scores"], yolos_results["labels"], yolos_results["boxes"]
                ):
                    s = float(score)
                    if s < CONFIDENCE_THRESHOLD:
                        continue
                    label_id = label_t.item() if hasattr(label_t, 'item') else int(label_t)
                    cls_name = yolos_model.config.id2label[label_id]
                    if not any(cat in cls_name.lower() for cat in allowed_cats):
                        continue
                    b = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
                    raw_preds.append({"box": b, "score": s, "cls": cls_name})
            except Exception as e:
                print(f"YOLOS failed for {bid}: {e}")

        raw_preds.sort(key=lambda x: x["score"], reverse=True)
        
        filtered_preds = []
        if USE_NMS:
            from compare_models import compute_iou_router
                
            for pred in raw_preds:
                keep = True
                for kept in filtered_preds:
                    if compute_iou_router(pred["box"], kept["box"]) > NMS_IOU_THRESHOLD:
                        keep = False
                        break
                if keep:
                    filtered_preds.append(pred)
        else:
            filtered_preds = raw_preds
            
        boxes = [p["box"] for p in filtered_preds]
        labels = [(p["cls"], p["score"]) for p in filtered_preds]
            
        valid_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            pad = 15
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad); y2 = min(img.height, y2 + pad)
            w = x2 - x1; h = y2 - y1
            if w > 10 and h > 10:
                valid_crops.append(img.crop((x1, y1, x2, y2)))
                
        prediction_sources = valid_crops.copy()
        
        # Zone assignment
        prediction_zones = sf.assign_zones_to_micro_crops(boxes, macro_boxes)
        valid_zones = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            pad = 15
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad); y2 = min(img.height, y2 + pad)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                valid_zones.append(prediction_zones[i])
        prediction_zones = valid_zones.copy()
        
        # Draw bounding boxes on source image
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w <= 10 or h <= 10: continue
            
            zone = prediction_zones[idx] if idx < len(prediction_zones) else "UNKNOWN"
            cls_name, score = labels[idx]
            
            draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
            text = f"{cls_name} {score:.2f} [{zone}]"
            try:
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+2, text_bbox[3]+2], fill="green")
            except:
                pass
            draw.text((x1+2, y1+2), text, fill="white", font=font)
            
        if GLOBAL_MODE == 2:
            prediction_sources.append(img)
            prediction_zones.append("UNKNOWN")
        elif GLOBAL_MODE == 1 and len(prediction_sources) == 0:
            prediction_sources.append(img)
            prediction_zones.append("UNKNOWN")
        elif len(prediction_sources) == 0:
            # Absolute fallback if 0 crops and global mode 0
            prediction_sources = [img]
            prediction_zones = ["UNKNOWN"]
        
        source_embs = []
        batch_size = 16
        for j in range(0, len(prediction_sources), batch_size):
            batch = prediction_sources[j:j+batch_size]
            embs = get_embeddings(gr_model, gr_processor, batch, device)
            source_embs.extend(embs)
            
        sorted_lists = []
        for i, emb in enumerate(source_embs):
            norm = np.linalg.norm(emb)
            if norm == 0: norm = 1e-10
            normalized_emb = emb / norm
            sims = np.dot(normalized_catalog, normalized_emb).flatten()
            source_zone = prediction_zones[i]
            sims = sf.apply_similarity_filters(sims, valid_catalog_ids, source_zone, bundle_section)
            top_indices = np.argsort(sims)[::-1][:60]
            ranked_products = [valid_catalog_ids[idx] for idx in top_indices]
            sorted_lists.append(ranked_products)
            
        top_preds = []
        seen = set()
        M = len(sorted_lists)
        indices = [0] * M
        while len(top_preds) < 15:
            added_in_round = False
            for src_idx in range(M):
                if len(top_preds) >= 15: break
                while indices[src_idx] < len(sorted_lists[src_idx]):
                    item = sorted_lists[src_idx][indices[src_idx]]
                    indices[src_idx] += 1
                    if item not in seen:
                        seen.add(item)
                        top_preds.append(item)
                        added_in_round = True
                        break
            if not added_in_round:
                break
                
        # Get actual ground truth products for this bundle
        gt_products = df_train[df_train['bundle_asset_id'] == bid]['product_asset_id'].tolist()
        
        product_images = []
        for pid in gt_products:
            p_path = os.path.join("data", "products", f"{pid}.jpg")
            if os.path.exists(p_path):
                product_images.append(p_path)
            
        gt_mosaic = create_mosaic(product_images, gt_products, gt_products, max_cols=5)
        
        pred_images = []
        for pid in top_preds:
            p_path = os.path.join("data", "products", f"{pid}.jpg")
            if os.path.exists(p_path):
                pred_images.append(p_path)
            else:
                pred_images.append(Image.new('RGB', (250, 350), color='gray'))
        pred_mosaic = create_mosaic(pred_images, top_preds, gt_products, max_cols=5)
        
        # Combine draw_img, gt_mosaic, and pred_mosaic
        final_height = max(draw_img.height, gt_mosaic.height + pred_mosaic.height + 50)
        
        final_img = Image.new('RGB', (draw_img.width + max(gt_mosaic.width, pred_mosaic.width) + 20, final_height), 'white')
        final_img.paste(draw_img, (0, 0))
        
        draw_final = ImageDraw.Draw(final_img)
        
        x_offset = draw_img.width + 10
        draw_final.text((x_offset, 10), f"Ground Truth ({len(gt_products)} items)", fill="black", font=font)
        final_img.paste(gt_mosaic, (x_offset, 60))
        
        y_offset = 60 + gt_mosaic.height + 20
        draw_final.text((x_offset, y_offset), f"YOLOS No-NMS + Semantic Preds (Top 15)", fill="black", font=font)
        final_img.paste(pred_mosaic, (x_offset, y_offset + 50))
        
        final_img.save(os.path.join(out_dir, f"{bid}_debug.jpg"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visual_prediction_debug.py <out_dir> <detection_model>")
        sys.exit(1)
    out_dir = sys.argv[1]
    detection_model = sys.argv[2] if len(sys.argv) > 2 else "clothing"
    random_selection = sys.argv[3] if len(sys.argv) > 3 else False
    main(out_dir, detection_model, random_selection)
