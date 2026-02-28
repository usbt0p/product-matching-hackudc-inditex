import os
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import math
import warnings

import create_dual_side_by_side

# Definir la ruta y el tamaño (ejemplo: 35)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_size = 35

try:
    font = ImageFont.truetype(font_path, font_size)
except OSError:
    # Si por alguna razón no la encuentra, intenta con una genérica de Linux
    print("Fuente específica no encontrada, usando fuente por defecto.")
    font = ImageFont.load_default()

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

def create_mosaic(images_or_paths, max_cols=3, img_size=(250, 350)):
    """Creates a grid mosaic from a list of image paths or PIL Images."""
    if not images_or_paths:
        return Image.new('RGB', img_size, color='white')
        
    n = len(images_or_paths)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    
    mosaic = Image.new('RGB', (cols * img_size[0], rows * img_size[1]), color='white')
    
    for i, item in enumerate(images_or_paths):
        try:
            if isinstance(item, str):
                if not os.path.exists(item):
                    continue
                img = Image.open(item).convert('RGB')
            else:
                img = item.copy()
            # Resize and crop to maintain aspect ratio
            img.thumbnail((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            
            # Create a white background of exact size and center thumbnail
            bg = Image.new('RGB', img_size, color='white')
            offset = ((img_size[0] - img.width) // 2, (img_size[1] - img.height) // 2)
            bg.paste(img, offset)
            
            x = (i % cols) * img_size[0]
            y = (i // cols) * img_size[1]
            mosaic.paste(bg, (x, y))
        except Exception as e:
            print(f"Skipping mosaic item: {e}")
            
    return mosaic

def main():
    print("Loading datasets...")
    df_train = pd.read_csv('data_csvs/bundles_product_match_train.csv')
    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\nLoading YOLOv8-Clothing model...")
    try:
        v8_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
        yolov8_model = YOLO(v8_model_path)
    except Exception as e:
        print(f"Failed to load YOLOv8-Clothing: {e}")
        return
        
    print("\nLoading GR-Lite model...")
    gr_model, gr_processor = load_gr_lite(device)
    if not gr_model:
        return
        
    print("\nLoading precomputed GR-Lite catalog embeddings...")
    catalog_emb_path = "catalog_grlite_embeddings.npy"
    catalog_ids_path = "valid_grlite_ids.npy"
    
    if os.path.exists(catalog_emb_path) and os.path.exists(catalog_ids_path):
        catalog_embeddings = np.load(catalog_emb_path)
        valid_catalog_ids = np.load(catalog_ids_path, allow_pickle=True).tolist()
    else:
        print("GR-Lite catalog embeddings missing. Please run run_gr_lite.py first.")
        return

    # Normalize catalog embeddings
    catalog_norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    catalog_norms[catalog_norms == 0] = 1e-10
    normalized_catalog = catalog_embeddings / catalog_norms
    
    # Font
    try:
        # Usamos la ruta completa de Linux que definiste arriba
        linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        font = ImageFont.truetype(linux_font_path, 35)       # Tamaño para las cajas de YOLO
        title_font = ImageFont.truetype(linux_font_path, 55) # Tamaño para los títulos
    except IOError as e:
        print(f"Advertencia: No se pudo cargar la fuente. Usando por defecto. Error: {e}")
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        
    out_dir = "visual_debug_slot_filling"
    os.makedirs(out_dir, exist_ok=True)
    
    random.seed(123)
    train_bundles = df_train['bundle_asset_id'].unique().tolist()
    sample_bundles = random.sample(train_bundles, 25)
    
    w_global = 0.4
    w_local = 0.6
    
    for bid in sample_bundles:
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path):
            continue
            
        print(f"Inferring and Visualizing {bid}...")
        img_orig = Image.open(img_path).convert("RGB")
        
        # 1. Overlay YOLO detection 
        v8_img = img_orig.copy()
        draw = ImageDraw.Draw(v8_img)
        
        results = yolov8_model.predict(img_path, conf=0.1, iou=0.45, agnostic_nms=False, verbose=False)
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
        for b in filtered_boxes:
            box = b['box']
            conf = b['conf']
            cls_id = b['cls_id']
            cls_name = yolov8_model.names[cls_id]
            
            xmin, ymin, xmax, ymax = map(int, box)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=10)
            
            text = f"{cls_name} {conf:.2f}"
            
            # Check bbox
            try:
                text_bbox = draw.textbbox((xmin, ymin), text, font=font)
                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+10, text_bbox[3]+10], fill="lime")
            except:
                pass
            draw.text((xmin+3, ymin+3), text, fill="black", font=font)
            
            # Crop logic
            pad = 15
            x1 = max(0, xmin - pad)
            y1 = max(0, ymin - pad)
            x2 = min(img_orig.width, xmax + pad)
            y2 = min(img_orig.height, ymax + pad)
            valid_crops.append(img_orig.crop((x1, y1, x2, y2)))
                
        # 2. Extract Hybrid Embeddings (Round-Robin Slot-Filling)
        # Determine prediction sources
        prediction_sources = valid_crops.copy()
        
        # MANDATORY GLOBAL INCLUSION 
        prediction_sources.append(img_orig)
            
        # Fallback if 0 sources
        if not prediction_sources:
            prediction_sources = [img_orig]
            
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
            top_indices = np.argsort(sims)[::-1][:60]
            ranked_products = [valid_catalog_ids[idx] for idx in top_indices]
            sorted_lists.append(ranked_products)
            
        # Slot-Filling (Round-Robin)
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
                        
            # Prevent infinite loop
            if not added_in_round:
                break
        
        # 3. Get Predicted Top-15 Products Mosaic
        pred_paths = [os.path.join("data", "products", f"{p}.jpg") for p in top_preds]
        
        # Determine which preds are correct (Ground Truth hits)
        gt_prods = df_train[df_train['bundle_asset_id'] == bid]['product_asset_id'].tolist()
        
        # Let's open the images and add borders locally to avoid modifying create_mosaic for now
        bordered_preds = []
        for p_id in top_preds:
            p_img_path = os.path.join("data", "products", f"{p_id}.jpg")
            if os.path.exists(p_img_path):
                img = Image.open(p_img_path).convert("RGB")
                if p_id in gt_prods:
                    # Add green border
                    bordered = Image.new('RGB', (img.width + 80, img.height + 80), color="lime")
                    bordered.paste(img, (40, 40))
                    bordered_preds.append(bordered)
                else:
                    bordered_preds.append(img)
            else:
                bordered_preds.append(Image.new('RGB', (250, 350), color="white"))
                
        pred_mosaic = create_mosaic(bordered_preds, max_cols=3, img_size=(250, 350))
        
        # 4. Get Ground Truth Products
        gt_prods = df_train[df_train['bundle_asset_id'] == bid]['product_asset_id'].tolist()
        gt_paths = [os.path.join("data", "products", f"{p}.jpg") for p in gt_prods]
        gt_mosaic = create_mosaic(gt_paths, max_cols=3, img_size=(250, 350))
        
        # 5. Assemble
        # v8_img | Predicted Top-15 Mosaic | GT Mosaic
        w1, h1 = v8_img.size
        w2, h2 = pred_mosaic.size
        w3, h3 = gt_mosaic.size
        
        # Scale mosaics to match height of input image
        scale_pred = h1 / h2 if h2 > 0 else 1
        new_w2 = int(w2 * scale_pred)
        pred_mosaic = pred_mosaic.resize((new_w2, h1))
        
        scale_gt = h1 / h3 if h3 > 0 else 1
        new_w3 = int(w3 * scale_gt)
        gt_mosaic = gt_mosaic.resize((new_w3, h1))
        
        final_img = Image.new('RGB', (w1 + new_w2 + new_w3 + 60, h1 + 80), color="#1e1e1e")
        draw_final = ImageDraw.Draw(final_img)
        
        # Headers
        draw_final.text((20, 20), "YOLOv8 Source Detections", fill="white", font=title_font)
        draw_final.text((w1 + 40, 20), f"Top-15 Hybrid Predictions", fill="white", font=title_font)
        draw_final.text((w1 + new_w2 + 60, 20), f"Ground Truth ({len(gt_prods)} items)", fill="white", font=title_font)
        
        # Paste
        final_img.paste(v8_img, (10, 80))
        final_img.paste(pred_mosaic, (w1 + 30, 80))
        final_img.paste(gt_mosaic, (w1 + new_w2 + 50, 80))
        
        final_img.save(os.path.join(out_dir, f"{bid}_debug.jpg"))

    print(f"\nSaved debugging visualizations to {out_dir}/")

if __name__ == "__main__":
    main()
