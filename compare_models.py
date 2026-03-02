'''Compare the performance of different segmenters on the same images, and plot them side by side.

The segmenters are:
- Grounding DINO
- YOLO-Clothing
- YOLOS-Fashionpedia
- Slot Filling Router (an ensemble of the above that uses intermediate logic to combine the results on four sections: UPPER, LOWER, SHOES, DEFAULT)
'''

import os
import random
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModelForObjectDetection
from ultralytics import YOLO as NativeYOLO
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings("ignore")

def create_triple_side_by_side(img1, img2, img3, text1="Grounding DINO", text2="YOLO-Clothing", text3="YOLOS-Fashionpedia"):
    # (Existing function code kept exactly the same...)
    w1, h1 = img1.size
    w2, h2 = img2.size
    w3, h3 = img3.size
    
    def scale_height(w, h, target_h):
        if h != target_h:
            return int((w / h) * target_h)
        return w
        
    w2 = scale_height(w2, h2, h1)
    img2 = img2.resize((w2, h1))
    
    w3 = scale_height(w3, h3, h1)
    img3 = img3.resize((w3, h1))
        
    dst = Image.new('RGB', (w1 + w2 + w3, h1))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (w1, 0))
    dst.paste(img3, (w1 + w2, 0))
    
    draw = ImageDraw.Draw(dst)
    
    try:
        linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(linux_font_path, 45)
    except IOError as e:
        print(f"Advertencia: No se pudo cargar la fuente. Usando por defecto. Error: {e}")
        font = ImageFont.load_default()
    
    draw.rectangle([0, 0, 350, 45], fill="black")
    draw.text((10, 10), text1, fill="white", font=font)
    
    draw.rectangle([w1, 0, w1 + 350, 45], fill="black")
    draw.text((w1 + 10, 10), text2, fill="white", font=font)
    
    draw.rectangle([w1 + w2, 0, w1 + w2 + 350, 45], fill="black")
    draw.text((w1 + w2 + 10, 10), text3, fill="white", font=font)
    
    return dst

def compute_iou_router(boxA, boxB):
    """Calcula la Intersección sobre Unión (IoU)."""
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    interArea = max(0.0, float(xB - xA)) * max(0.0, float(yB - yA))
    boxAreaA = (float(boxA[2]) - float(boxA[0])) * (float(boxA[3]) - float(boxA[1]))
    boxAreaB = (float(boxB[2]) - float(boxB[0])) * (float(boxB[3]) - float(boxB[1]))
    return interArea / float(boxAreaA + boxAreaB - interArea) if (boxAreaA + boxAreaB - interArea) > 0 else 0.0

def filter_redundant_boxes(boxes_list, iou_thresh=0.75):
    """Aplica NMS para limpiar la sobredetección."""
    keep = []
    # Ordenar por confianza si existe, si no por área (cajas más pequeñas suelen ser más precisas en ropa)
    if all('score' in b for b in boxes_list):
        boxes_list = sorted(boxes_list, key=lambda x: x['score'], reverse=True)
    else:
        # Calculate area using coordinates (x_min, y_min, x_max, y_max)
        boxes_list = sorted(boxes_list, key=lambda x: (float(x['box'][2])-float(x['box'][0]))*(float(x['box'][3])-float(x['box'][1])) )
    
    for item in boxes_list:
        if not any(compute_iou_router(item['box'], k['box']) > iou_thresh for k in keep):
            keep.append(item)
    return keep

def get_union_box(boxes):
    if not boxes: return None
    x1 = min(float(b[0]) for b in boxes)
    y1 = min(float(b[1]) for b in boxes)
    x2 = max(float(b[2]) for b in boxes)
    y2 = max(float(b[3]) for b in boxes)
    return [x1, y1, x2, y2]

def slot_filling_router(dino_preds, fash_preds, cloth_preds, img_width, img_height):
    """
    Rellena los slots del MoE (UPPER, LOWER, SHOES, DEFAULT) 
    usando el mejor modelo para cada caso y fallbacks.
    """
    slots = {'UPPER': [], 'LOWER': [], 'SHOES': [], 'DEFAULT': []}

    # --- 1. SLOT: SHOES (Unir en una sola bounding box si hay más de uno) ---
    shoes = [p for p in cloth_preds if 'shoe' in p['label'].lower()]
    if not shoes:
        shoes = [p for p in fash_preds if 'shoe' in p['label'].lower()]
        
    if shoes:
        union_shoe = get_union_box([s['box'] for s in shoes])
        slots['SHOES'] = [{'box': union_shoe, 'route': 'SHOES', 'source': shoes[0].get('source', 'Unknown') + " (Merged)"}]

    # --- 2. SLOT: ACCESORY / HEAD ---
    # Consentimiento para HEAD (DINO + otro modelo)
    dino_head = [p for p in dino_preds if p['label'].lower() in ['head', 'hat']]
    cloth_head = [p for p in cloth_preds if p['label'].lower() in ['accessories', 'bags', 'bag', 'head', 'hat']]
    fash_head = [p for p in fash_preds if p['label'].lower() in ['hat', 'glasses', 'sunglasses']]
    
    # Consenso espacial para head/accessories
    accs_to_keep = []
    
    # Si DINO detecta cabeza, requerimos que otro yolo también detecte algo similar ahí
    if dino_head:
        all_others = cloth_head + fash_head
        for dh in dino_head:
            has_consensus = any(compute_iou_router(dh['box'], o['box']) > 0.3 for o in all_others)
            if has_consensus:
                accs_to_keep.append({'box': dh['box'], 'route': 'DEFAULT', 'source': 'Consensus Head'})

    # Accesorios generales (bolsos, gafas) - Si YOLOS o YOLOv8 lo detectan con alto IOU, lo mantenemos (o si conf es altisima)
    other_accs = [p for p in fash_preds if p['label'].lower() in ['bag', 'wallet', 'tie', 'scarf', 'belt']]
    other_accs += [p for p in cloth_preds if p['label'].lower() in ['accessories', 'bags', 'bag']]
    
    # Conservamos los más confidentes o los que tienen consenso
    slots['DEFAULT'].extend(accs_to_keep)
    for a in filter_redundant_boxes(other_accs, 0.5):
        slots['DEFAULT'].append({'box': a['box'], 'route': 'DEFAULT', 'source': a.get('source', 'Unknown')})

    # --- 3 & 4. SLOTS: UPPER & LOWER (Prioridad DINO -> YOLO-Clothing posicional) ---
    
    # Extraemos upper/lower de DINO
    dino_uppers = [p for p in dino_preds if 'upper' in p['label'].lower() or 'top' in p['label'].lower()]
    dino_lowers = [p for p in dino_preds if 'lower' in p['label'].lower() or 'bottom' in p['label'].lower()]
    
    # Ropa general de YOLO-Clothing
    cloth_general = [p for p in cloth_preds if 'clothing' in p['label'].lower()]

    # Unimos y aplicamos NMS a todos juntos
    combined_ul = dino_uppers + dino_lowers + cloth_general
    filtered_ul = filter_redundant_boxes(combined_ul, 0.75)

    # En lugar de asegurar 1 único UPPER y 1 único LOWER, volcamos todos los que han sobrevivido al NMS global.
    for p in filtered_ul:
        if p.get('source') == 'Grounding DINO':
            if 'upper' in p['label'].lower() or 'top' in p['label'].lower():
                slots['UPPER'].append({'box': p['box'], 'route': 'UPPER', 'source': p.get('source', 'Unknown')})
            else:
                slots['LOWER'].append({'box': p['box'], 'route': 'LOWER', 'source': p.get('source', 'Unknown')})
        else:
            # Es de YOLO-Clothing (general 'clothing'). Lo asignamos a UPPER o LOWER según su centro vertical.
            y_center = (float(p['box'][1]) + float(p['box'][3])) / 2.0
            if y_center < float(float(img_height) * 0.5):
                slots['UPPER'].append({'box': p['box'], 'route': 'UPPER (Clothing)', 'source': p.get('source', 'Unknown')})
            else:
                slots['LOWER'].append({'box': p['box'], 'route': 'LOWER (Clothing)', 'source': p.get('source', 'Unknown')})

    final_rois = slots['UPPER'] + slots['LOWER'] + slots['SHOES'] + slots['DEFAULT']
    
    return final_rois

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading descriptions...")
    # This file has body like segments we want to use to filter by semantic body part
    with open("unique_product_descriptions.txt", "r") as f:
        descriptions = [line.strip().replace('/', ' ').lower() for line in f if line.strip()]
    
    # DINO Prompt
    safe_dino_desc = descriptions[:30]
    dino_prompt = ". ".join(safe_dino_desc) + "."
    
    print("Loading datasets...")
    df_bundles = pd.read_csv('data_csvs/bundles_dataset.csv')
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    test_bundles = set(df_test['bundle_asset_id'].unique())
    train_bundles = df_bundles[~df_bundles['bundle_asset_id'].isin(test_bundles)]['bundle_asset_id'].tolist()
    
    # random.seed(42)
    sample_bundles = random.sample(train_bundles, 25)
    print(f"Sampled 25 train bundles: {sample_bundles}")
    
    # Load Models
    print("\nLoading Grounding DINO...")
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

    print("\nLoading YOLOv8-Clothing...")
    try:
        v8_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
        yolov8_model = NativeYOLO(v8_model_path)
    except Exception as e:
        print(f"Failed to load YOLOv8-Clothing: {e}")
        yolov8_model = None
        
    print("\nLoading YOLOS-Fashionpedia...")
    try:
        yolos_processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        yolos_model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia").to(device)
    except Exception as e:
        print(f"Failed to load YOLOS-Fashionpedia: {e}")
        yolos_processor = None
        yolos_model = None
        
    out_dir = "compare_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Font
    try:
        linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        box_font = ImageFont.truetype(linux_font_path, 35)       # Tamaño para las cajas de YOLO
        title_font = ImageFont.truetype(linux_font_path, 55) # Tamaño para los títulos
    except IOError as e:
        print(f"Advertencia: No se pudo cargar la fuente. Usando por defecto. Error: {e}\n\
            Esto puede causar que el texto sea minúsculo o no se vea.")
        box_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    for bid in sample_bundles:
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path):
            continue
            
        print(f"\nProcessing {bid}...")
        img_orig = Image.open(img_path).convert("RGB")
        
        # --- Grounding DINO ---
        dino_img = img_orig.copy()
        try:
            inputs = dino_processor(images=img_orig, text=dino_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = dino_model(**inputs)
                
            target_sizes = torch.tensor([img_orig.size[::-1]])
            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.5,
                target_sizes=target_sizes
            )[0]
            
            dino_preds = []
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                dino_preds.append({
                    "box": box,
                    "score": float(score),
                    "label": label
                })
                
            dino_preds.sort(key=lambda x: x["score"], reverse=True)
                
            filtered_dino = []
            for pred in dino_preds:
                keep = True
                for kept in filtered_dino:
                    if compute_iou_router(pred["box"], kept["box"]) > 0.6:
                        keep = False
                        break
                if keep:
                    filtered_dino.append(pred)
            
            draw = ImageDraw.Draw(dino_img)
            for pred in filtered_dino:
                box = pred["box"]
                score = pred["score"]
                label = pred["label"]
                
                xmin, ymin, xmax, ymax = box
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
                
                # Draw text background
                text = f"{label} {score:.2f}"
                text_bbox = draw.textbbox((xmin, ymin), text, font=box_font)
                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+4, text_bbox[3]+4], fill="red")
                draw.text((xmin+2, ymin+2), text, fill="white", font=box_font)
        except Exception as e:
            print(f"DINO failed for {bid}: {e}")
            
        # --- YOLOv8-Clothing ---
        v8_img = img_orig.copy()
        if yolov8_model:
            try:
                v8_res = yolov8_model.predict(img_path, conf=0.45, iou=0.5, agnostic_nms=True, max_det=1000, verbose=False)
                
                draw = ImageDraw.Draw(v8_img)
                # Parse results directly
                for box_data in v8_res[0].boxes:
                    box = box_data.xyxy[0].cpu().numpy()
                    conf = box_data.conf[0].cpu().numpy()
                    cls_id = int(box_data.cls[0].cpu().numpy())
                    cls_name = yolov8_model.names[cls_id]
                    
                    xmin, ymin, xmax, ymax = box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=4)
                    
                    # Draw text background
                    text = f"{cls_name} {conf:.2f}"
                    text_bbox = draw.textbbox((xmin, ymin), text, font=box_font)
                    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+4, text_bbox[3]+4], fill="blue")
                    draw.text((xmin+2, ymin+2), text, fill="white", font=box_font)
                    
            except Exception as e:
                print(f"YOLOv8 failed for {bid}: {e}")
        
        # --- YOLOS-Fashionpedia ---
        yolos_img = img_orig.copy()
        if yolos_model and yolos_processor:
            try:
                inputs = yolos_processor(images=img_orig, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = yolos_model(**inputs)
                
                target_sizes = torch.tensor([img_orig.size[::-1]])
                results = yolos_processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]
                
                # Extract and sort predictions
                yolos_preds = []
                allowed_cats = [
                    'shirt', 'top', 't-shirt', 'sweatshirt', 'sweater', 'cardigan',
                    'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress',
                    'glasses', 'hat', 'tie', 'glove',
                    'belt', 'sock', 'shoe', 'bag', 'scarf', 'collar',
                    'lapel', 'buckle'
                ]
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    # Safely handle label if it's already a string, int, or a tensor
                    if hasattr(label, 'item'):
                        label_id = label.item()
                    else:
                        label_id = int(label)
                        
                    cls_name = yolos_model.config.id2label[label_id]
                    if not any(cat in cls_name.lower() for cat in allowed_cats):
                        continue
                        
                    yolos_preds.append({
                        "box": box.cpu().tolist() if hasattr(box, 'cpu') else list(box),
                        "score": score.item() if hasattr(score, 'item') else float(score),
                        "label_id": label_id
                    })
                
                yolos_preds.sort(key=lambda x: x["score"], reverse=True)
                
                filtered_yolos = []
                for pred in yolos_preds:
                    if pred["score"] < 0.5:
                        continue
                        
                    keep = True
                    for kept in filtered_yolos:
                        if compute_iou_router(pred["box"], kept["box"]) > 0.6:
                            keep = False
                            break
                    if keep:
                        filtered_yolos.append(pred)
                
                draw = ImageDraw.Draw(yolos_img)
                for pred in filtered_yolos:
                    box = pred["box"]
                    score = pred["score"]
                    label_id = pred["label_id"]
                    cls_name = yolos_model.config.id2label[label_id]
                    
                    # Convert box to same format as others
                    yolo_formatted_pred = {
                        "box": box,
                        "score": score,
                        "label": cls_name,
                        "source": "YOLOS-Fashionpedia"
                    }
                    
                    xmin, ymin, xmax, ymax = box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=4)
                    
                    text = f"{cls_name} {score:.2f}"
                    text_bbox = draw.textbbox((xmin, ymin), text, font=box_font)
                    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+4, text_bbox[3]+4], fill="green")
                    draw.text((xmin+2, ymin+2), text, fill="white", font=box_font)
                    
                    pred["source"] = "YOLOS-Fashionpedia"
                    pred["label"] = cls_name
                    
            except Exception as e:
                print(f"YOLOS failed for {bid}: {e}")
        
        # Format predictions for router
        # Convert dino_preds
        dino_for_router = []
        for p in filtered_dino:
            dino_for_router.append({
                "box": p["box"],
                "label": p["label"],
                "score": p["score"],
                "source": "Grounding DINO"
            })
            
        # Convert clothing preds
        cloth_for_router = []
        if yolov8_model and 'v8_res' in locals():
            for box_data in v8_res[0].boxes:
                box = box_data.xyxy[0].cpu().numpy().tolist()
                conf = float(box_data.conf[0].cpu().numpy())
                cls_id = int(box_data.cls[0].cpu().numpy())
                cls_name = yolov8_model.names[cls_id]
                cloth_for_router.append({
                    "box": box,
                    "label": cls_name,
                    "score": conf,
                    "source": "YOLOv8-Clothing"
                })
        
        # yolos_preds is already roughly formatted, but we use filtered_yolos
        fash_for_router = []
        for p in filtered_yolos:
            fash_for_router.append({
                "box": p["box"],
                "label": p["label"],
                "score": p["score"],
                "source": "YOLOS-Fashionpedia"
            })
            
        # Route
        final_rois = slot_filling_router(dino_for_router, fash_for_router, cloth_for_router, img_orig.width, img_orig.height)
        
        # Combine
        combined = create_triple_side_by_side(dino_img, v8_img, yolos_img)
        
        # CREATE ROUTED IMAGE
        routed_img = img_orig.copy()
        draw = ImageDraw.Draw(routed_img)
        
        COLORS = {
            'UPPER': 'blue',
            'LOWER': 'green',
            'SHOES': 'red',
            'DEFAULT': 'purple'
        }
        
        for roi in final_rois:
            box = roi['box']
            route = roi['route']
            source = roi.get('source', '')
            
            xmin, ymin, xmax, ymax = box
            color = COLORS.get(route, "orange")
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=6)
            
            text_str = f"[{route}] {source}"
            text_bbox = draw.textbbox((xmin, ymin), text_str, font=box_font)
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+4, text_bbox[3]+4], fill=color)
            draw.text((xmin+2, ymin+2), text_str, fill="white", font=box_font)
            
        # Make a quadro side-by-side by appending the routed image to the combined image
        w_comb, h_comb = combined.size
        w_rout, h_rout = routed_img.size
        
        def scale_height(w, h, target_h):
            if h != target_h:
                return int((w / h) * target_h)
            return w
            
        w_rout_scaled = scale_height(w_rout, h_rout, h_comb)
        routed_img = routed_img.resize((w_rout_scaled, h_comb))
        
        final_combined = Image.new('RGB', (w_comb + w_rout_scaled, h_comb))
        final_combined.paste(combined, (0, 0))
        final_combined.paste(routed_img, (w_comb, 0))
        
        draw_combined = ImageDraw.Draw(final_combined)
        draw_combined.rectangle([w_comb, 0, w_comb + 350, 45], fill="black")
        
        try:
            linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font_title = ImageFont.truetype(linux_font_path, 45)
        except IOError:
            warnings.warn("Advertencia: No se pudo cargar la fuente. Usando por defecto. Error: {e}\n\
                Esto puede causar que el texto sea minúsculo o no se vea.")
            font_title = ImageFont.load_default()
            
        draw_combined.text((w_comb + 10, 10), "MoE Slot Router", fill="white", font=font_title)
        
        final_combined.save(os.path.join(out_dir, f"{bid}_compare.jpg"))
        print(f"Saved {bid}_compare.jpg")

    print(f"\nFinished! Visualizations saved to '{out_dir}/'")

if __name__ == "__main__":
    main()
