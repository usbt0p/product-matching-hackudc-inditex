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
    # Create a new image combining three horizontally
    w1, h1 = img1.size
    w2, h2 = img2.size
    w3, h3 = img3.size
    
    # Scale to match height
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
    
    # Font
    try:
        # Usamos la ruta completa de Linux que definiste arriba
        linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        font = ImageFont.truetype(linux_font_path, 45)       # Tamaño para las cajas de YOLO
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading descriptions...")
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
    
    random.seed(42)
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
    
    # # For bounding box texts
    # try:
    #     box_font = ImageFont.truetype("DejavuSans-Bold.ttf", 20)
    # except IOError:
    #     box_font = ImageFont.load_default()

    # Font
    try:
        # Usamos la ruta completa de Linux que definiste arriba
        linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        box_font = ImageFont.truetype(linux_font_path, 35)       # Tamaño para las cajas de YOLO
        title_font = ImageFont.truetype(linux_font_path, 55) # Tamaño para los títulos
    except IOError as e:
        print(f"Advertencia: No se pudo cargar la fuente. Usando por defecto. Error: {e}")
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
                
            filtered_dino = []
            for pred in dino_preds:
                keep = True
                for kept in filtered_dino:
                    if compute_iou(pred["box"], kept["box"]) > 0.6:
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
                    'jumpsuit', 'glasses', 'hat', 'hair accessory', 'tie', 'glove',
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
                
                filtered_yolos = []
                for pred in yolos_preds:
                    if pred["score"] < 0.5:
                        continue
                        
                    keep = True
                    for kept in filtered_yolos:
                        if compute_iou(pred["box"], kept["box"]) > 0.6:
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
                    
                    xmin, ymin, xmax, ymax = box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=4)
                    
                    text = f"{cls_name} {score:.2f}"
                    text_bbox = draw.textbbox((xmin, ymin), text, font=box_font)
                    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2]+4, text_bbox[3]+4], fill="green")
                    draw.text((xmin+2, ymin+2), text, fill="white", font=box_font)
            except Exception as e:
                print(f"YOLOS failed for {bid}: {e}")
        
        # Combine
        combined = create_triple_side_by_side(dino_img, v8_img, yolos_img)
        combined.save(os.path.join(out_dir, f"{bid}_compare.jpg"))
        print(f"Saved {bid}_compare.jpg")

    print(f"\nFinished! Visualizations saved to '{out_dir}/'")

if __name__ == "__main__":
    main()
