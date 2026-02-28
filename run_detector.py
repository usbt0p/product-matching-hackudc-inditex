import os
import time
import torch
import pandas as pd
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Setup robust session
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[403, 429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

def load_image_safe(url):
    try:
        response = session.get(url, stream=True, timeout=10)
        response.raise_for_status()
        time.sleep(0.05)
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading datasets...")
    df_bundles = pd.read_csv('data_csvs/bundles_dataset.csv')
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    test_bundles = df_test['bundle_asset_id'].unique()
    
    # We process the entire test set
    subset_bundles = test_bundles
    
    print("Loading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # The prompt explicitly looking for clothing items
    text_prompt = "clothing. garment. shirt. pants. dress. skirt. outerwear. shoes. bag. hat. accessory."
    
    output_dir = "detected_bundles"
    crops_dir = "bundle_crops"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    print(f"Processing {len(subset_bundles)} bundles...")
    for bundle_id in tqdm(subset_bundles):
        try:
            bundle_url = df_bundles[df_bundles['bundle_asset_id'] == bundle_id]['bundle_image_url'].iloc[0]
        except Exception:
            continue
            
        if pd.isna(bundle_url):
            continue

        img = load_image_safe(bundle_url)
        if img is None:
            continue
            
        # Run Detection
        inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Target sizes are given as (height, width)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=target_sizes
        )[0]
        
        # Draw bounding boxes and save crops
        draw = ImageDraw.Draw(img)
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]
        
        bundle_crops_dir = os.path.join(crops_dir, bundle_id)
        os.makedirs(bundle_crops_dir, exist_ok=True)
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            ymin, xmin, ymax, xmax = box # Depending on format. Grounding DINO typically outputs [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
            
            # Draw on original image for visualization
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            draw.text((xmin, ymin), f"{label} {score:.2f}", fill="red")
            
            # Crop and save Region of Interest (ROI)
            # Expand the crop slightly to give context
            pad = 10
            crop_box = (max(0, xmin-pad), max(0, ymin-pad), min(img.width, xmax+pad), min(img.height, ymax+pad))
            crop_img = img.crop(crop_box)
            crop_img.save(os.path.join(bundle_crops_dir, f"crop_{i}_{score:.2f}.jpg"))
            
        # Save visualization
        img.save(os.path.join(output_dir, f"{bundle_id}_detected.jpg"))
        
    print(f"Done! Check the '{output_dir}' directory for visualized bounded boxes and '{crops_dir}' for the individual garments.")

if __name__ == "__main__":
    main()
