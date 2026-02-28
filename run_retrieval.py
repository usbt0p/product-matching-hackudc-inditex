import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from collections import defaultdict

def main():
    print("Loading datasets...")
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    test_bundles = df_test['bundle_asset_id'].unique()
    
    # We will only infer on the bundles that have cropped directories
    # Since we only extracted 10 bundles for now in run_detector.py
    crops_dir = "bundle_crops"
    available_bundles = [b for b in test_bundles if os.path.exists(os.path.join(crops_dir, b))]
    
    print(f"Found crop directories for {len(available_bundles)} bundles.")
    if len(available_bundles) == 0:
        print("No crops found. Run run_detector.py first on all bundles.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DINOv2 model since we already have its catalog embeddings cached!
    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    embeddings_file = "catalog_embeddings.npy"
    ids_file = "valid_catalog_ids.npy"

    if os.path.exists(embeddings_file) and os.path.exists(ids_file):
        print("\nLoading precomputed catalog embeddings cache...")
        catalog_embeddings = np.load(embeddings_file)
        valid_catalog_ids = np.load(ids_file, allow_pickle=True).tolist()
    else:
        print("\nError: Catalog embeddings not found. Please run generate_submission.py once to cache them.")
        return

    print("\nPredicting from isolated ROIs (crops)...")
    
    submission_rows = []
    
    for bundle_id in tqdm(available_bundles, desc="Predicting Test Bundles"):
        bundle_crops_path = os.path.join(crops_dir, bundle_id)
        crop_files = [f for f in os.listdir(bundle_crops_path) if f.endswith(".jpg")]
        
        bundle_scores = defaultdict(float)
        
        if len(crop_files) == 0:
            # Fallback if detection found 0 boxes
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": ""})
            continue
            
        # Process each crop in the bundle
        for crop_file in crop_files:
            img_path = os.path.join(bundle_crops_path, crop_file)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            inputs = processor(images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            crop_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            similarities = cosine_similarity(crop_emb, catalog_embeddings)[0]
            
            # Get Top-5 for THIS specific crop
            top_k_crop = 5
            top_indices = np.argsort(similarities)[::-1][:top_k_crop]
            
            for rank, idx in enumerate(top_indices):
                pid = valid_catalog_ids[idx]
                score = similarities[idx]
                
                # We can aggregate scores. E.g., Max score wins for the whole bundle
                if score > bundle_scores[pid]:
                    bundle_scores[pid] = score
                    
        # After processing all crops for the bundle, get the top 15 overall products
        if not bundle_scores:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": ""})
            continue
            
        sorted_pids = sorted(bundle_scores.keys(), key=lambda k: bundle_scores[k], reverse=True)
        top_predicted_ids = sorted_pids[:15]
        
        # A row per product in the bundle
        for pred_id in top_predicted_ids:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pred_id})

    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv("submission_roi.csv", index=False)
    
    print(f"\nSuccess! Generated submission_roi.csv with {len(df_submission)} records.")

if __name__ == "__main__":
    main()
