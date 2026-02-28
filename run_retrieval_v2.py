import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel
import open_clip
from tqdm import tqdm
from collections import defaultdict

def main():
    print("Loading datasets...")
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    test_bundles = df_test['bundle_asset_id'].unique()
    
    crops_dir = "bundle_crops"
    available_bundles = [b for b in test_bundles if os.path.exists(os.path.join(crops_dir, b))]
    print(f"Found crop directories for {len(available_bundles)} bundles.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DINOv2
    print("Loading DINOv2 model...")
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    # Load Marqo-FashionSigLIP
    print("Loading Marqo-FashionSigLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
    clip_model.eval().to(device)

    print("\nLoading precomputed catalog embeddings cache (DINO)...")
    dino_cat_emb = np.load("catalog_embeddings.npy")
    dino_cat_ids = np.load("valid_catalog_ids.npy", allow_pickle=True).tolist()

    print("\nLoading precomputed catalog embeddings cache (CLIP)...")
    clip_cat_emb = np.load("catalog_clip_embeddings.npy")
    clip_cat_ids = np.load("valid_clip_ids.npy", allow_pickle=True).tolist()

    # Align the catalogs perfectly in case there's a mismatch
    id_to_clip_idx = {pid: idx for idx, pid in enumerate(clip_cat_ids)}
    
    valid_pids = []
    aligned_dino = []
    aligned_clip = []
    
    for i, pid in enumerate(dino_cat_ids):
        if pid in id_to_clip_idx:
            valid_pids.append(pid)
            aligned_dino.append(dino_cat_emb[i])
            aligned_clip.append(clip_cat_emb[id_to_clip_idx[pid]])
            
    final_dino_emb = np.array(aligned_dino)
    final_clip_emb = np.array(aligned_clip)
    print(f"Perfectly aligned {len(valid_pids)} common items between arrays.")

    submission_rows = []
    
    # Weight configuration
    W_DINO = 0.5
    W_CLIP = 0.5
    
    for bundle_id in tqdm(available_bundles, desc="Dual Predicting"):
        bundle_crops_path = os.path.join(crops_dir, bundle_id)
        crop_files = [f for f in os.listdir(bundle_crops_path) if f.endswith(".jpg") or f.endswith(".png")]
        
        bundle_scores = defaultdict(float)
        
        if len(crop_files) == 0:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": ""})
            continue
            
        for crop_file in crop_files:
            img_path = os.path.join(bundle_crops_path, crop_file)
            try:
                img = Image.open(img_path).convert("RGB")
                # Filter out crops that are too small (noise)
                if img.width < 100 or img.height < 100:
                    continue
            except Exception:
                continue

            # DINO logic
            d_inputs = dino_processor(images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                d_outputs = dino_model(**d_inputs)
            crop_emb_dino = d_outputs.last_hidden_state[:, 0, :].cpu().numpy()
            d_sims = cosine_similarity(crop_emb_dino, final_dino_emb)[0]
            
            # CLIP logic
            c_input = clip_preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                c_emb = clip_model.encode_image(c_input)
                c_emb /= c_emb.norm(dim=-1, keepdim=True)
                crop_emb_clip = c_emb.cpu().numpy()
            c_sims = cosine_similarity(crop_emb_clip, final_clip_emb)[0]
            
            # Fuse similarities
            fused_sims = (W_DINO * d_sims) + (W_CLIP * c_sims)
            
            top_indices = np.argsort(fused_sims)[::-1][:5]
            for idx in top_indices:
                pid = valid_pids[idx]
                score = fused_sims[idx]
                if score > bundle_scores[pid]:
                    bundle_scores[pid] = score
                    
        if not bundle_scores:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": ""})
            continue
            
        sorted_pids = sorted(bundle_scores.keys(), key=lambda k: bundle_scores[k], reverse=True)
        top_predicted_ids = sorted_pids[:15]
        
        for pred_id in top_predicted_ids:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pred_id})

    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv("submission_level2.csv", index=False)
    print(f"\nSuccess! Generated submission_level2.csv with {len(df_submission)} records.")

if __name__ == "__main__":
    main()
