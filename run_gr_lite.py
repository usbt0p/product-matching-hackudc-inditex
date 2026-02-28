import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from dotenv import load_dotenv
from tqdm import tqdm
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
        # We need a 1024-D holistic embedding. 
        # Typically for ViT architectures, we take pooler_output or the mean of last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embs = outputs.pooler_output
        else:
            # Global average pooling
            embs = outputs.last_hidden_state.mean(dim=1)
    
    return embs.cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load GR-Lite model
    model_path = "gr_lite.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please wait for the download to finish.")
        return
        
    print("Loading GR-Lite model (DINOv3 backbone)...")
    model, processor = load_gr_lite(device)
    if not model:
        return
    
    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    
    ordered_catalog_ids = df_products['product_asset_id'].tolist()
    test_bundle_ids = df_test['bundle_asset_id'].unique().tolist()
    
    # 1. Cache Catalog Embeddings
    catalog_emb_path = "catalog_grlite_embeddings.npy"
    catalog_ids_path = "valid_grlite_ids.npy"
    
    if os.path.exists(catalog_emb_path) and os.path.exists(catalog_ids_path):
        print("\nLoading precomputed GR-Lite catalog embeddings...")
        catalog_embeddings = np.load(catalog_emb_path)
        valid_catalog_ids = np.load(catalog_ids_path, allow_pickle=True).tolist()
    else:
        print("\nComputing GR-Lite catalog embeddings from local images...")
        embeddings_list = []
        valid_catalog_ids = []
        batch_size = 32
        
        batch_imgs = []
        batch_ids = []
        
        for pid in tqdm(ordered_catalog_ids, desc="Catalog"):
            img_path = os.path.join("data", "products", f"{pid}.jpg")
            if not os.path.exists(img_path):
                continue
            
            try:
                img = Image.open(img_path).convert("RGB")
                batch_imgs.append(img)
                batch_ids.append(pid)
            except Exception:
                continue
                
            if len(batch_imgs) >= batch_size:
                embs = get_embeddings(model, processor, batch_imgs, device)
                embeddings_list.extend(embs)
                valid_catalog_ids.extend(batch_ids)
                batch_imgs = []
                batch_ids = []
                
        # Process remaining
        if len(batch_imgs) > 0:
            embs = get_embeddings(model, processor, batch_imgs, device)
            embeddings_list.extend(embs)
            valid_catalog_ids.extend(batch_ids)
            
        catalog_embeddings = np.array(embeddings_list)
        np.save(catalog_emb_path, catalog_embeddings)
        np.save(catalog_ids_path, valid_catalog_ids)
        print(f"Computed {len(valid_catalog_ids)} catalog embeddings.")

    # Normalize catalog embeddings for dot product (Cosine Similarity)
    catalog_norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    catalog_norms[catalog_norms == 0] = 1e-10
    normalized_catalog = catalog_embeddings / catalog_norms

    # 2. Process Test Bundles
    print("\nExtracting Bundle embeddings and Predicting...")
    
    submission_rows_top15 = []
    submission_rows_top5 = []
    
    batch_size = 16
    for i in tqdm(range(0, len(test_bundle_ids), batch_size), desc="Bundles"):
        batch_bids = test_bundle_ids[i:i+batch_size]
        batch_imgs = []
        valid_bids = []
        
        for bid in batch_bids:
            img_path = os.path.join("data", "bundles", f"{bid}.jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_imgs.append(img)
                    valid_bids.append(bid)
                except Exception:
                    pass
                
        if len(batch_imgs) == 0:
            continue
            
        query_embs = get_embeddings(model, processor, batch_imgs, device)
            
        for bid, query_feat in zip(valid_bids, query_embs):
            q_norm = np.linalg.norm(query_feat)
            if q_norm == 0: q_norm = 1e-10
            normalized_query = query_feat / q_norm
            
            # Cosine similarity
            similarities = np.dot(normalized_catalog, normalized_query.T).flatten()
            
            # Top 15
            top_indices = np.argsort(similarities)[::-1][:15]
            
            for idx in top_indices:
                submission_rows_top15.append({"bundle_asset_id": bid, "product_asset_id": valid_catalog_ids[idx]})
                
            # Top 5
            for idx in top_indices[:5]:
                submission_rows_top5.append({"bundle_asset_id": bid, "product_asset_id": valid_catalog_ids[idx]})

    # 3. Save submissions
    df_sub_15 = pd.DataFrame(submission_rows_top15)
    df_sub_15.to_csv("submission_gr_lite_top15.csv", index=False)
    
    df_sub_5 = pd.DataFrame(submission_rows_top5)
    df_sub_5.to_csv("submission_gr_lite_top5.csv", index=False)
    
    print(f"\nSaved Top-15 submission to submission_gr_lite_top15.csv! ({len(df_sub_15)} rows)")
    print(f"Saved Top-5 submission to submission_gr_lite_top5.csv! ({len(df_sub_5)} rows)")

if __name__ == "__main__":
    main()
