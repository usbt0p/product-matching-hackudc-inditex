import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import open_clip

# Global session with retries and headers to prevent Zara CDN 404/403 blocks
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[403, 429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})

def load_image_safe(url):
    try:
        response = session.get(url, stream=True, timeout=10)
        response.raise_for_status()
        time.sleep(0.05) # Small delay to be polite
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading Marqo FashionSigLIP model...")
    # Using marqo-fashionSigLIP for bleeding edge Fashion retrieval
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
    model.eval().to(device)

    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    ordered_catalog_ids = df_products['product_asset_id'].tolist()
    catalog_urls = df_products['product_image_url'].tolist()
    
    embeddings_list = []
    valid_catalog_ids = []
    
    print("Caching CLIP catalog embeddings from local data directory...")
    for pid in tqdm(ordered_catalog_ids, desc="CLIP Catalog"):
        local_path = os.path.join("data", "products", f"{pid}.jpg")
        
        # If the download script hasn't gotten to it yet, or it failed, fallback to none
        if not os.path.exists(local_path):
            continue
            
        try:
            img = Image.open(local_path).convert("RGB")
        except Exception:
            continue
        
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy()[0]
            
        embeddings_list.append(emb)
        valid_catalog_ids.append(pid)

    catalog_embeddings = np.array(embeddings_list)
    np.save("catalog_clip_embeddings.npy", catalog_embeddings)
    np.save("valid_clip_ids.npy", [str(x) for x in valid_catalog_ids])
    print("Done! CLIP Cache created.")

if __name__=="__main__":
    main()
