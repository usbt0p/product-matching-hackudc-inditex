import os
import torch
import json
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from semantic_filtering import SemanticFilter
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading Grounding DINO...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    sf = SemanticFilter()
    
    import pandas as pd
    df_test = pd.read_csv('data_csvs/bundles_product_match_test.csv')
    test_bundle_ids = df_test['bundle_asset_id'].unique().tolist()
    
    macro_cache = {}
    cache_path = "test_dino_macro.json"
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            macro_cache = json.load(f)
            
    # Process only what's missing
    missing = [b for b in test_bundle_ids if b not in macro_cache]
    if not missing:
        print("All test bundles already precomputed!")
        return
        
    print(f"Extracting DINO macro-regions for {len(missing)} test bundles...")
    for bid in tqdm(missing):
        img_path = os.path.join("data", "bundles", f"{bid}.jpg")
        if not os.path.exists(img_path):
            macro_cache[bid] = []
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            macros = sf.extract_macro_regions(img, model, processor, device)
            # Convert float32 numpy arrays to generic lists for JSON serialization
            serializable_macros = []
            for m in macros:
                serializable_macros.append({
                    "box": [float(x) for x in m["box"]],
                    "zone": m["zone"],
                    "score": float(m["score"])
                })
            macro_cache[bid] = serializable_macros
        except Exception as e:
            print(f"Error on {bid}: {e}")
            macro_cache[bid] = []
            
        # Periodically save
        if len(macro_cache) % 50 == 0:
            with open(cache_path, 'w') as f:
                json.dump(macro_cache, f)
                
    with open(cache_path, 'w') as f:
        json.dump(macro_cache, f)
        
    print("Done precomputing DINO macro-regions.")

if __name__ == "__main__":
    main()
