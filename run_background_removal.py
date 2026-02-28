import os
import warnings
warnings.filterwarnings("ignore")
os.environ["ORT_LOG_LEVEL"] = "3"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm
import pandas as pd
import os

def main():
    crops_dir = "bundle_crops"
    out_dir = "bundle_crops_rembg"
    os.makedirs(out_dir, exist_ok=True)
    
    bundles = os.listdir(crops_dir)
    for bundle in tqdm(bundles, desc="Background Removal"):
        b_in = os.path.join(crops_dir, bundle)
        b_out = os.path.join(out_dir, bundle)
        os.makedirs(b_out, exist_ok=True)
        
        for crop in os.listdir(b_in):
            if not crop.endswith(".jpg"): continue
            in_path = os.path.join(b_in, crop)
            out_path = os.path.join(b_out, crop)
            if os.path.exists(out_path): continue
            
            try:
                img = Image.open(in_path).convert("RGBA")
                no_bg = remove(img)
                white_bg = Image.new("RGBA", no_bg.size, "WHITE")
                white_bg.paste(no_bg, (0, 0), no_bg)
                final_img = white_bg.convert("RGB")
                final_img.save(out_path)
            except Exception as e:
                print(f"Failed {in_path}: {e}")

if __name__ == '__main__':
    main()
