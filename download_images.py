'''Download images from the dataset.

Expects the following CSV files in the data_csvs/ folder:
- bundles_dataset.csv
- product_dataset.csv

With this format:

```bundles_dataset.csv
bundle_asset_id,bundle_id_section,bundle_image_url
B_dc4cb6c331e3,2,https://<url>
B_0115d9356e71,1,https://<url>
B_356f9aad7c3e,1,https://<url>
B_a5edb4861ebe,3,https://<url>
```

```product_dataset.csv
product_asset_id,product_image_url,product_description
I_881901188071,https://<url>,HAND BAG-RUCKSACK
I_f052906c12b9,https://<url>,FLAT SHOES
I_4e2056ae9d96,https://<url>,TROUSERS
I_d3e8b8b3a6f1,https://<url>,T-SHIRT
```
'''

import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session

global_session = setup_session()

def download_image(url, save_path):
    if pd.isna(url):
        return False
    if os.path.exists(save_path):
        return True # Already downloaded
    
    try:
        response = global_session.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        # Save a dummy file or ignore
        return False

def download_dataset(df, id_col, url_col, output_dir, desc_str):
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    # Drop NAs
    df = df.dropna(subset=[url_col])
    
    print(f"Starting {desc_str} download ({len(df)} images) to {output_dir}")
    
    # Use max_workers=10 so we don't spam Zara CDN too hard and get blocked
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _, row in df.iterrows():
            asset_id = str(row[id_col])
            url = row[url_col]
            save_path = os.path.join(output_dir, f"{asset_id}.jpg")
            tasks.append(executor.submit(download_image, url, save_path))
            
        success_count = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc=desc_str):
            if future.result():
                success_count += 1
                
    print(f"Finished {desc_str}. Successfully downloaded {success_count}/{len(df)} images.\n")

def main():
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    
    print("Loading CSVS...")
    df_bundles = pd.read_csv('data_csvs/bundles_dataset.csv')
    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    
    # Download Bundles
    bundles_dir = os.path.join(base_dir, "bundles")
    download_dataset(df_bundles, "bundle_asset_id", "bundle_image_url", bundles_dir, "Bundles")
    
    # Download Products
    products_dir = os.path.join(base_dir, "products")
    download_dataset(df_products, "product_asset_id", "product_image_url", products_dir, "Products")
    
if __name__ == "__main__":
    main()
