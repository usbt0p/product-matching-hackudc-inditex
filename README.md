# Fashion Retrieval Pipeline 🚀

This repository contains an advanced Two-Stage Multimodal AI Pipeline designed to match messy real-world photography `bundles` against standard studio `products` for the Inditex fashion retrieval challenge.

## 🛠️ Pipeline Flow

The workflow is decoupled into dedicated, modular steps to maximize speed and logic separation:

### 1. Data Downloading 📥
* **Script:** `download_images.py`
* **What it does:** Downloads all Zara images from their CDNs concurrently (making sure it isn't blocked by 403s limits), saving them into local nested directories `data/bundles/` and `data/products/` locally so subsequent embeddings won't be I/O bounded ever again.

### 2. Zero-Shot Detection & Region of Interest (ROI) 🔍
* **Script:** `run_detector.py`
* **What it does:** Uses `Grounding DINO` (a State-of-the-Art language-image object detector) to automatically detect any clothing, garment, bag or shoes inside a `bundle` photo, regardless of what it is. It then **crops** these items so background noise is ignored and saves them into the `bundle_crops/` directory.

### 3. Catalog Embedding Storage 🧠
* **Script:** `run_cache_clip.py`
* **What it does:** Loading all 27,000+ catalog studio images every time we want to predict would be madness. This scripts takes the fashion catalog and converts every piece into dense mathematical vectors (features/embeddings) using **Marqo-FashionSigLIP** (a model exclusively fine-tuned for high fashion). The result gets cached into a fast numpy `.npy` file. *(Note: We also precomputed DINOv2 embeddings in `catalog_embeddings.npy`)*.

### 4. Dual-Ensemble Retrieval & Inference 🎯
* **Script:** `run_retrieval_v2.py`
* **What it does:** The grand finale. It reads all our isolated cropped bounding boxes (`bundle_crops/`) and embeds them through **both** DINOv2 (perfect for shapes/textures/seams) and FashionSigLIP (perfect for conceptual/style matching). It calculates the 50/50 averaged cosine similarity between the cropped image and the 27,000 possibilities, outputting the `Top 15` most similar products in `submission_level2.csv`.
* *Bonus:* It has size constraints built-in to skip ultra-small bad crops that DINO accidentally highlights.

### (Optional / Broken) Background Removal 🖼️
* **Script:** `run_background_removal.py`
* **What it does:** Tries to use `rembg` (U-Net) to completely remove complex street-views from crops, standardizing the background to plain white. *Currently skipped* because it generates too many artifacts that confuse embedding networks.

## 🚀 How to Run End-to-End
```bash
# 1. Download database locally (takes ~30 mins due to Zara rate limits)
python download_images.py

# 2. Extract crops for the test bundles
python run_detector.py

# 3. Create Fashion CLIP memory cache
python run_cache_clip.py

# 4. Generate final predictions (Top 15 per bundle)
python run_retrieval_v2.py
```
