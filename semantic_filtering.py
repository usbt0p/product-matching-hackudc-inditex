import pandas as pd
import numpy as np
import torch
import os
from PIL import Image

class SemanticFilter:
    def __init__(self, data_dir="data_csvs"):
        self.data_dir = data_dir
        self.catalog_meta = {}
        
        # Taxonomic Mapping to Body Zones (Extendable)
        # We classify descriptions into: UPPER, LOWER, FEET, HEAD, ACCESSORY, FULL_BODY
        self.body_zone_map = {
            "UPPER": [
                "shirt", "t-shirt", "top", "sweater", "cardigan", "jacket", "vest", 
                "coat", "sweatshirt", "blazer", "blouse", "waistcoat", "polo", "hoodie", "pullover"
            ],
            "LOWER": [
                "trousers", "pants", "jeans", "shorts", "skirt", "leggings", "joggers", "chinos", "bermudas"
            ],
            "FEET": [
                "shoes", "sneakers", "boots", "sandals", "heels", "flats", "mules", "loafers", "trainers", "footwear", "ankle boot"
            ],
            "HEAD": [
                "hat", "cap", "beanie", "glasses", "sunglasses", "hair accessory", "scrunchie", "headband"
            ],
            "ACCESSORY": [
                "bag", "wallet", "belt", "buckle", "scarf", "tie", "necktie", "glove", "necklace", "backpack", "rucksack"
            ],
            "FULL_BODY": [
                "dress", "jumpsuit", "overall", "romper", "suit"
            ]
        }
        
    def _parse_zone(self, description):
        if not isinstance(description, str):
            return "UNKNOWN"
            
        desc_lower = description.lower()
        for zone, keywords in self.body_zone_map.items():
            for kw in keywords:
                # We do simple substring since descriptions might be compound like "HAND BAG-RUCKSACK"
                if kw in desc_lower:
                    return zone
        return "UNKNOWN"
        
    def precompute_metadata(self, save_path="catalog_semantic_meta.pkl"):
        """
        Builds a map: product_asset_id -> {"inferred_section": [1,2,3], "body_zone": "UPPER"/"LOWER"/...}
        """
        if os.path.exists(save_path):
            print(f"Loading cached semantics from {save_path}...")
            self.catalog_meta = pd.read_pickle(save_path)
            return self.catalog_meta

        print("Building Semantic Catalog Metadata...")
        
        # Load CSVs
        bundles_df = pd.read_csv(os.path.join(self.data_dir, "bundles_dataset.csv"))
        train_matches_df = pd.read_csv(os.path.join(self.data_dir, "bundles_product_match_train.csv"))
        products_df = pd.read_csv(os.path.join(self.data_dir, "product_dataset.csv"))
        
        # Map bundle_id to its section
        bundle_to_section = dict(zip(bundles_df['bundle_asset_id'], bundles_df['bundle_id_section']))
        
        # Map products to sections they appear in using vectorized map
        train_matches_df['section'] = train_matches_df['bundle_asset_id'].map(bundle_to_section)
        
        # Group by product_asset_id and aggregate sections into sets
        print("Grouping train sections...")
        prod_to_sections = train_matches_df.dropna(subset=['section']).groupby('product_asset_id')['section'].apply(set).to_dict()
        
        # Build final metadata dictionary using grouped descriptions to minimize _parse_zone calls
        print("Parsing body zones...")
        meta_dict = {}
        # Fill products that have nan descriptions with empty string to avoid groupby failing
        products_df['product_description'] = products_df['product_description'].fillna("")
        
        for desc, group in products_df.groupby('product_description'):
            zone = self._parse_zone(desc)
            for p_id in group['product_asset_id']:
                sections = prod_to_sections.get(p_id, set())
                meta_dict[p_id] = {
                    "inferred_sections": sections,
                    "body_zone": zone
                }
        pd.to_pickle(meta_dict, save_path)
        print("Semantic Catalog Built and Saved!")
        return self.catalog_meta

    def get_bundle_section(self, bundle_id):
        bundles_df = pd.read_csv(os.path.join(self.data_dir, "bundles_dataset.csv"))
        match = bundles_df[bundles_df['bundle_asset_id'] == bundle_id]
        if not match.empty:
            return match['bundle_id_section'].iloc[0]
        return None

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou

    def extract_macro_regions(self, img_orig, dino_model, dino_processor, device="cuda"):
        """Run Grounding DINO to extract macro regions."""
        if not dino_model: return []
        
        prompt = "head. upper body. lower body. feet. bag."
        inputs = dino_processor(images=img_orig, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            
        target_sizes = torch.tensor([img_orig.size[::-1]])
        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3, # We want to catch the general bodies, even if low confidence
            text_threshold=0.3,
            target_sizes=target_sizes
        )[0]
        
        macro_boxes = []
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            # Map DINO text label back to our standardized Body Zones
            l = label.lower()
            zone = "UNKNOWN"
            if "upper body" in l: zone = "UPPER"
            elif "lower body" in l: zone = "LOWER"
            elif "feet" in l: zone = "FEET"
            elif "head" in l: zone = "HEAD"
            elif "bag" in l: zone = "ACCESSORY"
            
            if zone != "UNKNOWN":
                macro_boxes.append({
                    "box": box,
                    "zone": zone,
                    "score": score
                })
        return macro_boxes

    def assign_zones_to_micro_crops(self, micro_boxes, macro_boxes):
        """
        Assigns the macro Body Zone to each YOLO crop based on the highest intersection area.
        micro_boxes: list of YOLO boxes [x1, y1, x2, y2]
        """
        assigned_zones = []
        for m_box in micro_boxes:
            best_zone = "UNKNOWN"
            best_iou = 0.0
            
            for macro in macro_boxes:
                iou = self.compute_iou(m_box, macro["box"])
                if iou > best_iou and iou > 0.1: # Minimum IoU to be considered inside macro region
                    best_iou = iou
                    best_zone = macro["zone"]
                    
            assigned_zones.append(best_zone)
        return assigned_zones

    def apply_similarity_filters(self, raw_sims, valid_catalog_ids, source_zone, bundle_section=None):
        """
        Adjust cosine similarity array based on semantic rules.
        - Soft constraint (Multiplier 0.7): Product inferred section differs from bundle section.
        - Hard constraint (Multiplier 0.0): Spatial zone of crop contradicts taxonomic zone of product.
        """
        filtered_sims = raw_sims.copy()
        
        for idx, p_id in enumerate(valid_catalog_ids):
            meta = self.catalog_meta.get(p_id)
            if not meta: continue
            
            # 1. Soft Filter: Section Mismatch
            # If we know the bundle section, and we know the product sections, and there is no overlap
            if bundle_section is not None and meta["inferred_sections"]:
                if bundle_section not in meta["inferred_sections"]:
                    filtered_sims[idx] *= 0.7 # Penalize cross-section recommendations
                    
            # 2. Hard Filter: Spatial Zone Contradiction
            prod_zone = meta["body_zone"]
            if source_zone != "UNKNOWN" and prod_zone != "UNKNOWN":
                # Certain things are ambiguous or overlap, but direct opposites should be 0.
                if source_zone == "UPPER" and prod_zone in ["LOWER", "FEET"]:
                    filtered_sims[idx] *= 0.0
                elif source_zone == "LOWER" and prod_zone in ["UPPER", "HEAD", "FEET"]:
                    filtered_sims[idx] *= 0.0
                elif source_zone == "FEET" and prod_zone in ["UPPER", "LOWER", "HEAD"]:
                    filtered_sims[idx] *= 0.0
                elif source_zone == "HEAD" and prod_zone in ["FEET", "LOWER"]:
                    filtered_sims[idx] *= 0.0

        return filtered_sims
