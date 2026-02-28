import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

class ResidualDomainMapper(nn.Module):
    """Mapeador de dominio con conexión residual y expansión de canal."""
    def __init__(self, dim=1024, hidden_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        delta = self.proj(x)
        out = x + delta  # Conexión residual clave
        return F.normalize(out, p=2, dim=-1)

def info_nce_loss(preds, targets, temperature=0.05):
    """
    Pérdida contrastiva. Atrae pares correctos de la diagonal
    y repele todos los demás cruces de la matriz del batch.
    """
    sim_matrix = torch.matmul(preds, targets.T) / temperature
    labels = torch.arange(preds.size(0)).to(preds.device)
    return F.cross_entropy(sim_matrix, labels)

def train_mapper(bundle_embs, product_embs, epochs=15, batch_size=256, emb_dim=1024):
    """Entrena el mapeador alineando vectores con Cosine Embedding Loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualDomainMapper(dim=emb_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CosineEmbeddingLoss()

    dataset = TensorDataset(torch.tensor(bundle_embs).to(device), torch.tensor(product_embs).to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    target = torch.ones(batch_size).to(device)

    model.train()
    best_model = None
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for b_emb, p_emb in loader:
            optimizer.zero_grad()
            
            mapped_b = model(b_emb)
            
            loss = info_nce_loss(mapped_b, p_emb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/batches:.4f}")
        
        if epoch_loss/batches < best_loss:
            best_loss = epoch_loss/batches
            best_model = model.state_dict()
    
    model.load_state_dict(best_model)
    return model

def load_gr_lite(device):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("No HF_TOKEN found in .env")
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
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embs = outputs.pooler_output
        else:
            embs = outputs.last_hidden_state.mean(dim=1)
    return embs.cpu().numpy()

def main(epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    b_embs_path = "train_b_embs.npy"
    p_embs_path = "train_p_embs.npy"
    
    if os.path.exists(b_embs_path) and os.path.exists(p_embs_path):
        print("Loading pre-extracted embeddings...")
        X_train_bundles = np.load(b_embs_path)
        Y_train_products = np.load(p_embs_path)
        print(f"Loaded {len(X_train_bundles)} samples.")
    else:
        print("\nLoading YOLOv8-Clothing model...")
        try:
            from huggingface_hub import hf_hub_download
            v8_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
            yolo_model = YOLO(v8_model_path)
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            return
            
        print("Loading GR-Lite model (DINOv3 backbone)...")
        gr_model, gr_processor = load_gr_lite(device)
        if not gr_model:
            return
            
        catalog_emb_path = "catalog_grlite_embeddings.npy"
        catalog_ids_path = "valid_grlite_ids.npy"
        
        if os.path.exists(catalog_emb_path) and os.path.exists(catalog_ids_path):
            print("\nLoading precomputed GR-Lite catalog embeddings...")
            catalog_embeddings = np.load(catalog_emb_path)
            valid_catalog_ids = np.load(catalog_ids_path, allow_pickle=True).tolist()
        else:
            print("\nGR-Lite catalog embeddings missing.")
            return
            
        catalog_id_to_idx = {pid: idx for idx, pid in enumerate(valid_catalog_ids)}
        
        df_train = pd.read_csv('data_csvs/bundles_product_match_train.csv')
        
        X_train_list = []
        Y_train_list = []
        
        print("\nExtracting embeddings via Pseudo-labeling (max similarity to catalog)...")
        # Ensure array shapes
        for idx_row in tqdm(range(len(df_train)), desc="Processing train bundles"):
            row = df_train.iloc[idx_row]
            bid = row['bundle_asset_id']
            pid = row['product_asset_id']
            
            if pid not in catalog_id_to_idx:
                continue
                
            img_path = os.path.join("data", "bundles", f"{bid}.jpg")
            if not os.path.exists(img_path):
                continue
                
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
                
            # YOLO crops
            results = yolo_model.predict(img_path, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            crops = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                pad = 15
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img.width, x2 + pad)
                y2 = min(img.height, y2 + pad)
                
                w = x2 - x1
                h = y2 - y1
                if w > 10 and h > 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
            
            crops.append(img) # Mandatory Global Image
            
            # Embeddings
            source_embs = []
            batch_size = 16
            for j in range(0, len(crops), batch_size):
                batch = crops[j:j+batch_size]
                embs = get_embeddings(gr_model, gr_processor, batch, device)
                source_embs.extend(embs)
                
            source_embs = np.array(source_embs)
            
            p_idx = catalog_id_to_idx[pid]
            p_emb = catalog_embeddings[p_idx]
            
            p_emb_norm = p_emb / (np.linalg.norm(p_emb) + 1e-10)
            norms = np.linalg.norm(source_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            source_embs_norm = source_embs / norms
            
            sims = np.dot(source_embs_norm, p_emb_norm.T)
            best_idx = np.argmax(sims)
            
            best_b_emb = source_embs[best_idx]
            
            X_train_list.append(best_b_emb)
            Y_train_list.append(p_emb)
            
        X_train_bundles = np.array(X_train_list)
        Y_train_products = np.array(Y_train_list)
        
        np.save(b_embs_path, X_train_bundles)
        np.save(p_embs_path, Y_train_products)
        print(f"\nSaved extracted embeddings: {len(X_train_bundles)} samples")

    print("\nTraining Domain Mapper...")
    mapper_model = train_mapper(X_train_bundles, Y_train_products, epochs=epochs, batch_size=256, emb_dim=1024)
    
    torch.save(mapper_model.proj.state_dict(), "domain_mapper.pt")
    print("Saved domain_mapper.pt!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    else:
        epochs = 15
    main(epochs)
