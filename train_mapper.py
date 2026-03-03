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

from run_gr_lite import get_embeddings, load_gr_lite

warnings.filterwarnings("ignore")

from torch.optim.swa_utils import AveragedModel, SWALR

class SuperDomainMapper(nn.Module):
    """Mapper residual con temperatura aprendible tipo CLIP.

    Mejoras sobre ResidualDomainMapper:
    - logit_scale aprendible (Learnable Temperature, estilo CLIP/SigLIP)
    - Input Dropout para robustez adicional
    - Compatible con SWA y grad-clipping
    """
    def __init__(self, dim=1024, hidden_dim=2048, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std
        # Inicialización: log(1 / 0.07)  ≈  2.659  (temperatura inicial ~0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.proj = nn.Sequential(
            nn.Dropout(0.1),          # Input dropout
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.noise_std
        delta = self.proj(x)
        out = x + delta
        return F.normalize(out, p=2, dim=-1)


# Alias para retrocompatibilidad con checkpoints ya guardados
ResidualDomainMapper = SuperDomainMapper


def hard_negative_info_nce_loss(preds, targets, logit_scale, margin=0.15, hard_ratio=0.15):
    """InfoNCE con Online Hard Negative Mining (OHNM) y temperatura aprendible."""
    scale = torch.clamp(logit_scale.exp(), max=100.0)
    sim_matrix = torch.matmul(preds, targets.T) * scale

    idx = torch.arange(preds.size(0), device=preds.device)

    # Margen sobre los positivos (diagonal)
    sim_matrix[idx, idx] -= (margin * scale)

    # Máscara para aislar negativos
    mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    mask[idx, idx] = False
    negatives = sim_matrix[mask].view(preds.size(0), -1)

    # Top-K negativos más duros
    k = max(1, int(negatives.size(1) * hard_ratio))
    hard_negatives, _ = torch.topk(negatives, k, dim=1)

    # Reconstruir logits: columna 0 = positivo, resto = hard negatives
    positives = sim_matrix[idx, idx].unsqueeze(1)
    hard_logits = torch.cat([positives, hard_negatives], dim=1)

    labels = torch.zeros(preds.size(0), dtype=torch.long, device=preds.device)
    return F.cross_entropy(hard_logits, labels)


def memory_bank_loss(preds, targets, memory_bank, logit_scale, margin=0.1, smoothing=0.15):
    """InfoNCE con Cross-Batch Memory (XBM) y Label Smoothing.

    - El positivo siempre es índice 0 en los logits.
    - Los negativos incluyen el batch actual (~511) + la memoria histórica (~8192).
    - Label smoothing evita que el modelo colapse con pseudo-etiquetas ruidosas.
    """
    scale = torch.clamp(logit_scale.exp(), max=100.0)

    # Positivo con margen
    pos_sim = (preds * targets).sum(dim=-1, keepdim=True) - margin

    # Negativos del batch actual (excluye la diagonal)
    neg_sim_batch = torch.matmul(preds, targets.T)
    eye = torch.eye(preds.size(0), device=preds.device, dtype=torch.bool)
    neg_sim_batch.masked_fill_(eye, -1e9)

    # Negativos históricos del banco de memoria
    neg_sim_memory = torch.matmul(preds, memory_bank.T)

    # Concatenar: [positivo | batch_negs | memory_negs]
    logits = torch.cat([pos_sim, neg_sim_batch, neg_sim_memory], dim=1) * scale
    labels = torch.zeros(preds.size(0), dtype=torch.long, device=preds.device)
    return F.cross_entropy(logits, labels, label_smoothing=smoothing)


def train_super_mapper(X_bundles, Y_products, epochs=600, lr=3e-4):
    """Entrena el SuperDomainMapper con MixUp (alpha=0.2), OHNM (top-15%) y OneCycleLR."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapper = SuperDomainMapper().to(device)
    optimizer = optim.AdamW(mapper.parameters(), lr=lr, weight_decay=1e-3)

    dataset = TensorDataset(
        torch.tensor(X_bundles).float(),
        torch.tensor(Y_products).float()
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # OneCycleLR: warmup automático en el primer 10% de pasos, luego cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(loader),
        epochs=epochs,
        pct_start=0.1,        # 10% warmup
        anneal_strategy='cos',
        div_factor=25,        # LR inicial = max_lr / 25
        final_div_factor=1e4  # LR final = max_lr / (25 * 1e4)
    )

    swa_model = AveragedModel(mapper)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5)
    swa_start = int(epochs * 0.7)  # SWA arranca en el último 30%

    for epoch in range(epochs):
        mapper.train()
        epoch_loss = 0.0
        batches = 0

        for b_emb, p_emb in loader:
            b_emb, p_emb = b_emb.to(device), p_emb.to(device)
            optimizer.zero_grad()

            # --- Manifold MixUp (30% de probabilidad, alpha=0.2) ---
            if np.random.rand() < 0.3:
                alpha = 0.2  # ajustado: mezcla más conservadora, evita barro semántico
                lam = float(np.random.beta(alpha, alpha))
                index = torch.randperm(b_emb.size(0), device=device)
                b_emb = lam * b_emb + (1 - lam) * b_emb[index]
                p_emb = lam * p_emb + (1 - lam) * p_emb[index]
                b_emb = F.normalize(b_emb, p=2, dim=-1)
                p_emb = F.normalize(p_emb, p=2, dim=-1)
            # --------------------------------------------------------

            mapped_b = mapper(b_emb)
            loss = hard_negative_info_nce_loss(mapped_b, p_emb, mapper.logit_scale)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            # OneCycleLR necesita step() en cada iteración, no por época
            if epoch <= swa_start:
                scheduler.step()

            epoch_loss += loss.item()
            batches += 1

        if epoch > swa_start:
            swa_model.update_parameters(mapper)
            swa_scheduler.step()

        temp = mapper.logit_scale.exp().item()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/batches:.4f} | Temp: {1/temp:.4f} | LR: {current_lr:.2e}")

    torch.optim.swa_utils.update_bn(loader, swa_model)
    return swa_model.module


def train_xbm_mapper(X_bundles, Y_products, epochs=500, lr=3e-4, mem_size=8192):
    """SuperDomainMapper con Cross-Batch Memory (XBM) + Label Smoothing + SWA.

    La memoria FIFO guarda los últimos `mem_size` embeddings de producto,
    ampliando el campo de negativos de ~511 (batch) a ~8703 sin coste de VRAM extra.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapper = SuperDomainMapper().to(device)
    optimizer = optim.AdamW(mapper.parameters(), lr=lr, weight_decay=1e-3)

    dataset = TensorDataset(
        torch.tensor(X_bundles).float(),
        torch.tensor(Y_products).float()
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=max(1, len(loader)),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
    )

    swa_model   = AveragedModel(mapper)
    swa_start   = int(epochs * 0.8)  # SWA arranca en el último 20%

    # Banco de memoria inicializado aleatoriamente (normalizado)
    memory_bank = F.normalize(torch.randn(mem_size, 1024), dim=1).to(device)
    ptr = 0

    for epoch in range(epochs):
        mapper.train()
        total_loss = 0.0
        batches = 0

        for b_emb, p_emb in loader:
            b_emb, p_emb = b_emb.to(device), p_emb.to(device)
            optimizer.zero_grad()

            # MixUp conservador (alpha=0.2)
            if np.random.rand() < 0.3:
                lam = float(np.random.beta(0.2, 0.2))
                idx = torch.randperm(b_emb.size(0), device=device)
                b_emb = F.normalize(lam * b_emb + (1 - lam) * b_emb[idx], p=2, dim=-1)
                p_emb = F.normalize(lam * p_emb + (1 - lam) * p_emb[idx], p=2, dim=-1)

            mapped_b = mapper(b_emb)
            loss = memory_bank_loss(mapped_b, p_emb, memory_bank, mapper.logit_scale)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            if epoch < swa_start:
                scheduler.step()

            # Actualizar banco FIFO con los embeddings de producto del batch actual
            bs = p_emb.size(0)
            with torch.no_grad():
                end = ptr + bs
                if end <= mem_size:
                    memory_bank[ptr:end] = p_emb.detach()
                else:
                    overflow = end - mem_size
                    memory_bank[ptr:] = p_emb[:-overflow].detach()
                    memory_bank[:overflow] = p_emb[-overflow:].detach()
                ptr = end % mem_size

            total_loss += loss.item()
            batches += 1

        if epoch >= swa_start:
            swa_model.update_parameters(mapper)

        if epoch % 50 == 0:
            temp = mapper.logit_scale.exp().item()
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/batches:.4f} | "
                  f"Temp: {1/temp:.4f} | LR: {lr_now:.2e}")

    torch.optim.swa_utils.update_bn(loader, swa_model)
    return swa_model.module

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

    print("\nTraining XBM Domain Mapper (XBM + Label Smoothing + MixUp + SWA)...")
    mapper_model = train_xbm_mapper(X_train_bundles, Y_train_products, epochs=epochs)
    
    torch.save(mapper_model.state_dict(), "domain_mapper_xbm.pt")
    print("Saved domain_mapper_xbm.pt!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    else:
        epochs = 600
    main(epochs)
