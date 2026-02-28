import torch
from transformers import AutoConfig, AutoModel

def main():
    print("Loading config for facebook/dinov2-large...")
    config = AutoConfig.from_pretrained('facebook/dinov2-large')
    print("Initializing model from config...")
    # Initialize without weights to be fast
    with torch.device('meta'):
        model = AutoModel.from_config(config)
    
    hf_keys = list(model.state_dict().keys())
    
    print("Loading gr_lite.pt keys...")
    d = torch.load('gr_lite.pt', map_location='cpu')
    gr_keys = list(d.keys())
    
    print("\nFormat HF Dinov2:")
    print(hf_keys[:5])
    
    print("\nFormat GR-Lite:")
    print(gr_keys[:5])
    
    # Strip prefix 'model.model.' or 'model.' if applicable
    matched = 0
    missing = []
    
    stripped_gr_keys = []
    for k in gr_keys:
        if k.startswith('model.model.'):
            stripped_gr_keys.append(k.replace('model.model.', '', 1))
        elif k.startswith('model.'):
            stripped_gr_keys.append(k.replace('model.', '', 1))
        else:
            stripped_gr_keys.append(k)
            
    gr_keys_set = set(stripped_gr_keys)
    hf_keys_set = set(hf_keys)
    
    print("\nMissing in GR-Lite:")
    print(list(hf_keys_set - gr_keys_set)[:10])
    
    print("\nExtra in GR-Lite:")
    print(list(gr_keys_set - hf_keys_set)[:10])
    
    print(f"\nIntersection: {len(hf_keys_set.intersection(gr_keys_set))} keys match.")

if __name__ == "__main__":
    main()
