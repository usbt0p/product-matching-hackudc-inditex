
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_popularity_prior(catalog_ids, train_csv):
    """Calcula un prior de frecuencia normalizado para cada producto del catálogo."""
    df_train = pd.read_csv(train_csv)
    counts = df_train['product_asset_id'].value_counts()
    
    # Mapear conteos al array de ids, default 0
    priors = np.array([counts.get(pid, 0) for pid in catalog_ids], dtype=np.float32)
    
    # Normalizar entre 0 y 0.05 (bonus máximo del 5% a la similitud coseno)
    if priors.max() > 0:
        priors = (priors / priors.max()) * 0.05
    return priors

# Uso en tu bucle (sumar directamente a las similitudes de GR-Lite):
# sims = np.dot(normalized_catalog, normalized_emb).flatten()
# sims += popularity_priors

def plot_popularity(train_csv_path, save_path="real_popularity_plot.png", gettopk=10):
    """Grafica la frecuencia de productos y la guarda."""
    df_train = pd.read_csv(train_csv_path)
    counts = df_train['product_asset_id'].value_counts()

    # get top k products
    topk_products = counts.head(gettopk)
    # associate top k products with their urls
    df_products = pd.read_csv('data_csvs/product_dataset.csv')
    topk_products_urls = df_products[df_products['product_asset_id'].isin(topk_products.index)]['product_image_url'].tolist()
    
    plt.figure(figsize=(12, 6))
    counts.plot(kind='line', linewidth=2, color='darkred')
    plt.fill_between(range(len(counts)), counts.values, color='darkred', alpha=0.3)
    plt.title('Distribución de Productos (Best Sellers)')
    plt.xlabel('Productos (Ordenados)')
    plt.ylabel('Apariciones')
    plt.xticks([]) 
    plt.savefig(save_path)
    plt.close()

    return topk_products_urls

# plot_popularity('data_csvs/bundles_product_match_train.csv')

if __name__ == "__main__":
    p = plot_popularity('data_csvs/bundles_product_match_train.csv')
    for url in p:
        print(url)