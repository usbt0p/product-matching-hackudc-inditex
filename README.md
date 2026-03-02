# Inditex Fashion Retrieval - HackUDC 2026 - Repositorio ganador

> **Zara / Inditex challenge**: Develop a solution that, starting from an image of a model, identifies the items they are wearing (e.g., dress, heels, necklace, or handbag) and, for each one, returns the corresponding product reference from a predefined catalog. See details on [docs/RETO.md](docs/RETO.md), or [on the HackUDC 2026 challenge website](https://live.hackudc.gpul.org/challenges/).

---

> [!TIP] 
> Este es el repositorio ganador del Hackathon Inditex Fashion Retrieval en el HackUDC 2026. Participé como _soloVSsquad_ y obtive una puntuación final del 71.05% (calculada como el porcentaje de lineas correctas de la submission sobre sobre la ground truth).

<img src="imgs/results.png" alt="Results of the Hackathon" width="550">

---

| | |
|:---:|:---:|
| ![Image 1](imgs/1.png) | ![Image 2](imgs/2.png) |
| ![Image 3](imgs/3.png) | ![Image 4](imgs/4.png) |


> [!WARNING]
> **Este repo fue vibecodeado en un hackathon de madrudada con 3 horas de sueño en un lapso de 36 horas.** El código funciona, pero no está organizado para producción. Y no es bonito. De hecho, es feo. 
>
>El historial de git es cuestionable ya que tuve que reiniciarlo a partir de un zip de una version anterior,la estructura y abstracción de código es cuestionable y casi todo está estructurado en scripts (con pocas clases y bastante duplicación de código), y hay muchas cuestionables tomadas a las 5 de la madrugada. Estás avisado. Lo refinaré cuando pueda, pero funciona para el objetivo que tiene, siguiendo las instrucciones [del setup](#setup)

---

## ¿Qué hace esto?

Dada una foto de "bundle" (outfit, modelo en la calle, campaña de ropa), localizamos cada prenda y buscamos los 15 productos más parecidos del catálogo de Inditex. La métrica es **Recall@15**.

El pipeline tiene tres etapas:

```
Bundle foto → Detección de prendas → Embeddings → Búsqueda en catálogo → Top-15
```

Decidí tratar el problema como uno de búsqueda / recuperación de información, y no centrarme tanto en finetuning de modelos, etc. Básicamente, al hacer similaridad entre embeddings de query y catálogo, el problema se reduce a una búsqueda de vecinos cercanos, pero intentando refinar al máximo la respuesta.

Es importante denotar que se evaluarán hasta un máximo de 15 productos asociados a cada bundle (las 15 primeras filas), por lo que no tiene sentido devolver más de 15 productos, y nos interesa repartir adecuadamente los 15 productos entre las diferentes prendas del bundle.

---

## Técnicas implementadas

### 🔍 Detección de prendas
- **YOLOv8-Clothing** como detector principal de crops, estos actuan como las queries de búsqueda para el catálogo de prendas
- **Slot Filling Router** (`compare_models.py`): combina tres modelos (Grounding DINO + YOLOS-Fashionpedia + YOLOv8) y los enruta a slots semánticos UPPER / LOWER / SHOES / DEFAULT usando un sistema de votación con IoU

### 🧠 Embeddings
- **GR-Lite** (backbone DINOv3 de Meta, fine-tuneado por otra peña) como extractor de features visual, mucho mejor que CLIP para ropa. Es estado del arte y obtiene muy buenos embeddings para ropa.
- **SuperDomainMapper** (`train_mapper.py`): red neuronal simple que corrige el domain gap entre fotos de campaña y fotos de producto de catálogo. Es simplemente un mapeo lineal entre embeddings de entrada y embeddings de salida, con algo de dropout y regularización. Entrenada con:
  - Temperatura aprendible (à la CLIP / SigLIP). Te quitas un hiperparámetro de encima
  - Online Hard Negative Mining — solo los negativos más difíciles del batch: se elige un porcentaje de negativos más difíciles para entrenar, ya que la red aprende a generalizar mejor con menos datos
  - Manifold MixUp sobre embeddings para regularizar (sumar embeddings de vecinos cercanos) (similar a la idea de los Variational Autoencoders, hacer que los embeddings de vecinos cercanos sean similares y que el espacio de embeddings sea "transitable")
  - Scheduler OneCycleLR con warmup automático (esto la vdd no sabia muy bien lo que era)

### 🔎 Búsqueda
- **Alpha Query Expansion (AQE / α-QE)**: antes de buscar los 15 finales, "limpiamos" el embedding de la query fusionándolo con sus vecinos más cercanos del catálogo. Arrastra la predicción al centro del cluster correcto. Es opcional.
- **Temporal Proximity Weighting**: las URLs de Inditex llevan un timestamp (`ts=`) que indica cuándo se subió la imagen. Prendas de la misma temporada tienen timestamps similares → aplicamos una campana de Gauss para bonificar productos sincrónicos y penalizar los de otras colecciones. Hay que aprovechar todos los datos!
- **Semantic Filtering** (`semantic_filtering.py`): filtrado adicional por zona corporal usando las detecciones de DINO como referencia macro. No aporta una gran ganancia pero es un guardrail útil.

### 🔬 LoRA fine-tuning
- `train_lora.py`: fine-tunea el backbone GR-Lite directamente sobre los pares bundle↔producto con LoRA + Gradient Checkpointing + Gradient Accumulation para no petar la 3090 (si peta estando en remoto estoy cocinado).

---

## Archivos principales

| Archivo | Para qué sirve |
|---|---|
| `run_slot_filling_submission_no_postprocess.py` | **El script principal.** Genera el CSV de submission final |
| `compare_models.py` | Slot Filling Router: combina DINO + YOLOS + YOLOv8 en slots semánticos |
| `train_mapper.py` | SuperDomainMapper: cierra el domain gap bundle→catálogo |
| `train_lora.py` | Fine-tuning LoRA del backbone GR-Lite |
| `semantic_filtering.py` | Filtrado por zona corporal basado en metadatos semánticos |
| `precompute_dino.py` | Precalcula cajas "macro" de Grounding DINO para los bundles de test |
| `download_images.py` | Descarga imágenes de producto desde las URLs CDN |
| `visual_debug_yolos_no_nms.py` | Debug visual del pipeline de detección |
| `unique_product_descriptions.txt` | Lista de descripciones únicas de producto para el prompt de DINO |
| `requirements.txt` | Dependencias del proyecto |
| `LICENSE.md` | Apache 2.0 |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Necesitas un HF_TOKEN en .env para GR-Lite
echo "HF_TOKEN=hf_..." > .env
```

```bash
# Precomputar embeddings del catálogo (una vez)
python run_gr_lite.py

# Precomputar cajas DINO para test (una vez)
python precompute_dino.py

# Entrenar el Domain Mapper (opcional, mejora resultados)
python train_mapper.py

# Generar submission
python run_slot_filling_submission_no_postprocess.py

# Para visualizar predicciones de un bundle en particular
python visual_debug_yolos_no_nms.py

# Y para ver detecciones y su agregacion con el slot filling router
python compare_models.py
```

---

## Licencia

Apache 2.0 — ver [LICENSE.md](LICENSE.md). Open source, úsalo como quieras.

---

~ usbt0p :coffee:
