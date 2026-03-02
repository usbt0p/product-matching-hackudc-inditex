# Audit de archivos — ¿qué se puede borrar?

## ✅ Archivos que SÍ se usan (mantener)

| Archivo | Rol |
|---|---|
| [run_slot_filling_submission_no_postprocess.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_slot_filling_submission_no_postprocess.py) | Script principal de submission |
| [compare_models.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/compare_models.py) | Slot Filling Router (importado por el anterior) |
| [semantic_filtering.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/semantic_filtering.py) | Filtrado semántico por zona corporal |
| [train_mapper.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/train_mapper.py) | SuperDomainMapper + entrenamiento |
| [train_lora.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/train_lora.py) | LoRA fine-tuning del backbone |
| [precompute_dino.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/precompute_dino.py) | Precálculo de cajas DINO macro (genera [test_dino_macro.json](file:///home/auria/hackudc_lucas/inditex_dos/inditex/test_dino_macro.json)) |
| [download_images.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/download_images.py) | Descarga imágenes CDN (usado por train_lora) |
| [visual_debug_yolos_no_nms.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/visual_debug_yolos_no_nms.py) | Debug visual del pipeline actual |
| [plot_loss.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/plot_loss.py) | Plotea el loss del mapper |
| [unique_product_descriptions.txt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/unique_product_descriptions.txt) | Prompt de DINO |
| [requirements.txt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/requirements.txt) | Dependencias |
| [LICENSE.md](file:///home/auria/hackudc_lucas/inditex_dos/inditex/LICENSE.md), [README.md](file:///home/auria/hackudc_lucas/inditex_dos/inditex/README.md), [RETO.md](file:///home/auria/hackudc_lucas/inditex_dos/inditex/RETO.md) | Documentación |
| [evaluate_slot_filling.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/evaluate_slot_filling.py) | Evaluación local (útil para iterar) |

---

## 🗑️ Archivos candidatos a borrar

### Scripts de pipelines anteriores (superados por el actual)

| Archivo | Por qué está obsoleto |
|---|---|
| [run_slot_filling_submission.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_slot_filling_submission.py) | Versión anterior del actual (con postprocess). Reemplazado por `_no_postprocess` |
| [run_hybrid_submission.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_hybrid_submission.py) | Pipeline híbrido CLIP+GRLite. Superado por el mapper |
| [run_retrieval.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_retrieval.py) | Primer retrieval con CLIP puro. Superado |
| [run_retrieval_v2.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_retrieval_v2.py) | Segunda iteración CLIP. Superada |
| [run_cache_clip.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_cache_clip.py) | Precacheo de embeddings CLIP. Ya no se usa CLIP como modelo principal |
| [run_detector.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_detector.py) | Detector standalone, absorbido por el pipeline |
| [run_background_removal.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/run_background_removal.py) | Experimento con rembg. No en el pipeline final |
| [visualize_predictions.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/visualize_predictions.py) | Versión antigua del debug (reemplazada por [visual_debug_yolos_no_nms.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/visual_debug_yolos_no_nms.py)) |
| [evaluate_hybrid.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/evaluate_hybrid.py) | Evalúa el pipeline híbrido CLIP, ya obsoleto |
| [check_keys.py](file:///home/auria/hackudc_lucas/inditex_dos/inditex/check_keys.py) | Script de diagnóstico one-shot |

### Archivos de datos / embeddings obsoletos

| Archivo | Por qué |
|---|---|
| [catalog_clip_embeddings.npy](file:///home/auria/hackudc_lucas/inditex_dos/inditex/catalog_clip_embeddings.npy) + [valid_clip_ids.npy](file:///home/auria/hackudc_lucas/inditex_dos/inditex/valid_clip_ids.npy) | Embeddings de CLIP, ya no es el modelo principal |
| [catalog_embeddings.npy](file:///home/auria/hackudc_lucas/inditex_dos/inditex/catalog_embeddings.npy) + [valid_catalog_ids.npy](file:///home/auria/hackudc_lucas/inditex_dos/inditex/valid_catalog_ids.npy) | Embeddings de DINOv2 estándar, reemplazados por [catalog_grlite_embeddings.npy](file:///home/auria/hackudc_lucas/inditex_dos/inditex/catalog_grlite_embeddings.npy) |
| [domain_mapper_DEFAULT.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper_DEFAULT.pt), [_LOWER.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper_LOWER.pt), [_SHOES.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper_SHOES.pt), [_UPPER.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper_UPPER.pt) | Versión anterior (mappers por zona). Reemplazados por [domain_mapper_super.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper_super.pt) |
| [domain_mapper.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper.pt) | Legacy mapper (sin learnable temperature). Fallback en código, pero [domain_mapper_super.pt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/domain_mapper_super.pt) es mejor |
| `train_b_embs_DEFAULT/LOWER/SHOES/UPPER.npy` | Embeddings por zona de versión anterior |
| `train_p_embs_DEFAULT/LOWER/SHOES/UPPER.npy` | Ídem |
| [mobileclip2_b.ts](file:///home/auria/hackudc_lucas/inditex_dos/inditex/mobileclip2_b.ts) | MobileCLIP, experimento descartado (~250 MB) |
| [catalog_semantic_meta.pkl](file:///home/auria/hackudc_lucas/inditex_dos/inditex/catalog_semantic_meta.pkl) | Metadatos semánticos en formato antiguo |
| `submission_*.csv` (varios) | Submissions de experimentación, solo guardar el último bueno |

### Directorios de salida con contenido acumulado

| Directorio | Revisar |
|---|---|
| `bundle_crops/`, `bundle_crops_rembg/` | Crops generados, regenerables |
| `compare_output/` | Visualizaciones de compare_models |
| `detected_bundles/` | Outputs de run_detector |
| `visual_debug/`, `visual_debug_slot_filling/`, `visual_debug_yolos_no_nms/` | Debug visual, borrable si necesitas espacio |

### Documentos de proceso interno

| Archivo | Nota |
|---|---|
| [INVESTIGATION.md](file:///home/auria/hackudc_lucas/inditex_dos/inditex/INVESTIGATION.md) | Notas de investigación del equipo (28KB). Interesante como historial pero no es código |
| [cosas_que_le_cuestan.txt](file:///home/auria/hackudc_lucas/inditex_dos/inditex/cosas_que_le_cuestan.txt) | Notas internas informales |
| [loss.log](file:///home/auria/hackudc_lucas/inditex_dos/inditex/loss.log) / [training.log](file:///home/auria/hackudc_lucas/inditex_dos/inditex/training.log) | Logs de entrenamientos pasados |

---

## Resumen: espacio recuperable estimado

| Categoría | Espacio aprox. |
|---|---|
| [mobileclip2_b.ts](file:///home/auria/hackudc_lucas/inditex_dos/inditex/mobileclip2_b.ts) | ~250 MB |
| Embeddings CLIP obsoletos | ~165 MB |
| Embeddings DINOv2 estándar obsoletos | ~165 MB |
| Domainmappers por zona (4×16MB) | ~64 MB |
| Embeddings de entrenamiento por zona (8 archivos) | ~52 MB |
| Submissions CSV experimentales | ~1 MB |
| **Total aprox.** | **~700 MB** |
