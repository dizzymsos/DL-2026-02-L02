# Laboratorio 02 — Redes convolucionales 1D (Deep Learning 2026-02)

Clasificación multiclase y ordinal de la etapa de deterioro cognitivo (escala GDS)
a partir de 15 respuestas binarias de un test de orientación, usando una red
**Conv1D + MLP** en PyTorch.

- **Curso:** Deep Learning, Universidad Católica del Norte
- **Profesor:** Dr. Juan Bekios Calfa
- **Integrantes:** Alejandra León, Michelle Olivares

Este repositorio parte del código base entregado por el profesor
([jbekios/DL-2026-02-L02](https://github.com/jbekios/DL-2026-02-L02)). Sobre él se
implementó la cabeza ordinal **CORAL** y se ejecutaron los experimentos del
informe.

## Experimentos

| Algoritmo | Arquitectura | Pérdida | Predicción |
|---|---|---|---|
| `conv1d_softmax` | Conv1D + MLP + `Linear(K)` | `CrossEntropyLoss` | `argmax` |
| `conv1d_coral` | Conv1D + MLP + `CoralLayer(K-1)` | BCE sobre umbrales | conteo de umbrales |
| `conv1d_coral` + `--use-weights` | igual que CORAL | BCE ponderada por número efectivo de muestras | conteo de umbrales |

Las tres variantes comparten el mismo tronco (`Conv1DBackbone`) y se evalúan en
los seis objetivos (`GDS`, `GDS_R1` ... `GDS_R5`) con validación cruzada anidada
5 × 3 y búsqueda de hiperparámetros por MAE interno.

## Qué implementamos

Todo está en dos archivos; el resto del código base no se modificó.

| Archivo | Pieza | Qué hace |
|---|---|---|
| `src/tasks/ordinal_task.py` | `labels_to_levels` | Codifica la clase `y` como `K-1` umbrales binarios `1[y > k]` |
| `src/models/coral.py` | `CoralLayer` | Puntaje compartido `wᵀh` + `K-1` sesgos ordenados (`softplus` + `cumsum`) |
| `src/models/coral.py` | `Conv1DCoralNet` | `CoralLayer` sobre el tronco Conv1D |
| `src/tasks/ordinal_task.py` | `coral_loss` | BCE con logits por umbral; con pesos, pondera cada muestra por su clase real |
| `src/tasks/ordinal_task.py` | `logits_to_ordinal_predictions` | Cuenta los umbrales con `σ(z_k) > 0.5` |
| `src/tasks/ordinal_task.py` | `effective_number_weights` | `w_c = (1-β)/(1-β^n_c)`, normalizados a media 1; peso 0 si `n_c = 0` |

**Ajuste de los sesgos (`BIAS_SCALE = 10`).** Con 20 épocas y `lr = 0.001`, Adam
no alcanzaba a separar los umbrales de CORAL y el modelo predecía de más la
clase 0 (en `GDS_R4`, accuracy 0.69 frente a 0.81 de la clase mayoritaria). Los
sesgos se reparametrizaron con un factor de escala en `coral.py`, sin tocar el
`Trainer` ni la arquitectura. El orden de los umbrales se mantiene.

## Cómo reproducir

```bash
conda env create -f environment.yml   # solo la primera vez
conda activate lab_pytorch

# Copiar el dataset (no se versiona)
cp "/ruta/al/archivo/15 atributos R0-R5.sav" dataset/

# Corrida 1: Softmax y CORAL en los seis objetivos -> results/
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --algorithm all

# Corrida 2: CORAL con pesos en los seis objetivos -> results/coral_pesos/
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets \
  --algorithm conv1d_coral --use-weights --output-dir results/coral_pesos
```

Las semillas están fijas, así que repetir las corridas entrega las mismas
métricas. Cada corrida escribe tablas, rankings, gráficos, matrices de confusión
y la configuración elegida en cada fold (`seleccion_hp.csv`).

> **Nota sobre `GDS`:** la etapa 7 tiene solo 2 casos, por lo que `GDS` usa 2
> folds externos y no admite folds internos estratificados: se omite la búsqueda
> y se usa la configuración base.

## Resultados principales

MAE ordinal (media ± desviación estándar de los folds externos, menor es mejor):

| Objetivo | Softmax | CORAL | CORAL + pesos |
|---|---|---|---|
| GDS | **0.474 ± 0.015** | 0.493 ± 0.029 | 0.621 ± 0.028 |
| GDS_R1 | **0.104 ± 0.022** | 0.107 ± 0.017 | 0.121 ± 0.027 |
| GDS_R2 | 0.291 ± 0.041 | **0.278 ± 0.033** | 0.285 ± 0.039 |
| GDS_R3 | **0.097 ± 0.021** | **0.097 ± 0.022** | 0.099 ± 0.016 |
| GDS_R4 | 0.170 ± 0.009 | **0.166 ± 0.010** | 0.183 ± 0.010 |
| GDS_R5 | **0.225 ± 0.007** | 0.231 ± 0.010 | 0.237 ± 0.014 |

- Ninguna variante domina: las diferencias entre Softmax y CORAL son menores que
  la desviación entre folds.
- Los pesos suben el balanced accuracy en los seis objetivos a costa de accuracy
  y MAE.
- La búsqueda elige la red más pequeña (canales 8-16, `k = 3`) en 48 de 75 folds.

Las tablas completas y el análisis están en el informe.

## Estructura

```text
DL-2026-02-L02/
|-- main.py                  # CLI: arma ExperimentConfig y llama a ExperimentRunner
|-- src/
|   |-- config.py            # constantes, configuraciones y HYPERPARAMETER_GRID
|   |-- utils.py             # semillas y dispositivo
|   |-- experiment.py        # ExperimentRunner: algoritmos x objetivos
|   |-- data/                # carga, codificación de etiquetas, folds y Dataset (1, 15)
|   |-- models/              # Conv1DBackbone, Conv1DSoftmaxNet, coral.py (implementado), ModelFactory
|   |-- tasks/               # SoftmaxTask, ordinal_task.py (implementado), TaskFactory
|   |-- training/            # Trainer, GridSearch, NestedCrossValidator
|   |-- evaluation/          # métricas
|   `-- reporting/           # tablas, rankings y gráficos
|-- docs/                    # documentación del código base
|-- dataset/                 # datos locales (no versionados)
|-- results/                 # salidas de main.py (no versionadas)
|-- environment.yml
`-- README.md
```

## Documentación del código base

1. [El problema](docs/01_problema.md): GDS, reagrupaciones y desbalance.
2. [Arquitectura](docs/02_arquitectura.md): convolución 1D, formas y clases.
3. [Uso](docs/03_uso.md): entorno, argumentos y salidas.
4. [Extensión ordinal](docs/04_extension_ordinal.md): especificación de CORAL.