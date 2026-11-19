# 2. Arquitectura

## Convolucion 1D en breve

PyTorch implementa la **correlacion cruzada** (no invierte el kernel). Para
una entrada con `C_in` canales y largo `L`, el canal de salida `j` en la
posicion `i` es:

```text
y[j, i] = b[j] + sum_{c=0}^{C_in-1} sum_{m=0}^{k-1} w[j, c, m] * x[c, s*i + d*m - p]
```

- `k`: tamaño del kernel, `s`: stride, `p`: padding, `d`: dilation.
- Largo de salida:

```text
L_out = floor((L + 2p - d(k - 1) - 1) / s) + 1
```

- Parametros de una capa: `C_out * (C_in * k + 1)`.

En este laboratorio `s = 1`, `d = 1` y `p = k // 2` con `k` impar, por lo que
`L_out = L = 15` en todas las capas convolucionales.

## Arquitectura base

```text
x                      (B, 15)        fila tabular
unsqueeze(1)           (B, 1, 15)     1 canal, largo 15
Conv1DBlock 1 -> 16    (B, 16, 15)    Conv1d(k=3, p=1) + BatchNorm1d + ReLU
Conv1DBlock 16 -> 32   (B, 32, 15)    Conv1d(k=3, p=1) + BatchNorm1d + ReLU
Flatten                (B, 480)       32 * 15
MLPBlock               (B, 32)        Linear + ReLU + Dropout
----------------------------------------------------------------
Cabeza Softmax         (B, K)         Linear(32, K)       -> CrossEntropyLoss
Cabeza CORAL   (B, K-1)       CoralLayer(32, K)   -> BCE en umbrales
```

Parametros con la configuracion por defecto y K = 3:

| Capa | Calculo | Parametros |
|---|---|---|
| Conv1d 1→16, k=3 | 16·(1·3+1) | 64 |
| BatchNorm1d(16) | 2·16 | 32 |
| Conv1d 16→32, k=3 | 32·(16·3+1) | 1568 |
| BatchNorm1d(32) | 2·32 | 64 |
| Linear 480→32 | 480·32+32 | 15392 |
| Linear 32→3 | 32·3+3 | 99 |
| **Total** | | **17219** |

### Por que `Flatten` y no pooling global

Un `AdaptiveAvgPool1d(1)` descartaria la posicion: el modelo sabria que un
filtro se activo, pero no si fue en los items temporales o en los toponimos.
Con `Flatten` el MLP recibe la respuesta de cada filtro en cada posicion.

### Por que BatchNorm1d

`BatchNorm1d(C)` normaliza cada canal usando la media y varianza del batch
(sobre las dimensiones batch y largo). Estabiliza el entrenamiento cuando las
entradas son binarias y poco variables. En `eval()` usa estadisticas acumuladas.

## Clases y responsabilidades

```text
main.py ──> ExperimentRunner ──> NestedCrossValidator ──> GridSearch ──> Trainer
                 │                       │                                 │
                 │                       └──> MetricsCalculator            ├──> ModelFactory ──> Conv1DSoftmaxNet / Conv1DCoralNet
                 └──> ReportWriter                                         └──> TaskFactory  ──> SoftmaxTask / CoralTask
```

| Modulo | Clase / funcion | Responsabilidad |
|---|---|---|
| `src/config.py` | `ModelConfig`, `TrainingConfig`, `ValidationConfig`, `ExperimentConfig` | Configuracion inmutable y validada |
| `src/config.py` | `HYPERPARAMETER_GRID`, `apply_overrides` | Grid por algoritmo y aplicacion sobre la config base |
| `src/data/loader.py` | `DataFrameLoader` | Leer CSV/SAV y extraer X `(N, 15)` en el orden de `FEATURE_COLUMNS` |
| `src/data/preprocessing.py` | `TargetEncoder` | Recodificar el objetivo a `0 .. K-1` |
| `src/data/preprocessing.py` | `StratifiedSplitter` | Folds estratificados y deteccion de clases raras |
| `src/data/dataset.py` | `CognitiveSequenceDataset` | Entregar `(1, 15)` float32 y etiqueta int64 |
| `src/models/blocks.py` | `Conv1DBlock`, `Conv1DFeatureExtractor`, `MLPBlock`, `Conv1DBackbone` | Bloques reutilizables del tronco comun |
| `src/models/conv_softmax.py` | `Conv1DSoftmaxNet` | Experimento 1 (K logits) |
| `src/models/coral.py` | `CoralLayer`, `Conv1DCoralNet` | Experimento 2 (K-1 logits) |
| `src/models/factory.py` | `ModelFactory` | Crear el modelo segun el algoritmo |
| `src/tasks/base.py` | `ClassificationTask` | Interfaz: `build_criterion()` y `predict()` |
| `src/tasks/softmax_task.py` | `SoftmaxTask` | `CrossEntropyLoss` + `argmax` |
| `src/tasks/ordinal_task.py` | `CoralTask` + funciones ordinales | BCE en umbrales + conteo |
| `src/tasks/factory.py` | `TaskFactory` | Crear la tarea segun el algoritmo |
| `src/training/trainer.py` | `Trainer` | Ciclo de entrenamiento, prediccion y evaluacion |
| `src/training/search.py` | `GridSearch` | Busqueda interna (MAE, empate QWK) |
| `src/training/nested_cv.py` | `NestedCrossValidator` | Loop externo: seleccionar, reentrenar, evaluar |
| `src/evaluation/metrics.py` | `MetricsCalculator` | Metricas multiclase y ordinales |
| `src/reporting/reports.py` | `ReportWriter`, `Ranking`, `Plotter` | CSV, Markdown y PNG |
| `src/experiment.py` | `ExperimentRunner` | Recorrer algoritmos x objetivos y reportar |

## Patrones de diseño usados

- **Strategy** (`ClassificationTask`): la perdida y la regla de prediccion se
  inyectan en el `Trainer`. Agregar CORAL no cambia el ciclo de entrenamiento.
- **Factory / registro** (`ModelFactory`, `TaskFactory`): cada algoritmo se
  asocia a un modelo y una tarea por nombre. Se puede registrar otra
  implementacion sin tocar el resto del codigo.
- **Configuracion inmutable** (`dataclass(frozen=True)`): el grid crea nuevas
  configuraciones con `dataclasses.replace`, sin efectos laterales.

## Validacion anidada

```text
Para cada fold externo (5):
    train_ext, test_ext
    GridSearch sobre train_ext:
        Para cada configuracion del grid:
            Para cada fold interno (3): entrenar y medir MAE, QWK
        Elegir menor MAE promedio (empate: mayor QWK)
    Reentrenar la configuracion elegida con todo train_ext
    Evaluar una sola vez en test_ext
Reportar media ± std de las metricas externas
```
