# 4. Extension ordinal (implementada) > Implementado en src/models/coral.py y src/tasks/ordinal_task.py. Los sesgos de CoralLayer usan BIAS_SCALE = 10 (ver README).

El experimento `conv1d_coral` reutiliza todo lo entregado (datos, tronco
Conv1D, `Trainer`, grid, validacion anidada, metricas y reportes). Solo faltan
la cabeza CORAL y las funciones ordinales.

## Idea de CORAL

- Softmax aprende K direcciones independientes y no usa el orden.
- CORAL aprende un **puntaje latente** `s(x) = wᵀh` y K-1 **umbrales
  ordenados** `b_0 >= b_1 >= ... >= b_{K-2}`.
- Logits: `z_k = s(x) + b_k`, con `σ(z_k) ≈ P(Y > k)`.
- Una etiqueta `y = 2` con K = 5 se codifica `[1, 1, 0, 0]`.
- Prediccion: `ŷ = Σ_k 1[σ(z_k) > 0.5]`.
- Como todos los umbrales comparten `w`, las probabilidades quedan ordenadas
  (consistencia de rango).

## Esqueletos a completar

| Orden | Archivo | Firma | Formas |
|---|---|---|---|
| 1 | `src/tasks/ordinal_task.py` | `labels_to_levels(labels, num_classes)` | `(B,) -> (B, K-1)` |
| 2 | `src/models/coral.py` | `CoralLayer(input_size, num_classes)` | `(B, H) -> (B, K-1)` |
| 3 | `src/models/coral.py` | `Conv1DCoralNet(...)` | `(B, 15) -> (B, K-1)` |
| 4 | `src/tasks/ordinal_task.py` | `coral_loss(logits, labels, num_classes, class_weights=None)` | escalar |
| 5 | `src/tasks/ordinal_task.py` | `logits_to_ordinal_predictions(logits, threshold=0.5)` | `(B, K-1) -> (B,)` |
| 6 | `src/tasks/ordinal_task.py` | `effective_number_weights(labels, num_classes, beta=0.99)` | `(N,) -> (K,)` |

`CoralTask`, `CoralLoss`, `ModelFactory` y `TaskFactory` **ya estan
conectados**: al completar las funciones, `--algorithm conv1d_coral` funciona
sin tocar `main.py` ni el `Trainer`.

`Conv1DCoralNet` ya construye el tronco `self.backbone` (igual al de Softmax).
Falta crear la cabeza CORAL sobre `self.backbone.output_dim` y el `forward`.

## Pistas

- `CoralLayer`: `nn.Linear(input_size, 1, bias=False)` para el puntaje; un
  sesgo inicial y K-2 diferencias positivas con `softplus`; `cumsum` para
  construir umbrales decrecientes.
- `coral_loss`: `F.binary_cross_entropy_with_logits(..., reduction="none")`,
  sumar sobre los K-1 umbrales y promediar en el batch. Con `class_weights`,
  multiplicar cada muestra por el peso de su clase real.
- `effective_number_weights`: `w_c = (1 - β) / (1 - β^{n_c})`, normalizar a
  media 1 y evitar divisiones por cero si `n_c = 0`.
- No aplicar sigmoide en el `forward`: la perdida trabaja con logits.
- No mezclar `CrossEntropyLoss` con la cabeza CORAL.

## Prueba rapida de formas

```python
import torch
from src.models.coral import CoralLayer, Conv1DCoralNet
from src.tasks.ordinal_task import labels_to_levels, logits_to_ordinal_predictions

print(labels_to_levels(torch.tensor([0, 2, 4]), num_classes=5))
# tensor([[0., 0., 0., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.]])

model = Conv1DCoralNet(num_features=15, num_classes=3)
logits = model(torch.rand(8, 15).round())
print(logits.shape)                                # torch.Size([8, 2])
print(logits_to_ordinal_predictions(logits).shape)  # torch.Size([8])
```

## Experimentos esperados en el informe

```bash
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --algorithm all
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets \
  --algorithm conv1d_coral --use-weights --output-dir results/coral_pesos
```

Tabla minima por objetivo (media ± std externa):

| Modelo | Acc | F1_m | BalAcc | MAE | QWK | Acc±1 | Err≥2 |
|---|---|---|---|---|---|---|---|
| Conv1D + Softmax | | | | | | | |
| Conv1D + CORAL | | | | | | | |
| Conv1D + CORAL + pesos | | | | | | | |

## Preguntas para el informe

1. ¿Que patrones locales puede aprender un kernel de tamaño 3 sobre los 15 items?
2. ¿Que pasa con la Conv1D si se permutan las columnas de entrada? ¿Y con una MLP?
3. ¿Por que se usa `Flatten` en lugar de un pooling global?
4. ¿Como garantiza CORAL la consistencia ordinal?
5. ¿En que objetivos CORAL mejora MAE / errores graves frente a Softmax?
6. ¿Que efecto tienen los pesos por numero efectivo en las clases raras?
