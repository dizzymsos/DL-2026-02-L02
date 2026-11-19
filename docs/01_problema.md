# 1. El problema

## Contexto clinico

Se desea predecir la etapa de deterioro cognitivo de una persona a partir de
15 respuestas binarias (correcta / incorrecta) de un test de orientacion. El
laboratorio **no** es un diagnostico clinico: el objetivo es formular, entrenar
y evaluar modelos de forma rigurosa.

## La escala GDS

La Global Deterioration Scale (Reisberg) ordena el deterioro en siete etapas.
El archivo tiene 1119 casos:

| Etapa | Descripcion | n | % |
|---|---|---|---|
| 1 | Sin deterioro cognitivo | 149 | 13,3 |
| 2 | Deterioro muy leve | 500 | 44,7 |
| 3 | Deterioro leve | 298 | 26,6 |
| 4 | Deterioro moderado | 108 | 9,7 |
| 5 | Deterioro moderadamente grave | 42 | 3,8 |
| 6 | Deterioro grave | 20 | 1,8 |
| 7 | Deterioro muy grave | 2 | 0,2 |

Las etapas estan **ordenadas**: confundir la etapa 2 con la 3 no es lo mismo
que confundir la 2 con la 7. Accuracy trata ambos errores igual; MAE, QWK y
errores graves no.

## Seis experimentos independientes

Cada columna objetivo es un experimento distinto (un modelo por columna):

| Columna | Agrupacion desde GDS | K | n por clase |
|---|---|---|---|
| `GDS` | 1, 2, 3, 4, 5, 6, 7 | 7 | 149 / 500 / 298 / 108 / 42 / 20 / 2 |
| `GDS_R1` | {1,2,3}→1; {4,5}→2; {6,7}→3 | 3 | 947 / 150 / 22 |
| `GDS_R2` | {1,2}→1; {3}→2; {4–7}→3 | 3 | 649 / 298 / 172 |
| `GDS_R3` | {1,2,3}→1; {4–7}→3 | 2 | 947 / 172 |
| `GDS_R4` | {1}→1; {2,3,4}→2; {5,6,7}→3 | 3 | 149 / 906 / 64 |
| `GDS_R5` | {1}→1; {2,3}→2; {4–7}→3 | 3 | 149 / 798 / 172 |

Cada columna se recodifica a indices `0 .. K-1` conservando el orden. El
experimento por defecto es `GDS_R2`.

**Advertencia sobre `GDS`:** la etapa 7 tiene solo 2 muestras, por lo que
`StratifiedKFold` con 5 folds falla. Con `--all-targets`, `GDS` usa
`--gds-outer-folds` y `--gds-inner-folds` (2 por defecto). En un experimento
suelto hay que bajar `--outer-folds` / `--inner-folds` a mano.

## Las 15 entradas

| Grupo | Atributos (binarios) | Posiciones |
|---|---|---|
| Orientacion temporal | Dia, Mes, Año, Estacion | 0–3 |
| Orientacion espacial | Pais, Ciudad, CalleLugar, NumeroPiso | 4–7 |
| Toponimos locales | Miguel2, Gonzalez2, Avenida2, Imperial2, A682, Caldera2, Copiapo2 | 8–14 |

## Por que una capa Conv1D

En el Laboratorio 01 una MLP veia los 15 atributos como un vector sin
estructura. En este laboratorio la fila se interpreta como una **secuencia**
de largo 15 y 1 canal, con forma `(1, 15)`:

- Un kernel de tamaño `k` mira `k` items consecutivos a la vez y aprende
  patrones locales (por ejemplo, "falla Dia y Mes juntos").
- Los mismos pesos se comparten a lo largo de la secuencia (**weight
  sharing**): un patron aprendido en una posicion se detecta en cualquier otra.
- Con varios filtros (canales) se aprenden varios patrones en paralelo.
- Despues de las convoluciones se aplana (`Flatten`) para que el MLP conozca
  **en que posicion** se activo cada filtro.

**Importante:** a diferencia de una MLP, la Conv1D depende del **orden de las
columnas**. El orden de `FEATURE_COLUMNS` en `src/config.py` agrupa items
relacionados (temporal, espacial, toponimos), de modo que las ventanas locales
tengan sentido. Una pregunta del informe es que pasa si se permutan las
columnas.

## Formulacion de la salida

- **Multiclase categorica (`conv1d_softmax`)**: K logits, `CrossEntropyLoss`,
  prediccion por `argmax`. Ignora el orden de las clases.
- **Ordinal (`conv1d_coral`)**: K-1 logits acumulativos `P(Y > k)` con
  umbrales ordenados, BCE sobre los umbrales y prediccion por conteo. Usa el
  orden de las clases.

## Validacion y metricas

- Validacion cruzada anidada: 5 folds externos (reporte) x 3 internos
  (seleccion de hiperparametros), estratificada por clase.
- Seleccion interna por menor MAE (empate: mayor QWK). Nunca por accuracy.
- Se reporta solo media ± std de los folds **externos**.
- Metricas multiclase: accuracy, balanced accuracy, precision/recall/F1 macro.
- Metricas ordinales: MAE, QWK, accuracy ±1, errores graves (`|y - ŷ| >= 2`).
