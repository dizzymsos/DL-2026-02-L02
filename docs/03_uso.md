# 3. Uso

## Entorno

El entorno Conda `lab_pytorch` es el mismo del Laboratorio 01.

```bash
conda activate lab_pytorch
```

Si hay que recrearlo en otro equipo:

```bash
conda env create -f environment.yml
conda activate lab_pytorch
```

En VS Code: `Ctrl + Shift + P` → `Python: Select Interpreter` → `lab_pytorch`.

## Datos

Copiar el archivo del laboratorio a `dataset/` (no se versiona):

```bash
cp "/ruta/15 atributos R0-R5.sav" dataset/
```

Todos los comandos se ejecutan desde la raiz del repositorio.

## Argumentos

### Datos y experimentos

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--data-path` | (obligatorio) | Archivo `.csv` o `.sav` |
| `--target-name` | `GDS_R2` | Objetivo de un experimento suelto |
| `--all-targets` | off | Ejecuta `GDS`, `GDS_R1` ... `GDS_R5` |
| `--algorithm` | `conv1d_softmax` | `conv1d_softmax`, `conv1d_coral` o `all` |

### Modelo (configuracion base)

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--conv-channels` | `16 32` | Canales de cada bloque Conv1D (uno o mas enteros) |
| `--kernel-size` | `3` | Tamaño del kernel (impar) |
| `--hidden-dim` | `32` | Neuronas de la capa oculta del MLP |
| `--dropout` | `0.15` | Dropout del MLP |

### Entrenamiento

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--learning-rate` | `1e-3` | Adam |
| `--weight-decay` | `1e-4` | Adam |
| `--batch-size` | `32` | Tamaño de batch |
| `--epochs` | `20` | Epocas por cada entrenamiento |
| `--use-weights` | off | Pesos por numero efectivo (solo `conv1d_coral`) |
| `--beta` | `0.99` | Beta del numero efectivo |

### Validacion

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--outer-folds` | `5` | Folds externos (reporte) |
| `--inner-folds` | `3` | Folds internos (seleccion) |
| `--gds-outer-folds` | `2` | Folds externos de `GDS` con `--all-targets` |
| `--gds-inner-folds` | `2` | Folds internos de `GDS` con `--all-targets` |
| `--no-grid` | off | No recorre el grid: usa solo la configuracion base |

### Ejecucion y reportes

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--seed` | `42` | Semilla |
| `--device` | `cpu` | `cpu`, `cuda` o `auto` |
| `--output-dir` | `results` | Carpeta de CSV, Markdown y PNG |
| `--rank-metric` | `f1_macro` | Metrica del ranking generico |
| `--rank-mode` | `auto` | `auto` minimiza MAE y errores graves; maximiza el resto |

### Grid frente a configuracion fija

Con el grid activo (por defecto), cada entrada de `HYPERPARAMETER_GRID` en
`src/config.py` **sobrescribe** la configuracion base. Los flags que el grid
no toca (por ejemplo `--learning-rate`) siguen aplicando. Con `--no-grid` se
entrena solo la configuracion de la linea de comandos; el loop interno igual
se ejecuta para informar el MAE interno.

## Ejemplos de comandos

```bash
# 1. Verificar el entorno (rapido, sin grid)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --epochs 5 --no-grid

# 2. Experimento completo en GDS_R2 (grid + 5x3)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS_R2

# 3. GDS suelto: hay que bajar los folds
python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS \
  --outer-folds 2 --inner-folds 2

# 4. Probar otra arquitectura fija (tres bloques, kernel 5)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --no-grid \
  --conv-channels 8 16 32 --kernel-size 5 --hidden-dim 64 --dropout 0.3

# 5. Los seis objetivos con reportes
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets

# 6. Ranking por MAE
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets \
  --rank-metric mae_ordinal --rank-mode min

# 7. Experimento ordinal
python main.py --data-path "dataset/15 atributos R0-R5.sav" --algorithm conv1d_coral

# 8. Ordinal con pesos por numero efectivo
python main.py --data-path "dataset/15 atributos R0-R5.sav" --algorithm conv1d_coral \
  --use-weights --beta 0.99

# 9. Comparacion completa Softmax vs CORAL en los seis objetivos
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --algorithm all
```

## Que se ve en consola

Por cada experimento:

- una linea por fold externo con la configuracion elegida, el MAE interno y
  el F1 / MAE en el test externo,
- media ± std externa de todas las metricas,
- configuracion elegida con mas frecuencia,
- reporte por clase y matriz de confusion del ultimo fold externo.

Al final: rankings por F1, por MAE y por `--rank-metric`, y la lista de
archivos generados.

## Salidas en `results/`

| Archivo | Contenido |
|---|---|
| `resultados.csv` / `.md` | Tabla algoritmo x objetivo x metricas |
| `seleccion_hp.csv` | Configuracion elegida en cada fold externo |
| `ranking_f1.*`, `ranking_mae.*`, `ranking_<metrica>.*` | Rankings |
| `barras_f1.png`, `barras_mae.png` | Barras por objetivo y algoritmo |
| `confusion_<algoritmo>_<objetivo>.png` | Matriz del ultimo fold externo |

## Tiempos de referencia

En CPU, un objetivo con grid (4 configuraciones x 3 folds internos + 1
reentrenamiento, por 5 folds externos, 20 epocas) toma alrededor de 1 minuto
por algoritmo. `--all-targets` con un algoritmo toma del orden de 5–8 minutos.
Para depurar use `--epochs 5 --no-grid`.
