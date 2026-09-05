# Clasificación de actividades REHAB con KNN desde cero

Este proyecto implementa manualmente el algoritmo K-Nearest Neighbors (KNN)
para clasificar las 16 actividades del dataset REHAB utilizando las señales
correspondientes al Sensor 2.

El objetivo principal es implementar una técnica clásica de Machine Learning
sin utilizar frameworks que proporcionen el algoritmo previamente implementado.

## Dataset utilizado

Se utilizó la sección:

`Rehab_exercise/d01_raw_data`

del dataset REHAB.

El conjunto contiene 16 actividades identificadas de `000` a `015`.
Para este proyecto se utilizaron únicamente los archivos correspondientes
al Sensor 2:

`000_2.npy` hasta `015_2.npy`.

Cada repetición contiene:

- 880 observaciones temporales
- 6 canales: f1, f2, f3, f4, f5 y pitch3

Inicialmente se analizaron 4,616 repeticiones.

## Limpieza de datos

Durante el preprocesamiento se identificaron:

- 68 repeticiones completamente formadas por ceros
- 658 duplicados exactos adicionales después de retirar las señales en cero

Después de la limpieza se obtuvo un conjunto final de:

- 3,890 muestras
- 16 actividades

## Extracción de características

Cada repetición original contiene:

`880 x 6 = 5,280 valores`

Para crear un dataset más apropiado para un algoritmo clásico de Machine
Learning se extrajeron cinco características estadísticas de cada uno de
los seis canales:

- Media
- Desviación estándar
- Mínimo
- Máximo
- Rango

Por lo tanto:

`5 características x 6 canales = 30 características`

El nuevo dataset generado se encuentra en:

`data/rehab_features.csv`

y contiene:

- 3,890 filas
- 30 características
- 1 etiqueta de actividad
- 31 columnas en total

## Implementación de KNN

El algoritmo K-Nearest Neighbors fue implementado manualmente en Python.

No se utilizaron implementaciones de Machine Learning como:

- `KNeighborsClassifier`
- `train_test_split`
- `MinMaxScaler`
- `accuracy_score`
- `confusion_matrix`

El programa implementa manualmente:

- División estratificada de los datos
- Normalización Min-Max
- Distancia euclidiana
- Selección de vecinos
- Votación de clases
- Selección del valor de K
- Cálculo de accuracy
- Matriz de confusión

NumPy se utiliza únicamente para operaciones matemáticas y manejo de arreglos.

## División de los datos

El dataset se divide aproximadamente en:

- 70% entrenamiento
- 15% validación
- 15% prueba

Se utiliza una semilla fija de `42` para permitir reproducir los resultados.

## Selección de K

Se probaron los valores:

| K | Accuracy de validación |
|---|---:|
| 1 | 92.20% |
| 3 | 81.28% |
| 5 | 69.50% |
| 7 | 66.38% |
| 9 | 62.05% |

El mejor resultado se obtuvo con:

`K = 1`

## Resultado final

Después de seleccionar K, los conjuntos de entrenamiento y validación se
combinaron para realizar la evaluación final.

Resultados:

- Entrenamiento final: 3,292 muestras
- Prueba: 598 muestras
- Predicciones correctas: 566
- Predicciones incorrectas: 32
- Accuracy global: 94.65%

## Archivos principales

### `crear_dataset.py`

Carga los archivos originales del Sensor 2, realiza la limpieza de datos,
extrae las 30 características estadísticas y genera:

`data/rehab_features.csv`

### `knn.py`

Contiene la implementación manual de KNN, la selección del valor de K y
la evaluación final del modelo.

### `generar_graficas.py`

Genera las gráficas utilizadas para analizar y presentar los resultados.

## Instalación

Se recomienda utilizar Python 3.

Instalar las dependencias con:

```bash
pip install -r requirements.txt