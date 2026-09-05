import os
import csv
import random
import numpy as np


# ============================================================
# ruta del dataset
# ============================================================

ruta_base = os.path.dirname(os.path.abspath(__file__))

archivo_dataset = os.path.join(
    ruta_base,
    "data",
    "rehab_features.csv"
)


# ============================================================
# cargar el dataset
# ============================================================

def cargar_dataset(ruta):

    X = []
    y = []

    with open(ruta, "r", encoding="utf-8") as archivo:

        lector = csv.reader(archivo)

        # saltar encabezados
        next(lector)

        for fila in lector:

            caracteristicas = [
                float(valor)
                for valor in fila[:-1]
            ]

            actividad = fila[-1]

            X.append(caracteristicas)
            y.append(actividad)

    return (
        np.array(X, dtype=float),
        np.array(y)
    )


# ============================================================
# DIVISION 
# APROX. 70% TRAIN / 15% VALIDACIÓN / 15% TEST
# ============================================================

def dividir_datos(X, y, semilla=42):

    random.seed(semilla)

    indices_train = []
    indices_val = []
    indices_test = []

    actividades = sorted(set(y))

    for actividad in actividades:

        indices = [
            i
            for i in range(len(y))
            if y[i] == actividad
        ]

        random.shuffle(indices)

        total = len(indices)

        fin_train = int(
            total * 0.70
        )

        fin_val = (
            fin_train
            + int(total * 0.15)
        )

        indices_train.extend(
            indices[:fin_train]
        )

        indices_val.extend(
            indices[fin_train:fin_val]
        )

        indices_test.extend(
            indices[fin_val:]
        )

    # revolver los conjuntos finales
    random.shuffle(indices_train)
    random.shuffle(indices_val)
    random.shuffle(indices_test)

    return (
        X[indices_train],
        X[indices_val],
        X[indices_test],
        y[indices_train],
        y[indices_val],
        y[indices_test]
    )


# ============================================================
# MIN-MAX
# ============================================================

def calcular_min_max(X):

    minimos = np.min(
        X,
        axis=0
    )

    maximos = np.max(
        X,
        axis=0
    )

    return minimos, maximos


def normalizar(
    X,
    minimos,
    maximos
):

    rangos = (
        maximos - minimos
    )

    # Evitar división entre cero
    rangos[rangos == 0] = 1

    return (
        X - minimos
    ) / rangos


# ============================================================
# KNN
# ============================================================

def predecir_muestra(
    muestra,
    X_train,
    y_train,
    k
):

    # calcular distancia euclidiana
    diferencias = (
        X_train - muestra
    )

    distancias = np.sqrt(
        np.sum(
            diferencias ** 2,
            axis=1
        )
    )

    # ordenar distancias
    indices_ordenados = np.argsort(
        distancias
    )

    # seleccionar K vecinos
    vecinos = indices_ordenados[:k]

    etiquetas_vecinos = (
        y_train[vecinos]
    )

    # contar votos
    etiquetas, conteos = np.unique(
        etiquetas_vecinos,
        return_counts=True
    )

    mayor_conteo = np.max(
        conteos
    )

    ganadores = etiquetas[
        conteos == mayor_conteo
    ]

    # si existe un único ganador
    if len(ganadores) == 1:

        return ganadores[0]

    # en caso de empate,
    # seleccionar el vecino más cercano
    # entre las clases empatadas
    for indice in vecinos:

        etiqueta = y_train[indice]

        if etiqueta in ganadores:

            return etiqueta


# ============================================================
# REALIZAR PREDICCIONES
# ============================================================

def predecir(
    X_datos,
    X_train,
    y_train,
    k
):

    predicciones = []

    for muestra in X_datos:

        prediccion = predecir_muestra(
            muestra,
            X_train,
            y_train,
            k
        )

        predicciones.append(
            prediccion
        )

    return np.array(
        predicciones
    )


# ============================================================
# calcular accuracy
# ============================================================

def calcular_accuracy(
    y_real,
    y_pred
):

    correctas = 0

    for real, prediccion in zip(
        y_real,
        y_pred
    ):

        if real == prediccion:

            correctas += 1

    return (
        correctas / len(y_real)
    )


# ============================================================
# matriz de confusion
# ============================================================

def crear_matriz_confusion(
    y_real,
    y_pred,
    actividades
):

    cantidad = len(
        actividades
    )

    matriz = np.zeros(
        (cantidad, cantidad),
        dtype=int
    )

    indices = {
        actividad: i
        for i, actividad
        in enumerate(actividades)
    }

    for real, prediccion in zip(
        y_real,
        y_pred
    ):

        fila = indices[real]

        columna = indices[
            prediccion
        ]

        matriz[
            fila,
            columna
        ] += 1

    return matriz


# ============================================================
# principal
# ============================================================

def main():

    print("=" * 70)
    print("KNN DESDE CERO - DATASET REHAB")
    print("=" * 70)


    # ========================================================
    # 1. CARGAR DATASET
    # ========================================================

    X, y = cargar_dataset(
        archivo_dataset
    )

    actividades = sorted(
        set(y)
    )

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Muestras totales: "
        f"{len(X)}"
    )

    print(
        f"Características: "
        f"{X.shape[1]}"
    )

    print(
        f"Actividades: "
        f"{len(actividades)}"
    )


    # ========================================================
    # 2. DIVIDIR DATOS
    # ========================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = dividir_datos(
        X,
        y
    )


    print("\nDIVISIÓN INICIAL")
    print("-" * 70)

    print(
        f"Entrenamiento: "
        f"{len(X_train)}"
    )

    print(
        f"Validación: "
        f"{len(X_val)}"
    )

    print(
        f"Prueba: "
        f"{len(X_test)}"
    )


    # ========================================================
    # 3. NORMALIZAR PARA SELEC K
    # ========================================================

    minimos, maximos = calcular_min_max(
        X_train
    )

    X_train_norm = normalizar(
        X_train,
        minimos,
        maximos
    )

    X_val_norm = normalizar(
        X_val,
        minimos,
        maximos
    )


    # ========================================================
    # 4. PROBAR VALORES DE K
    # ========================================================

    valores_k = [
        1,
        3,
        5,
        7,
        9
    ]

    resultados_k = {}


    print("\n" + "=" * 70)
    print("SELECCIÓN DEL VALOR DE K")
    print("=" * 70)


    for k in valores_k:

        predicciones_val = predecir(
            X_val_norm,
            X_train_norm,
            y_train,
            k
        )

        accuracy_val = calcular_accuracy(
            y_val,
            predicciones_val
        )

        resultados_k[k] = (
            accuracy_val
        )

        print(
            f"K = {k}: "
            f"{accuracy_val * 100:.2f}%"
        )


    # ========================================================
    # 5. SELEC MEJOR K
    # ========================================================

    mejor_k = max(
        resultados_k,
        key=resultados_k.get
    )

    mejor_accuracy = (
        resultados_k[mejor_k]
    )


    print("\nMEJOR CONFIGURACIÓN")
    print("-" * 70)

    print(
        f"Mejor K: "
        f"{mejor_k}"
    )

    print(
        f"Accuracy de validación: "
        f"{mejor_accuracy * 100:.2f}%"
    )


    # ========================================================
    # 6. train + validacion
    # ========================================================

    X_train_final = np.concatenate(
        (
            X_train,
            X_val
        ),
        axis=0
    )

    y_train_final = np.concatenate(
        (
            y_train,
            y_val
        ),
        axis=0
    )


    print("\nENTRENAMIENTO FINAL")
    print("-" * 70)

    print(
        f"Muestras de entrenamiento final: "
        f"{len(X_train_final)}"
    )

    print(
        f"Muestras de prueba: "
        f"{len(X_test)}"
    )


    # ========================================================
    # 7. recalcular la normalizacion
    #    train
    # ========================================================

    minimos_finales, maximos_finales = calcular_min_max(
        X_train_final
    )

    X_train_final_norm = normalizar(
        X_train_final,
        minimos_finales,
        maximos_finales
    )

    X_test_norm = normalizar(
        X_test,
        minimos_finales,
        maximos_finales
    )


    # ========================================================
    # 8. FINAL
    # ========================================================

    print("\nRealizando predicciones finales...")


    predicciones_test = predecir(
        X_test_norm,
        X_train_final_norm,
        y_train_final,
        mejor_k
    )


    correctas = np.sum(
        y_test == predicciones_test
    )

    incorrectas = (
        len(y_test) - correctas
    )

    accuracy_final = (
        correctas / len(y_test)
    )


    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)

    print(
        f"K utilizado: "
        f"{mejor_k}"
    )

    print(
        f"Predicciones totales: "
        f"{len(y_test)}"
    )

    print(
        f"Predicciones correctas: "
        f"{correctas}"
    )

    print(
        f"Predicciones incorrectas: "
        f"{incorrectas}"
    )

    print(
        f"Accuracy global: "
        f"{accuracy_final * 100:.2f}%"
    )


    # ========================================================
    # 9. porcentaje de aciertos por actividad
    # ========================================================

    print("\n" + "=" * 70)
    print("PORCENTAJE DE ACIERTOS POR ACTIVIDAD")
    print("=" * 70)


    for actividad in actividades:

        posiciones = (
            y_test == actividad
        )

        reales_actividad = (
            y_test[posiciones]
        )

        pred_actividad = (
            predicciones_test[posiciones]
        )

        correctas_actividad = np.sum(
            reales_actividad
            == pred_actividad
        )

        total_actividad = len(
            reales_actividad
        )

        porcentaje = (
            correctas_actividad
            / total_actividad
        ) * 100


        print(
            f"Actividad {actividad}: "
            f"{correctas_actividad}/"
            f"{total_actividad} correctas "
            f"({porcentaje:.2f}%)"
        )


    # ========================================================
    # 10. matriz de confusion
    # ========================================================

    matriz = crear_matriz_confusion(
        y_test,
        predicciones_test,
        actividades
    )


    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSIÓN")
    print("=" * 70)

    print(
        "\nFilas = actividad real"
    )

    print(
        "Columnas = actividad predicha\n"
    )


    print(
        "Real\\Pred",
        end=" "
    )


    for actividad in actividades:

        print(
            f"{actividad:>4}",
            end=" "
        )

    print()


    for i, actividad in enumerate(
        actividades
    ):

        print(
            f"{actividad:>9}",
            end=" "
        )

        for valor in matriz[i]:

            print(
                f"{valor:>4}",
                end=" "
            )

        print()


    # ========================================================
    # 11. confusiones frecuentes
    # ========================================================

    errores = []


    for i in range(
        len(actividades)
    ):

        for j in range(
            len(actividades)
        ):

            if i != j:

                cantidad = (
                    matriz[i][j]
                )

                if cantidad > 0:

                    errores.append(
                        (
                            cantidad,
                            actividades[i],
                            actividades[j]
                        )
                    )


    errores.sort(
        reverse=True
    )


    print("\n" + "=" * 70)
    print("CONFUSIONES MÁS FRECUENTES")
    print("=" * 70)


    for cantidad, real, predicha in errores[:10]:

        print(
            f"Real {real} -> "
            f"Predicha {predicha}: "
            f"{cantidad} veces"
        )


if __name__ == "__main__":
    main()