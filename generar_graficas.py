import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SALIDA...
# ============================================================

ruta_base = os.path.dirname(os.path.abspath(__file__))

ruta_graficas = os.path.join(
    ruta_base,
    "graficas"
)

os.makedirs(
    ruta_graficas,
    exist_ok=True
)


# ============================================================
# DATOS REALES DEL PROYECTO
# ============================================================

actividades = [
    "000", "001", "002", "003",
    "004", "005", "006", "007",
    "008", "009", "010", "011",
    "012", "013", "014", "015"
]


# repeticiones originales
originales = [
    232, 212, 267, 250,
    287, 293, 260, 385,
    299, 307, 311, 235,
    293, 313, 359, 313
]


# repeticiones después de la limpieza
finales = [
    178, 192, 170, 200,
    276, 281, 211, 382,
    299, 307, 249, 189,
    201, 274, 270, 211
]


# resultados reales de validación de K
valores_k = [
    1, 3, 5, 7, 9
]

accuracy_validacion = [
    92.20,
    81.28,
    69.50,
    66.38,
    62.05
]


# accuracy final por actividad
accuracy_actividad = [
    100.00,
    96.67,
    92.59,
    100.00,
    100.00,
    100.00,
    96.97,
    98.28,
    84.78,
    78.72,
    92.11,
    100.00,
    100.00,
    92.86,
    95.12,
    93.94
]


# matriz de confusión real
matriz_confusion = np.array([

    [28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 29, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 25, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 1, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 57, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 2, 39, 1, 0, 4, 0, 0, 0, 0],

    [0, 0, 0, 3, 0, 0, 0, 2, 3, 37, 0, 0, 2, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 35, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 39, 2, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 39, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 31]

])


# ============================================================
# grafica 
# repeticiones originales v finales
# ============================================================

x = np.arange(
    len(actividades)
)

ancho = 0.38


plt.figure(
    figsize=(12, 6)
)


plt.bar(
    x - ancho / 2,
    originales,
    width=ancho,
    label="Originales"
)


plt.bar(
    x + ancho / 2,
    finales,
    width=ancho,
    label="Después de limpieza"
)


plt.title(
    "Repeticiones originales y finales por actividad"
)

plt.xlabel(
    "Actividad"
)

plt.ylabel(
    "Número de repeticiones"
)

plt.xticks(
    x,
    actividades
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


ruta = os.path.join(
    ruta_graficas,
    "01_originales_vs_finales.png"
)

plt.savefig(
    ruta,
    dpi=300
)

plt.close()


# ============================================================
# grafica 2
# resumen de la limpieza
# ============================================================

categorias_limpieza = [
    "Originales",
    "Ceros eliminados",
    "Duplicados eliminados",
    "Dataset final"
]

valores_limpieza = [
    4616,
    68,
    658,
    3890
]


plt.figure(
    figsize=(9, 6)
)


barras = plt.bar(
    categorias_limpieza,
    valores_limpieza
)


plt.title(
    "Resumen del proceso de limpieza del dataset"
)

plt.ylabel(
    "Número de repeticiones"
)

plt.xticks(
    rotation=10
)


for barra, valor in zip(
    barras,
    valores_limpieza
):

    plt.text(
        barra.get_x()
        + barra.get_width() / 2,

        barra.get_height() + 50,

        str(valor),

        ha="center"
    )


plt.ylim(
    0,
    5000
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


ruta = os.path.join(
    ruta_graficas,
    "02_limpieza_dataset.png"
)

plt.savefig(
    ruta,
    dpi=300
)

plt.close()


# ============================================================
# grafica 3
# accuracy de k
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    valores_k,
    accuracy_validacion,
    marker="o"
)


plt.title(
    "Accuracy de validación según el valor de K"
)

plt.xlabel(
    "Valor de K"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xticks(
    valores_k
)

plt.ylim(
    50,
    100
)

plt.grid(
    alpha=0.3
)


for k, accuracy in zip(
    valores_k,
    accuracy_validacion
):

    plt.text(
        k,
        accuracy + 1.2,
        f"{accuracy:.2f}%",
        ha="center"
    )


plt.tight_layout()


ruta = os.path.join(
    ruta_graficas,
    "03_accuracy_por_k.png"
)

plt.savefig(
    ruta,
    dpi=300
)

plt.close()


# ============================================================
# grafica 4
# accuracy por act
# ============================================================

plt.figure(
    figsize=(12, 6)
)


barras = plt.bar(
    actividades,
    accuracy_actividad
)


plt.title(
    "Porcentaje de aciertos por actividad"
)

plt.xlabel(
    "Actividad"
)

plt.ylabel(
    "Aciertos (%)"
)

plt.ylim(
    0,
    110
)


for barra, accuracy in zip(
    barras,
    accuracy_actividad
):

    plt.text(
        barra.get_x()
        + barra.get_width() / 2,

        barra.get_height() + 1,

        f"{accuracy:.1f}",

        ha="center",
        fontsize=8,
        rotation=90
    )


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


ruta = os.path.join(
    ruta_graficas,
    "04_accuracy_por_actividad.png"
)

plt.savefig(
    ruta,
    dpi=300
)

plt.close()


# ============================================================
# grafica 5
# matriz confusion
# ============================================================

plt.figure(
    figsize=(11, 9)
)


plt.imshow(
    matriz_confusion,
    interpolation="nearest",
    aspect="auto"
)


plt.title(
    "Matriz de confusión del modelo KNN"
)

plt.xlabel(
    "Actividad predicha"
)

plt.ylabel(
    "Actividad real"
)


plt.xticks(
    np.arange(len(actividades)),
    actividades,
    rotation=45
)

plt.yticks(
    np.arange(len(actividades)),
    actividades
)


# Mostrar valores dentro de cada celda
for i in range(
    matriz_confusion.shape[0]
):

    for j in range(
        matriz_confusion.shape[1]
    ):

        valor = matriz_confusion[i, j]

        if valor > 0:

            plt.text(
                j,
                i,
                str(valor),
                ha="center",
                va="center"
            )


plt.colorbar(
    label="Número de muestras"
)

plt.tight_layout()


ruta = os.path.join(
    ruta_graficas,
    "05_matriz_confusion.png"
)

plt.savefig(
    ruta,
    dpi=300
)

plt.close()


# ============================================================
# final
# ============================================================

print("=" * 65)
print("GRAFICAS GENERADAS CORRECTAMENTE")
print("=" * 65)

print(
    f"\nLas graficas fueron guardadas en:\n"
    f"{ruta_graficas}"
)

print("\nArchivos creados:")

print(
    "01_originales_vs_finales.png"
)

print(
    "02_limpieza_dataset.png"
)

print(
    "03_accuracy_por_k.png"
)

print(
    "04_accuracy_por_actividad.png"
)

print(
    "05_matriz_confusion.png"
)