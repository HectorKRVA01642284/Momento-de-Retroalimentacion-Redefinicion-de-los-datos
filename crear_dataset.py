import os
import csv
import hashlib
import numpy as np


# ============================================================
# rutas
# ============================================================

ruta_base = os.path.dirname(os.path.abspath(__file__))

ruta_datos = os.path.join(
    ruta_base,
    "Rehab_exercise",
    "d01_raw_data"
)

ruta_salida = os.path.join(
    ruta_base,
    "data"
)

os.makedirs(ruta_salida, exist_ok=True)

archivo_salida = os.path.join(
    ruta_salida,
    "rehab_features.csv"
)


# ============================================================
# NOMBRES DE LOS CANALES
# ============================================================

canales = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "pitch3"
]


# ============================================================
# EXTRAER CARACTERÍSTICAS
# ============================================================

def extraer_caracteristicas(repeticion):

    caracteristicas = []

    for canal in range(6):

        valores = repeticion[:, canal]

        media = np.mean(valores)
        desviacion = np.std(valores)
        minimo = np.min(valores)
        maximo = np.max(valores)
        rango = maximo - minimo

        caracteristicas.extend([
            media,
            desviacion,
            minimo,
            maximo,
            rango
        ])

    return caracteristicas


# ============================================================
# SEGUNDA RONDA
# IDENTIFICAR REPETICIONES CON ETIQUETAS CONFLICTIVAS
# ============================================================

print("=" * 65)
print("CREACION DEL NUEVO DATASET REHAB")
print("=" * 65)

print("\nBuscando repeticiones con etiquetas conflictivas...")


actividades_por_hash = {}


for actividad in range(16):

    etiqueta = f"{actividad:03d}"

    nombre_archivo = f"{etiqueta}_2.npy"

    ruta_archivo = os.path.join(
        ruta_datos,
        nombre_archivo
    )

    datos = np.load(ruta_archivo)

    for repeticion in datos:

        identificador = hashlib.sha256(
            repeticion.tobytes()
        ).hexdigest()

        if identificador not in actividades_por_hash:
            actividades_por_hash[identificador] = set()

        actividades_por_hash[identificador].add(etiqueta)


hashes_conflictivos = set()


for identificador, actividades in actividades_por_hash.items():

    if len(actividades) > 1:
        hashes_conflictivos.add(identificador)


print(
    "Patrones con etiquetas conflictivas:",
    len(hashes_conflictivos)
)


# ============================================================
# encabezados del csv
# ============================================================

encabezados = []


for canal in canales:

    encabezados.extend([
        f"{canal}_mean",
        f"{canal}_std",
        f"{canal}_min",
        f"{canal}_max",
        f"{canal}_range"
    ])


encabezados.append("actividad")


# ============================================================
# SEGUNDA RONDA
# LIMPIEZA Y EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================

filas_dataset = []

hashes_guardados = set()


total_original = 0
total_ceros = 0
total_conflictos = 0
total_duplicados = 0
total_guardados = 0


estadisticas_actividad = {}


for actividad in range(16):

    etiqueta = f"{actividad:03d}"

    nombre_archivo = f"{etiqueta}_2.npy"

    ruta_archivo = os.path.join(
        ruta_datos,
        nombre_archivo
    )

    datos = np.load(ruta_archivo)


    estadisticas_actividad[etiqueta] = {
        "original": len(datos),
        "ceros": 0,
        "conflictos": 0,
        "duplicados": 0,
        "guardados": 0
    }


    for repeticion in datos:

        total_original += 1


        # ----------------------------------------------------
        # revisar si toda la repetición contiene ceros
        # ----------------------------------------------------

        if np.all(repeticion == 0):

            total_ceros += 1

            estadisticas_actividad[
                etiqueta
            ]["ceros"] += 1

            continue


        # ----------------------------------------------------
        # identificador de la repetición
        # ----------------------------------------------------

        identificador = hashlib.sha256(
            repeticion.tobytes()
        ).hexdigest()


        # ----------------------------------------------------
        # eliminar posibles conflictos entre actividades
        # ----------------------------------------------------

        if identificador in hashes_conflictivos:

            total_conflictos += 1

            estadisticas_actividad[
                etiqueta
            ]["conflictos"] += 1

            continue


        # ----------------------------------------------------
        # eliminar duplicados exactos
        # ----------------------------------------------------

        if identificador in hashes_guardados:

            total_duplicados += 1

            estadisticas_actividad[
                etiqueta
            ]["duplicados"] += 1

            continue


        hashes_guardados.add(identificador)


        # ----------------------------------------------------
        # extraer las 30 características
        # ----------------------------------------------------

        caracteristicas = extraer_caracteristicas(
            repeticion
        )


        # agregar la actividad como etiqueta
        fila = caracteristicas + [etiqueta]

        filas_dataset.append(fila)


        total_guardados += 1

        estadisticas_actividad[
            etiqueta
        ]["guardados"] += 1


# ============================================================
# guardar csv
# ============================================================

with open(
    archivo_salida,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    escritor = csv.writer(archivo)

    escritor.writerow(encabezados)

    escritor.writerows(filas_dataset)


# ============================================================
# RESULTADOS POR ACTIVIDAD
# ============================================================

print("\n" + "=" * 65)
print("RESULTADOS POR ACTIVIDAD")
print("=" * 65)


for actividad in range(16):

    etiqueta = f"{actividad:03d}"

    info = estadisticas_actividad[etiqueta]

    print(
        f"Actividad {etiqueta}: "
        f"original={info['original']}, "
        f"ceros={info['ceros']}, "
        f"conflictos={info['conflictos']}, "
        f"duplicados={info['duplicados']}, "
        f"guardados={info['guardados']}"
    )


# ============================================================
# RESUMEN GENERAL...
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN GENERAL...")
print("=" * 65)

print(
    f"Repeticiones originales: {total_original}"
)

print(
    f"Repeticiones completamente en cero eliminadas: "
    f"{total_ceros}"
)

print(
    f"Repeticiones con etiquetas conflictivas eliminadas: "
    f"{total_conflictos}"
)

print(
    f"Duplicados exactos eliminados: "
    f"{total_duplicados}"
)

print(
    f"Repeticiones guardadas en el nuevo dataset: "
    f"{total_guardados}"
)

print(
    f"Numero de caracteristicas: "
    f"{len(encabezados) - 1}"
)

print(
    f"Numero total de columnas: "
    f"{len(encabezados)}"
)

print(
    f"\nDataset creado en:\n{archivo_salida}"
)