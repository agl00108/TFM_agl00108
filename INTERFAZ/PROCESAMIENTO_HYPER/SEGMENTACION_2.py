import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import spectral.io.envi as envi
import rasterio
import os

# Configura el número máximo de núcleos (si es necesario)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# Función para generar puntos en una cuadrícula regular dentro de un polígono
def generar_puntos_cuadricula(poligono, num_divisiones):
    """
    Genera puntos en una cuadrícula regular dentro de un polígono dado.

    Args:
        poligono: Polígono (shapely.geometry.Polygon) dentro del cual generar los puntos.
        num_divisiones: Número de divisiones en cada eje (cuadrícula num_divisiones x num_divisiones).

    Returns:
        Lista de puntos (shapely.geometry.Point) generados dentro del polígono.
    """
    puntos = []
    min_x, min_y, max_x, max_y = poligono.bounds
    ancho = max_x - min_x
    alto = max_y - min_y
    paso_x = ancho / num_divisiones
    paso_y = alto / num_divisiones

    x_coords = np.arange(min_x + paso_x / 2, max_x, paso_x)
    y_coords = np.arange(min_y + paso_y / 2, max_y, paso_y)

    for x in x_coords:
        for y in y_coords:
            punto = Point(x, y)
            if poligono.contains(punto):
                puntos.append(punto)

    return puntos


# Función para obtener valores espectrales y filtrar píxeles
def obtener_valores_espectrales(ruta_imagen, puntos, umbral=0.35, bandas_filtro=range(0, 191)):
    """
    Extrae los valores espectrales de una imagen hiperespectral y filtra píxeles según un umbral.

    Args:
        ruta_imagen: Ruta al archivo de la imagen hiperespectral (.dat).
        puntos: Lista de puntos (shapely.geometry.Point) donde extraer los valores.
        umbral: Umbral para filtrar píxeles (valores > umbral en bandas_filtro se eliminan).
        bandas_filtro: Índices de las bandas usadas para el filtrado.

    Returns:
        Lista de valores espectrales filtrados, coordenadas y bandas mantenidas.
    """
    img = envi.open(ruta_imagen.replace(".dat", ".hdr"), ruta_imagen)
    datos_imagen = np.array(img.load())
    altura, ancho, bandas = datos_imagen.shape

    # Selecciona todas las bandas inicialmente
    bandas_a_mantener = np.arange(bandas)

    with rasterio.open(ruta_imagen) as src:
        transformacion = src.transform

    datos_espectrales = []
    coordenadas = []
    for punto in puntos:
        col, fila = ~transformacion * (punto.x, punto.y)
        col, fila = int(col), int(fila)
        if 0 <= fila < altura and 0 <= col < ancho:
            espectro = datos_imagen[fila, col, :]
            # Filtrado basado en umbral y desviación estándar
            valores_filtro = espectro[bandas_filtro]
            std_valores = np.std(valores_filtro)
            if np.any(valores_filtro > umbral + std_valores):  # Considera desv. estándar
                continue  # Descarta el píxel si alguna banda supera el umbral + std
            datos_espectrales.append(espectro)
            coordenadas.append((punto.x, punto.y))

    return datos_espectrales, coordenadas, bandas_a_mantener

# Función para calcular medias por grupos de bandas
def calcular_medias_por_grupos(datos_espectrales, bandas_a_mantener):
    """
    Calcula la media de valores espectrales agrupando bandas según las reglas especificadas.

    Args:
        datos_espectrales: Lista de valores espectrales.
        bandas_a_mantener: Índices de las bandas mantenidas.

    Returns:
        Lista de medias calculadas por grupos de bandas.
    """
    medias = []
    for espectro in datos_espectrales:
        media_espectro = []
        # Bandas 1 a 199: media cada 10
        for i in range(1, 200, 10):
            if i < len(espectro):
                media_espectro.append(np.nanmean(espectro[i:min(i + 10, len(espectro))]))
            else:
                media_espectro.append(np.nan)
        # Bandas 199 a 283: media cada 5
        for i in range(199, 284, 5):
            if i < len(espectro):
                media_espectro.append(np.nanmean(espectro[i:min(i + 5, len(espectro))]))
            else:
                media_espectro.append(np.nan)
        # Resto de bandas: media cada 10
        for i in range(284, len(espectro), 10):
            media_espectro.append(np.nanmean(espectro[i:min(i + 10, len(espectro))]))
        medias.append(media_espectro)

    return medias

# Función principal para procesar el shapefile y guardar resultados
def procesar_shapefile_y_extraer_datos(ruta_shapefile, ruta_imagen, num_divisiones, ruta_excel_salida, especie=None):
    """
    Procesa un shapefile y una imagen hiperespectral para generar puntos en cuadrícula, filtrar píxeles y calcular medias.

    Args:
        ruta_shapefile: Ruta al archivo shapefile (.shp).
        ruta_imagen: Ruta a la imagen hiperespectral (.dat).
        num_divisiones: Número de divisiones para la cuadrícula.
        ruta_excel_salida: Ruta donde guardar el archivo Excel de salida.
        especie: Especie de la hoja (opcional).
    """
    gdf = gpd.read_file(ruta_shapefile)
    todos_dfs = []

    for idx, fila in gdf.iterrows():
        poligono = fila['geometry']
        puntos_cuadricula = generar_puntos_cuadricula(poligono, num_divisiones)
        valores_espectrales, coordenadas, bandas_a_mantener = obtener_valores_espectrales(ruta_imagen,
                                                                                          puntos_cuadricula)

        # Calcular medias por grupos de bandas
        medias = calcular_medias_por_grupos(valores_espectrales, bandas_a_mantener)

        # Crear DataFrame
        columnas_medias = (
                [f'Media_Bandas_{i + 1}_{min(i + 10, 200)}' for i in range(1, 200, 10)] +
                [f'Media_Bandas_{i + 1}_{min(i + 5, 284)}' for i in range(199, 284, 5)] +
                [f'Media_Bandas_{i + 1}_{min(i + 10, len(bandas_a_mantener))}' for i in
                 range(284, len(bandas_a_mantener), 10)]
        )
        df = pd.DataFrame(medias, columns=columnas_medias)
        df['Hoja'] = f"Hoja {idx}"
        df['X'] = [coord[0] for coord in coordenadas]
        df['Y'] = [coord[1] for coord in coordenadas]
        if especie:
            df['Especie'] = especie

        todos_dfs.append(df)

    if todos_dfs:
        df_final = pd.concat(todos_dfs, ignore_index=True)
        # Reorganizar columnas
        columnas_base = ['Hoja', 'X', 'Y']
        if especie:
            columnas_base.append('Especie')
        df_final = df_final[columnas_base + [col for col in df_final.columns if col.startswith('Media_')]]
        df_final.to_excel(ruta_excel_salida, index=False)
        print(f"Resultados guardados en: {ruta_excel_salida}")
    else:
        print("No se encontraron datos válidos después del filtrado.")

if __name__ == "__main__":
    procesar_shapefile_y_extraer_datos(
        ruta_shapefile="../../Resultados/MARTENA/Martena_1/1_Shapefile/mar1.shp",
        ruta_imagen="../../Resultados/MARTENA/Martena_1/2_Recorte/imagen_recortada.dat",
        num_divisiones=10,
        ruta_excel_salida="../../Resultados/MARTENA/Martena_1/3_Segmentacion_2/resultados_mar1_2.xlsx",
        especie="MAR"
    )