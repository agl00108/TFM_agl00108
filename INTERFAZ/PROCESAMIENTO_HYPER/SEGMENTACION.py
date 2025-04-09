import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import spectral.io.envi as envi
import rasterio
from sklearn.cluster import SpectralClustering
import os

# Configura el número máximo de núcleos a usar para el clustering
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


# Función para generar puntos aleatorios dentro de un polígono
def generar_puntos_aleatorios(poligono, num_puntos, intentos_maximos=1000):
    """
    Genera puntos aleatorios dentro de un polígono dado.

    Args:
        poligono: Polígono (shapely.geometry.Polygon) dentro del cual generar los puntos.
        num_puntos: Número de puntos a generar.
        intentos_maximos: Número máximo de intentos para generar cada punto.

    Returns:
        Lista de puntos (shapely.geometry.Point) generados dentro del polígono.
    """
    puntos = []
    min_x, min_y, max_x, max_y = poligono.bounds  # Obtiene los límites del polígono
    intentos = 0
    while len(puntos) < num_puntos and intentos < intentos_maximos:
        punto_aleatorio = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if poligono.contains(punto_aleatorio):  # Verifica si el punto está dentro del polígono
            puntos.append(punto_aleatorio)
        intentos += 1
    return puntos


# Función para obtener valores espectrales en puntos específicos
def obtener_valores_espectrales(ruta_imagen, puntos):
    """
    Extrae los valores espectrales de una imagen hiperespectral en las coordenadas de los puntos dados.

    Args:
        ruta_imagen: Ruta al archivo de la imagen hiperespectral (.dat).
        puntos: Lista de puntos (shapely.geometry.Point) donde extraer los valores.

    Returns:
        Lista de valores espectrales para cada punto.
    """
    # Abre la imagen hiperespectral usando la librería spectral
    img = envi.open(ruta_imagen.replace(".dat", ".hdr"), ruta_imagen)
    datos_imagen = np.array(img.load())  # Carga los datos de la imagen como un arreglo numpy
    altura, ancho, bandas = datos_imagen.shape  # Obtiene las dimensiones de la imagen

    # Selecciona las bandas a mantener (excluye bandas 0-21 y 420-448)
    bandas_a_mantener = np.concatenate([np.arange(22, 420), np.arange(449, bandas)])
    datos_imagen = datos_imagen[:, :, bandas_a_mantener]

    # Obtiene la transformación afín de la imagen para convertir coordenadas geográficas a píxeles
    with rasterio.open(ruta_imagen) as src:
        transformacion = src.transform

    datos_espectrales = []
    for punto in puntos:
        # Convierte las coordenadas del punto (x, y) a índices de píxel (columna, fila)
        col, fila = ~transformacion * (punto.x, punto.y)
        col, fila = int(col), int(fila)
        # Verifica si el píxel está dentro de los límites de la imagen
        if 0 <= fila < altura and 0 <= col < ancho:
            datos_espectrales.append(datos_imagen[fila, col, :])
        else:
            # Si el punto está fuera de la imagen, añade valores NaN
            datos_espectrales.append([np.nan] * len(bandas_a_mantener))
    return datos_espectrales


# Función principal para procesar el shapefile y realizar el clustering espectral
def procesar_shapefile_y_extraer_datos(ruta_shapefile, ruta_imagen, puntos_base_por_unidad_area, ruta_excel_salida, num_clusters, especie=None):
    """
    Procesa un shapefile y una imagen hiperespectral para realizar clustering espectral y guardar los resultados en un Excel.

    Args:
        ruta_shapefile: Ruta al archivo shapefile (.shp).
        ruta_imagen: Ruta a la imagen hiperespectral (.dat).
        puntos_base_por_unidad_area: Número de puntos a generar por unidad de área.
        ruta_excel_salida: Ruta donde guardar el archivo Excel de salida.
        num_clusters: Número de clusters para el clustering espectral.
    """
    gdf = gpd.read_file(ruta_shapefile)
    todos_dfs = []

    for idx, fila in gdf.iterrows():
        poligono = fila['geometry']
        area = poligono.area
        puntos_por_poligono = max(1, int(puntos_base_por_unidad_area * area))
        puntos_aleatorios = generar_puntos_aleatorios(poligono, puntos_por_poligono)
        valores_espectrales = obtener_valores_espectrales(ruta_imagen, puntos_aleatorios)

        df = pd.DataFrame(valores_espectrales).dropna()
        if len(df) < num_clusters:
            df_media = pd.DataFrame(df.mean().values.reshape(1, -1), columns=df.columns)
            df_media.insert(0, 'Hoja', f"Hoja {idx}")
            df_media.insert(1, 'Cluster', -1)
            if especie:
                df_media.insert(2, 'Especie', especie)  # Agregar columna de especie si se proporcionó
            todos_dfs.append(df_media)
        else:
            clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
            etiquetas = clustering.fit_predict(df)
            df['Cluster'] = etiquetas
            df_media = df.groupby('Cluster').mean().reset_index()
            df_media.insert(0, 'Hoja', f"Hoja {idx}")
            if especie:
                df_media.insert(2, 'Especie', especie)  # Agregar columna de especie si se proporcionó
            todos_dfs.append(df_media)

    df_final = pd.concat(todos_dfs, ignore_index=True)
    # Renombrar columnas (ajustar según si hay especie o no)
    columnas_base = ['Hoja', 'Cluster']
    if especie:
        columnas_base.append('Especie')
    df_final.columns = columnas_base + [f'Banda_{i}' for i in range(len(df_final.columns) - len(columnas_base))]
    df_final.to_excel(ruta_excel_salida, index=False)
    print(f"Resultados guardados en: {ruta_excel_salida}")