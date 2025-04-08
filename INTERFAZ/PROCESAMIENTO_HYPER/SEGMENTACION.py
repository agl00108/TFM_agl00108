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
def procesar_shapefile_y_extraer_datos(ruta_shapefile, ruta_imagen, puntos_base_por_unidad_area, ruta_excel_salida,
                                       num_clusters):
    """
    Procesa un shapefile y una imagen hiperespectral para realizar clustering espectral y guardar los resultados en un Excel.

    Args:
        ruta_shapefile: Ruta al archivo shapefile (.shp).
        ruta_imagen: Ruta a la imagen hiperespectral (.dat).
        puntos_base_por_unidad_area: Número de puntos a generar por unidad de área.
        ruta_excel_salida: Ruta donde guardar el archivo Excel de salida.
        num_clusters: Número de clusters para el clustering espectral.
    """
    # Lee el shapefile usando geopandas
    gdf = gpd.read_file(ruta_shapefile)
    todos_dfs = []  # Lista para almacenar los DataFrames de cada polígono

    # Itera sobre cada polígono (hoja) en el shapefile
    for idx, fila in gdf.iterrows():
        poligono = fila['geometry']  # Obtiene la geometría del polígono
        area = poligono.area  # Calcula el área del polígono
        # Calcula el número de puntos a generar basado en el área
        puntos_por_poligono = max(1, int(puntos_base_por_unidad_area * area))
        # Genera puntos aleatorios dentro del polígono
        puntos_aleatorios = generar_puntos_aleatorios(poligono, puntos_por_poligono)
        # Extrae los valores espectrales en esos puntos
        valores_espectrales = obtener_valores_espectrales(ruta_imagen, puntos_aleatorios)

        # Crea un DataFrame con los valores espectrales y elimina filas con NaN
        df = pd.DataFrame(valores_espectrales).dropna()
        if len(df) < num_clusters:
            # Si hay menos puntos que clusters, calcula la media sin clustering
            df_media = pd.DataFrame(df.mean().values.reshape(1, -1), columns=df.columns)
            df_media.insert(0, 'Hoja', f"Hoja {idx}")
            df_media.insert(1, 'Cluster', -1)  # Indica que no se realizó clustering
            todos_dfs.append(df_media)
        else:
            # Realiza el clustering espectral
            clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
            etiquetas = clustering.fit_predict(df)
            df['Cluster'] = etiquetas
            # Calcula la media de cada cluster
            df_media = df.groupby('Cluster').mean().reset_index()
            df_media.insert(0, 'Hoja', f"Hoja {idx}")
            todos_dfs.append(df_media)

    # Combina todos los DataFrames en uno solo
    df_final = pd.concat(todos_dfs, ignore_index=True)
    # Renombra las columnas para mayor claridad
    df_final.columns = ['Hoja', 'Cluster'] + [f'Banda_{i}' for i in range(len(df_final.columns) - 2)]
    # Guarda el resultado en un archivo Excel
    df_final.to_excel(ruta_excel_salida, index=False)
    print(f"Resultados guardados en: {ruta_excel_salida}")