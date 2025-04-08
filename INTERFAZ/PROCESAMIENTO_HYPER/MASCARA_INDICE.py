import spectral
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
import fiona
import os

"""
Función para calcular el Índice de Vegetación Estandarizado (EVI)

Args:
    banda_rojo: banda roja
    banda_nir: banda infrarroja cercana
    banda_azul: banda azul

Returns:
    El índice EVI calculado.
"""
def calcular_evi(banda_rojo, banda_nir, banda_azul):

    G = 2.5  # Factor de ganancia
    C1 = 6.0  # Coeficiente de la banda roja
    C2 = 7.5  # Coeficiente de la banda azul
    L = 1.0  # Factor de corrección para el suelo

    evi = G * (banda_nir - banda_rojo) / (banda_nir + C1 * banda_rojo - C2 * banda_azul + L)
    return evi

"""
Función para extraer metadatos de un archivo .hdr (información sobre la proyección, el tamaño de píxel y otras propiedades).

Args:
    ruta_hdr: Ruta al archivo de encabezado (.hdr)

Returns:
    Un diccionario con los metadatos extraídos.
"""
def extraer_metadatos_hdr(ruta_hdr):
    metadatos = {}
    with open(ruta_hdr, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                metadatos[key.strip()] = value.strip()

    if 'map info' in metadatos:
        map_info = metadatos['map info'].split(',')
        metadatos['map_info'] = {
            'projection': map_info[0].strip(),
            'x_ref': float(map_info[1].strip()),
            'y_ref': float(map_info[2].strip()),
            'pixel_width': float(map_info[3].strip()),
            'pixel_height': float(map_info[4].strip()),
            'crs': map_info[5].strip() if len(map_info) > 5 else None
        }
    return metadatos

"""
Convierte una máscara de vegetación en una imagen a un shapefile (vectorial).
Solo se exportan las áreas que cumplen con un área mínima (por defecto 1000).

Args:
    imagen: imagen original (no se utiliza en este caso)
    mascara: máscara de vegetación
    ruta_shapefile: ruta donde se guardará el shapefile
    metadatos: metadatos de la imagen (opcional)
    min_area: área mínima para considerar un polígono (por defecto 1000)

"""
def exportar_a_vector(imagen, mascara, ruta_shapefile, metadatos=None, min_area=1000):
    transform = Affine(1, 0, 0, 0, 1, 0)  # Definimos una transformación afín (espacial)
    geometries = shapes(mascara.astype(np.uint8), transform=transform)  # Extraemos las geometrías de la máscara
    schema = {'geometry': 'Polygon', 'properties': {'value': 'int'}}  # Esquema de los datos en el shapefile

    # Abrimos el shapefile para escribir las geometrías
    with fiona.open(ruta_shapefile, 'w', driver='ESRI Shapefile', schema=schema) as shapefile:
        for geom, value in geometries:
            if value == 1:  # Solo procesamos las geometrías correspondientes a la vegetación
                polygon = shape(geom)  # Convertimos la geometría en un polígono
                if polygon.area >= min_area:  # Filtramos por área mínima
                    shapefile.write({
                        'geometry': mapping(polygon),  # Escribimos la geometría en el shapefile
                        'properties': {'value': int(value)}  # Escribimos el valor del píxel (1 para vegetación)
                    })

# Función principal para procesar una imagen hiperespectral
def procesar_imagen(ruta_imagen, exportar_raster, exportar_vector, ruta_exportacion, nombre_archivo):
    """
    Procesa una imagen hiperespectral, calcula el EVI, genera una máscara de vegetación y 
    exporta los resultados en formato raster y vectorial (shapefile).
    """
    img = spectral.open_image(ruta_imagen)  # Abrimos la imagen hiperespectral
    imagen = img.load().astype(np.float32)  # Cargamos la imagen como un array de flotantes
    ruta_hdr = ruta_imagen  # La ruta del archivo .hdr es la misma que la de la imagen
    metadatos = extraer_metadatos_hdr(ruta_hdr)  # Extraemos los metadatos del archivo .hdr

    # Seleccionamos las bandas necesarias (rojo, NIR, azul) según el índice EVI
    banda_rojo = imagen[:, :, 200]  # Banda roja (por índice de banda)
    banda_nir = imagen[:, :, 307]  # Banda NIR (infrarrojo cercano)
    banda_azul = imagen[:, :, 53]  # Banda azul

    # Calculamos el EVI
    evi = calcular_evi(banda_rojo, banda_nir, banda_azul)

    # Generamos la máscara de vegetación (valores de EVI entre 0.21 y 2.0)
    mascara_vegetacion = (evi >= 0.21) & (evi <= 2)

    # Definimos la transformación espacial de la imagen
    transform = Affine(1, 0, 0, 0, 1, 0)

    # Exportar el resultado como archivo raster (opcional)
    if exportar_raster:
        ruta_mascara_tif = os.path.join(ruta_exportacion, nombre_archivo + '.tif')
        with rasterio.open(
            ruta_mascara_tif, 'w', driver='GTiff', height=mascara_vegetacion.shape[0],
            width=mascara_vegetacion.shape[1], count=1, dtype=np.uint8, transform=transform
        ) as dst:
            dst.write(mascara_vegetacion.astype(np.uint8), 1)  # Escribimos la máscara en el archivo raster
        print(f"Máscara raster guardada en: {ruta_mascara_tif}")

    # Exportar el resultado como shapefile (vectorial) (opcional)
    if exportar_vector:
        ruta_shapefile = os.path.join(ruta_exportacion, nombre_archivo + '.shp')
        exportar_a_vector(imagen, mascara_vegetacion, ruta_shapefile, metadatos=metadatos)
        print(f"Máscara vectorial guardada en: {ruta_shapefile}")
