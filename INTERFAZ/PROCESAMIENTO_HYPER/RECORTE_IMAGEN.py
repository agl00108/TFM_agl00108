import numpy as np
import spectral.io.envi as envi
import geopandas as gpd
from rasterio.mask import mask
import rasterio
from rasterio.transform import Affine

"""
Esta función carga una imagen hiperespectral en formato ENVI, la convierte temporalmente en GeoTIFF
para aplicar una máscara espacial definida por un shapefile, y guarda el resultado recortado nuevamente
en formato ENVI.

Args:
    shapefile (str): ruta al archivo .shp que contiene las geometrías para recortar la imagen.
    imagen (str): ruta al archivo .hdr de la imagen ENVI (se asume que el archivo .dat tiene el mismo nombre).
    salida (str): carpeta de salida donde se guardará la imagen recortada.
    output_filename (str, optional): nombre del archivo de salida sin extensión. Por defecto es "imagen_recortada".

Returns:
    None. La imagen recortada se guarda como un archivo .dat/.hdr en la carpeta especificada.
"""

def recortar_imagen(shapefile, imagen, salida, output_filename="imagen_recortada"):
    print(f"Intentando leer shapefile: {shapefile}")
    try:
        gdf = gpd.read_file(shapefile)
    except Exception as e:
        raise Exception(f"Error al leer el shapefile: {str(e)}")

    try:
        img = envi.open(imagen, imagen.replace(".hdr", ".dat"))
        img_data = np.array(img.load())
    except Exception as e:
        raise Exception(f"Error al abrir la imagen ENVI: {str(e)}")

    height, width, bands = img_data.shape
    transform = Affine(1, 0, 0, 0, 1, 0)

    temp_tif = "temp_image.tif"
    with rasterio.open(
        temp_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=img_data.dtype,
        transform=transform
    ) as dst:
        for i in range(bands):
            dst.write(img_data[:, :, i], i + 1)

    # Realizar el recorte
    polygons = [geom for geom in gdf.geometry]
    with rasterio.open(temp_tif) as src:
        out_image, out_transform = mask(src, polygons, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "ENVI",
            "count": out_image.shape[0],
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        output_image_path = f"{salida}/{output_filename}.dat"
        with rasterio.open(output_image_path, "w", **out_meta) as dest:
            dest.write(out_image)

    # Eliminar el archivo temporal
    import os
    os.remove(temp_tif)

    print(f"Imagen recortada guardada como: {output_image_path}")