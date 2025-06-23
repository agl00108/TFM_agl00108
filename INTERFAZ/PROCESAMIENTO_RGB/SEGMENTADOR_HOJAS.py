import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from ModeloRGB import predecir_con_modelo

class SegmentadorHojas:
    def __init__(self, modelo_path=None):
        self.imagen_original = None
        self.carpeta_salida = None
        self.contornos_info = []
        self.imagen_anotada = None

        try:
            if modelo_path is None:
                modelo_path = os.path.join(CURRENT_DIR, "modelo_vgg16.h5")
            self.modelo = load_model(modelo_path)
        except Exception as e:
            self.modelo = None
            raise Exception(f"No se pudo cargar el modelo desde {modelo_path}: {str(e)}")

    def procesar_hojas(self, imagen_path, carpeta_salida, inicio_num=1):
        if not os.path.exists(imagen_path):
            raise Exception("La imagen seleccionada no existe.")

        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)

        if self.modelo is None:
            raise Exception("El modelo no está cargado. Revise la ruta del modelo.")

        try:
            self.imagen_original = cv2.imread(imagen_path)
            if self.imagen_original is None:
                raise ValueError("No se pudo cargar la imagen.")

            gris = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
            desenfocada = cv2.GaussianBlur(gris, (7, 7), 0)
            _, binaria = cv2.threshold(desenfocada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            morfologica = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
            contornos, _ = cv2.findContours(morfologica, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            j = inicio_num - 1
            hojas_guardadas = 0
            self.contornos_info = []
            self.carpeta_salida = carpeta_salida

            for contorno in contornos:
                x, y, w, h = cv2.boundingRect(contorno)
                if w > 20 and h > 20:
                    hoja = self.imagen_original[y:y + h, x:x + w]
                    hoja_filename = f'hoja_{j + 1}.jpg'
                    j += 1
                    hoja_path = os.path.join(carpeta_salida, hoja_filename)
                    cv2.imwrite(hoja_path, hoja)
                    hojas_guardadas += 1
                    self.contornos_info.append({'contorno': contorno, 'filename': hoja_filename, 'rect': (x, y, w, h)})

            return hojas_guardadas

        except Exception as e:
            self.imagen_original = None
            self.contornos_info = []
            raise Exception(f"Error al procesar la imagen: {str(e)}")

    def predecir_variedades(self):
        if not self.modelo or not self.carpeta_salida or self.imagen_original is None:
            raise Exception("Falta el modelo, la carpeta de salida o la imagen original.")

        try:
            resultados = predecir_con_modelo(self.modelo, self.carpeta_salida)
            self.imagen_anotada = self.imagen_original.copy()

            for info in self.contornos_info:
                filename = info['filename']
                x, y, w, h = info['rect']
                resultado = resultados[resultados['Archivo'] == filename].iloc[0]
                prediccion = resultado['Predicción_Modelo']
                prob = resultado['Probabilidad']
                porcentaje = prob if prediccion == 'Picual' else 1 - prob
                etiqueta = f"{prediccion} {porcentaje:.0%}"

                cv2.rectangle(self.imagen_anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.imagen_anotada, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Reescalar la imagen anotada antes de guardarla
            escala = 0.5  # Reducir al 50% del tamaño original
            ancho = int(self.imagen_anotada.shape[1] * escala)
            alto = int(self.imagen_anotada.shape[0] * escala)
            imagen_reescalada = cv2.resize(self.imagen_anotada, (ancho, alto), interpolation=cv2.INTER_AREA)

            # Guardar la imagen reescalada temporalmente
            temp_path = os.path.join(self.carpeta_salida, "imagen_anotada_temp.jpg")
            cv2.imwrite(temp_path, imagen_reescalada)
            return temp_path

        except Exception as e:
            self.imagen_anotada = None
            raise Exception(f"Error al predecir variedades: {str(e)}")

    def descargar_imagen(self, ruta_guardado):
        if self.imagen_anotada is None:
            raise Exception("No hay imagen anotada para descargar.")

        escala = 0.8
        ancho = int(self.imagen_anotada.shape[1] * escala)
        alto = int(self.imagen_anotada.shape[0] * escala)
        imagen_reducida = cv2.resize(self.imagen_anotada, (ancho, alto), interpolation=cv2.INTER_AREA)
        cv2.imwrite(ruta_guardado, imagen_reducida)