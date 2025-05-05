import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow.keras.models import load_model
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from VGG16 import predecir_con_modelo

class SegmentadorHojasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentador de Hojas (RGB)")

        self.container = ttk.Frame(root)
        self.container.pack(padx=10, pady=10)

        # --- Interfaz Gráfica ---
        ttk.Label(self.container, text="Seleccionar imagen RGB (.jpg):").grid(row=0, column=0, pady=5)
        self.imagen_entry = ttk.Entry(self.container, width=50)
        self.imagen_entry.grid(row=0, column=1, pady=5)
        ttk.Button(self.container, text="Buscar", command=self.seleccionar_imagen).grid(row=0, column=2, pady=5)

        ttk.Label(self.container, text="Carpeta de salida:").grid(row=1, column=0, pady=5)
        self.salida_entry = ttk.Entry(self.container, width=50)
        self.salida_entry.grid(row=1, column=1, pady=5)
        ttk.Button(self.container, text="Buscar", command=self.seleccionar_salida).grid(row=1, column=2, pady=5)

        ttk.Label(self.container, text="Número inicial de la hoja:").grid(row=2, column=0, pady=5)
        self.inicio_entry = ttk.Entry(self.container, width=10)
        self.inicio_entry.grid(row=2, column=1, pady=5, sticky="w")
        self.inicio_entry.insert(0, "1")

        self.info_label = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label.grid(row=3, column=0, columnspan=3, pady=5)

        ttk.Button(self.container, text="Procesar", command=self.procesar_hojas).grid(row=4, column=0, columnspan=3, pady=10)

        self.boton_predecir = ttk.Button(self.container, text="Predecir Variedad", command=self.predecir_variedades)
        self.boton_predecir.grid(row=5, column=0, columnspan=3, pady=10)
        self.boton_predecir.grid_remove()

        self.boton_descargar = ttk.Button(self.container, text="Descargar Imagen", command=self.descargar_imagen)
        self.boton_descargar.grid(row=6, column=0, columnspan=3, pady=10)
        self.boton_descargar.grid_remove()

        # Variables de estado
        self.imagen_original = None
        self.carpeta_salida = None
        self.contornos_info = []
        self.imagen_anotada = None  # Para almacenar la imagen anotada

        # Cargar el modelo al iniciar la aplicación
        try:
            self.modelo = load_model(MODELO_PATH)
            self.info_label.config(text=f"Modelo cargado desde: {MODELO_PATH}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo desde {MODELO_PATH}: {str(e)}")
            self.modelo = None
            self.info_label.config(text="Error al cargar el modelo. El programa no funcionará correctamente.")

    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[("Archivos JPG", "*.jpg")])
        if ruta:
            self.imagen_entry.delete(0, tk.END)
            self.imagen_entry.insert(0, ruta)

    def seleccionar_salida(self):
        ruta = filedialog.askdirectory()
        if ruta:
            self.salida_entry.delete(0, tk.END)
            self.salida_entry.insert(0, ruta)

    def procesar_hojas(self):
        imagen_path = self.imagen_entry.get()
        self.carpeta_salida = self.salida_entry.get()
        try:
            inicio_num = int(self.inicio_entry.get())
        except ValueError:
            messagebox.showerror("Error", "El número inicial debe ser un entero válido.")
            return

        if not imagen_path or not self.carpeta_salida:
            messagebox.showerror("Error", "Debe seleccionar la imagen y la carpeta de salida.")
            return

        if not os.path.exists(imagen_path):
            messagebox.showerror("Error", "La imagen seleccionada no existe.")
            return

        if not os.path.exists(self.carpeta_salida):
            os.makedirs(self.carpeta_salida)

        if self.modelo is None:
            messagebox.showerror("Error", "El modelo no está cargado. Revise la ruta del modelo.")
            return

        self.info_label.config(text="Procesando imagen...")
        self.root.update_idletasks()

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

            for contorno in contornos:
                x, y, w, h = cv2.boundingRect(contorno)
                if w > 20 and h > 20:
                    hoja = self.imagen_original[y:y + h, x:x + w]
                    hoja_filename = f'hoja_{j + 1}.jpg'
                    j += 1
                    hoja_path = os.path.join(self.carpeta_salida, hoja_filename)
                    cv2.imwrite(hoja_path, hoja)
                    hojas_guardadas += 1
                    self.contornos_info.append({'contorno': contorno, 'filename': hoja_filename, 'rect': (x, y, w, h)})

            self.info_label.config(text=f"Se han guardado {hojas_guardadas} hojas en: {self.carpeta_salida}")
            messagebox.showinfo("Éxito", f"Proceso completado. Se guardaron {hojas_guardadas} hojas.")
            self.boton_predecir.grid()

        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
            self.info_label.config(text="Error al procesar la imagen.")
            self.imagen_original = None
            self.contornos_info = []

    def predecir_variedades(self):
        if not self.modelo or not self.carpeta_salida or self.imagen_original is None:
            messagebox.showerror("Error", "Falta el modelo, la carpeta de salida o la imagen original.")
            return

        self.info_label.config(text="Prediciendo variedades...")
        self.root.update_idletasks()

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

            # Guardar temporalmente en la carpeta de salida
            temp_path = os.path.join(self.carpeta_salida, "imagen_anotada_temp.jpg")
            cv2.imwrite(temp_path, self.imagen_anotada)
            cv2.imshow("Hojas Anotadas", self.imagen_anotada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.info_label.config(text="Predicciones completadas. Use 'Descargar Imagen' para guardar.")
            self.boton_descargar.grid()

        except Exception as e:
            messagebox.showerror("Error", f"Error al predecir variedades: {str(e)}")
            self.info_label.config(text="Error al predecir variedades.")
            self.imagen_anotada = None

    def descargar_imagen(self):
        if self.imagen_anotada is None:
            messagebox.showerror("Error", "No hay imagen anotada para descargar.")
            return

        # Abrir diálogo para guardar la imagen
        ruta_guardado = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen anotada"
        )
        if ruta_guardado:
            escala = 0.8
            ancho = int(self.imagen_anotada.shape[1] * escala)
            alto = int(self.imagen_anotada.shape[0] * escala)
            imagen_reducida = cv2.resize(self.imagen_anotada, (ancho, alto), interpolation=cv2.INTER_AREA)

            cv2.imwrite(ruta_guardado, imagen_reducida)
            self.info_label.config(text=f"Imagen guardada en: {ruta_guardado}")
            messagebox.showinfo("Éxito", f"Imagen descargada en: {ruta_guardado}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentadorHojasApp(root)
    root.mainloop()