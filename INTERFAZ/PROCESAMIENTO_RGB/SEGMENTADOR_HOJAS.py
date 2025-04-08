import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class SegmentadorHojasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentador de Hojas (RGB)")

        self.container = ttk.Frame(root)
        self.container.pack(padx=10, pady=10)

        # --- Interfaz Gráfica ---
        # Selección de la imagen
        ttk.Label(self.container, text="Seleccionar imagen RGB (.jpg):").grid(row=0, column=0, pady=5)
        self.imagen_entry = ttk.Entry(self.container, width=50)
        self.imagen_entry.grid(row=0, column=1, pady=5)
        ttk.Button(self.container, text="Buscar", command=self.seleccionar_imagen).grid(row=0, column=2, pady=5)

        # Selección de la carpeta de salida
        ttk.Label(self.container, text="Carpeta de salida:").grid(row=1, column=0, pady=5)
        self.salida_entry = ttk.Entry(self.container, width=50)
        self.salida_entry.grid(row=1, column=1, pady=5)
        ttk.Button(self.container, text="Buscar", command=self.seleccionar_salida).grid(row=1, column=2, pady=5)

        # Entrada para el número inicial de la hoja
        ttk.Label(self.container, text="Número inicial de la hoja:").grid(row=2, column=0, pady=5)
        self.inicio_entry = ttk.Entry(self.container, width=10)
        self.inicio_entry.grid(row=2, column=1, pady=5, sticky="w")
        self.inicio_entry.insert(0, "1")

        # Mensaje informativo
        self.info_label = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label.grid(row=3, column=0, columnspan=3, pady=5)

        # Botón para procesar
        ttk.Button(self.container, text="Procesar", command=self.procesar_hojas).grid(row=4, column=0, columnspan=3, pady=10)

    def seleccionar_imagen(self):
        """Abre un diálogo para seleccionar la imagen RGB."""
        ruta = filedialog.askopenfilename(filetypes=[("Archivos JPG", "*.jpg")])
        if ruta:
            self.imagen_entry.delete(0, tk.END)
            self.imagen_entry.insert(0, ruta)

    def seleccionar_salida(self):
        """Abre un diálogo para seleccionar la carpeta de salida."""
        ruta = filedialog.askdirectory()
        if ruta:
            self.salida_entry.delete(0, tk.END)
            self.salida_entry.insert(0, ruta)

    def procesar_hojas(self):
        """Procesa la imagen RGB para segmentar y guardar las hojas."""
        imagen_path = self.imagen_entry.get()
        carpeta_salida = self.salida_entry.get()
        try:
            inicio_num = int(self.inicio_entry.get())
        except ValueError:
            messagebox.showerror("Error", "El número inicial debe ser un entero válido.")
            return

        # Validaciones
        if not imagen_path or not carpeta_salida:
            messagebox.showerror("Error", "Debe seleccionar la imagen y la carpeta de salida.")
            return

        if not os.path.exists(imagen_path):
            messagebox.showerror("Error", "La imagen seleccionada no existe.")
            return

        # Crear la carpeta de salida si no existe
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)

        self.info_label.config(text="Procesando imagen...")
        self.root.update_idletasks()

        try:
            # Cargar la imagen
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                raise ValueError("No se pudo cargar la imagen.")

            # Convertir a escala de grises
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            # Aplicar desenfoque para reducir ruido
            desenfocada = cv2.GaussianBlur(gris, (7, 7), 0)

            # Binarización con umbral automático (Otsu)
            _, binaria = cv2.threshold(desenfocada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Operaciones morfológicas para cerrar huecos
            kernel = np.ones((5, 5), np.uint8)
            morfologica = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Encontrar contornos
            contornos, _ = cv2.findContours(morfologica, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Contador para nombrar las hojas, usando el valor inicial del usuario
            j = inicio_num - 1  # Se resta 1 porque se incrementa antes de guardar
            hojas_guardadas = 0

            # Procesar cada contorno
            for contorno in contornos:
                x, y, w, h = cv2.boundingRect(contorno)
                if w > 20 and h > 20:  # Filtrar objetos pequeños
                    hoja = imagen[y:y + h, x:x + w]
                    hoja_filename = f'hoja_{j + 1}.jpg'
                    j += 1
                    hoja_path = os.path.join(carpeta_salida, hoja_filename)
                    cv2.imwrite(hoja_path, hoja)
                    hojas_guardadas += 1

            self.info_label.config(text=f"Se han guardado {hojas_guardadas} hojas en: {carpeta_salida}")
            messagebox.showinfo("Éxito", f"Proceso completado. Se guardaron {hojas_guardadas} hojas.")

        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
            self.info_label.config(text="Error al procesar la imagen.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentadorHojasApp(root)
    root.mainloop()