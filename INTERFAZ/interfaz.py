import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from INTERFAZ.PROCESAMIENTO_HYPER.MASCARA_INDICE import procesar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.RECORTE_IMAGEN import recortar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.SEGMENTACION_2 import procesar_shapefile_y_extraer_datos
from INTERFAZ.PROCESAMIENTO_RGB.SEGMENTADOR_HOJAS import SegmentadorHojasApp
import os
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from INTERFAZ.PROCESAMIENTO_HYPER.RED_NEURONAL import comprobar_nuevos_datos

class InterfazApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes")
        self.step = 0
        self.imagen_hdr = ""
        self.shapefile_path = ""
        self.imagen_recortada_path = ""
        self.excel_path = ""

        self.container = ttk.Frame(root)
        self.container.pack(padx=10, pady=10)

        self.mostrar_paso_0()

    def limpiar_contenedor(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def seleccionar_archivo(self, tipo):
        if tipo == "hdr":
            return filedialog.askopenfilename(filetypes=[("Archivos .hdr", "*.hdr")])
        elif tipo == "dir":
            return filedialog.askdirectory()
        elif tipo == "excel":
            return filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Archivos Excel", "*.xlsx")])

    def mostrar_paso_0(self):
        self.limpiar_contenedor()
        ttk.Label(self.container, text="Seleccione el tipo de procesamiento:").grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Button(self.container, text="Procesamiento Hiperespectral", command=self.iniciar_procesamiento_hiperespectral).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.container, text="Procesamiento RGB", command=self.iniciar_procesamiento_rgb).grid(row=1, column=1, padx=5, pady=5)

    def iniciar_procesamiento_hiperespectral(self):
        self.step = 1
        self.mostrar_paso_1()

    def iniciar_procesamiento_rgb(self):
        self.limpiar_contenedor()
        SegmentadorHojasApp(self.root)

    def mostrar_paso_1(self):
        self.limpiar_contenedor()
        ttk.Label(self.container, text="Paso 1: Generar Máscara de Vegetación").grid(row=0, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text="Imagen hiperespectral (.hdr):").grid(row=1, column=0, pady=5)
        self.imagen_entry = ttk.Entry(self.container, width=50)
        self.imagen_entry.grid(row=1, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar", command=lambda: self.imagen_entry.insert(0, self.seleccionar_archivo("hdr"))).grid(row=1, column=2, pady=5)
        ttk.Label(self.container, text="Carpeta de exportación:").grid(row=2, column=0, pady=5)
        self.exportacion_entry = ttk.Entry(self.container, width=50)
        self.exportacion_entry.grid(row=2, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar", command=lambda: self.exportacion_entry.insert(0, self.seleccionar_archivo("dir"))).grid(row=2, column=2, pady=5)
        ttk.Label(self.container, text="Nombre del archivo (sin extensión):").grid(row=3, column=0, pady=5)
        self.nombre_entry = ttk.Entry(self.container, width=50)
        self.nombre_entry.grid(row=3, column=1, pady=5)
        self.var_raster = tk.BooleanVar()
        self.var_vector = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.container, text="Exportar como raster (.tiff)", variable=self.var_raster).grid(row=4, column=0, pady=5)
        ttk.Checkbutton(self.container, text="Exportar como vectorial (.shp)", variable=self.var_vector, state="disabled").grid(row=4, column=1, pady=5)
        self.info_label_1 = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label_1.grid(row=5, column=0, columnspan=3, pady=5)
        ttk.Button(self.container, text="Siguiente", command=self.procesar_paso_1).grid(row=6, column=0, columnspan=3, pady=10)

    def procesar_paso_1(self):
        self.imagen_hdr = self.imagen_entry.get()
        exportacion = self.exportacion_entry.get()
        nombre = self.nombre_entry.get()
        if not self.imagen_hdr or not exportacion or not nombre:
            messagebox.showerror("Error", "Por favor, complete todos los campos.")
            return
        self.info_label_1.config(text="Generando máscara de vegetación...")
        self.root.update_idletasks()
        try:
            procesar_imagen(self.imagen_hdr, self.var_raster.get(), True, exportacion, nombre)
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar la máscara: {str(e)}")
            self.info_label_1.config(text="Error al generar la máscara.")
            return
        self.shapefile_path = os.path.normpath(os.path.join(exportacion, nombre + '.shp'))
        if not os.path.exists(self.shapefile_path):
            messagebox.showerror("Error", "No se pudo generar el archivo shapefile.")
            self.info_label_1.config(text="Error: No se encontró el archivo shapefile.")
            return
        self.info_label_1.config(text=f"Archivo shapefile generado en: {self.shapefile_path}")
        self.step = 2
        self.mostrar_paso_2()

    def mostrar_paso_2(self):
        self.limpiar_contenedor()
        ttk.Label(self.container, text="Paso 2: Recorte de Imagen").grid(row=0, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text=f"Usando imagen: {self.imagen_hdr}").grid(row=1, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text=f"Usando shapefile: {self.shapefile_path}").grid(row=2, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text="Carpeta para guardar imagen recortada:").grid(row=3, column=0, pady=5)
        self.recorte_entry = ttk.Entry(self.container, width=50)
        self.recorte_entry.grid(row=3, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar", command=lambda: self.recorte_entry.insert(0, self.seleccionar_archivo("dir"))).grid(row=3, column=2, pady=5)
        self.info_label_2 = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label_2.grid(row=4, column=0, columnspan=3, pady=5)
        ttk.Button(self.container, text="Siguiente", command=self.procesar_paso_2).grid(row=5, column=0, columnspan=3, pady=10)

    def procesar_paso_2(self):
        recorte_dir = self.recorte_entry.get()
        if not recorte_dir:
            messagebox.showerror("Error", "Seleccione la carpeta para guardar la imagen recortada.")
            return
        self.info_label_2.config(text="Procesando recorte de imagen...")
        self.root.update_idletasks()
        try:
            recortar_imagen(self.shapefile_path, self.imagen_hdr, recorte_dir)
        except Exception as e:
            messagebox.showerror("Error", f"Error al recortar la imagen: {str(e)}")
            self.info_label_2.config(text="Error al recortar la imagen.")
            return
        self.imagen_recortada_path = os.path.normpath(os.path.join(recorte_dir, "imagen_recortada.dat"))
        if not os.path.exists(self.imagen_recortada_path):
            messagebox.showerror("Error", "No se pudo generar la imagen recortada.")
            self.info_label_2.config(text="Error: No se encontró la imagen recortada.")
            return
        self.info_label_2.config(text=f"Imagen recortada guardada en: {self.imagen_recortada_path}")
        self.step = 3
        self.mostrar_paso_3()

    def mostrar_paso_3(self):
        self.limpiar_contenedor()
        ttk.Label(self.container, text="Paso 3: Análisis Espectral").grid(row=0, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text=f"Usando imagen recortada: {self.imagen_recortada_path}").grid(row=1, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text=f"Usando shapefile: {self.shapefile_path}").grid(row=2, column=0, columnspan=3, pady=5)
        ttk.Label(self.container, text="Divisiones por cuadrícula:").grid(row=3, column=0, pady=5)
        self.divisiones_entry = ttk.Entry(self.container, width=50)
        self.divisiones_entry.grid(row=3, column=1, pady=5)
        ttk.Label(self.container, text="Especie (opcional):").grid(row=4, column=0, pady=5)
        self.especie_entry = ttk.Entry(self.container, width=50)
        self.especie_entry.grid(row=4, column=1, pady=5)
        ttk.Label(self.container, text="Ruta para guardar archivo Excel:").grid(row=5, column=0, pady=5)
        self.excel_entry = ttk.Entry(self.container, width=50)
        self.excel_entry.grid(row=5, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar", command=lambda: self.excel_entry.insert(0, self.seleccionar_archivo("excel"))).grid(row=5, column=2, pady=5)
        self.info_label_3 = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label_3.grid(row=6, column=0, columnspan=3, pady=5)
        ttk.Button(self.container, text="Finalizar", command=self.procesar_paso_3).grid(row=7, column=0, columnspan=3, pady=10)

    def procesar_paso_3(self):
        divisiones = self.divisiones_entry.get()
        especie = self.especie_entry.get()
        self.excel_path = self.excel_entry.get()

        if not divisiones or not self.excel_path:
            messagebox.showerror("Error", "Complete todos los campos obligatorios (excepto especie).")
            return

        try:
            divisiones = int(divisiones)
            if divisiones <= 0:
                raise ValueError("El número de divisiones debe ser mayor que cero.")
        except ValueError:
            messagebox.showerror("Error", "El número de divisiones debe ser un número entero positivo.")
            return

        self.info_label_3.config(text="Procesando análisis espectral...")
        self.root.update_idletasks()
        try:
            procesar_shapefile_y_extraer_datos(self.shapefile_path, self.imagen_recortada_path, divisiones, self.excel_path,
                                               especie=especie if especie else None)
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar el análisis espectral: {str(e)}")
            self.info_label_3.config(text="Error al realizar el análisis espectral.")
            return

        if os.path.exists(self.excel_path):
            self.info_label_3.config(text=f"Archivo Excel generado en: {self.excel_path}")
            self.info_label_3.config(text="Realizando predicción con el modelo preentrenado...")
            self.root.update_idletasks()
            try:
                data = pd.read_excel(self.excel_path)
                resultados = comprobar_nuevos_datos(data)
                resultados_path = os.path.splitext(self.excel_path)[0] + '_predicciones.xlsx'
                resultados.to_excel(resultados_path, index=False)
                messagebox.showinfo("Éxito", f"Predicciones completadas. Resultados guardados en: {resultados_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al realizar la predicción: {str(e)}")
                self.info_label_3.config(text="Error al realizar la predicción.")
                return

            respuesta = messagebox.askyesno("Volver al inicio", "¿Desea volver a la pantalla principal?")
            if respuesta:
                self.step = 0
                self.imagen_hdr = ""
                self.shapefile_path = ""
                self.imagen_recortada_path = ""
                self.excel_path = ""
                self.mostrar_paso_0()
            else:
                self.root.quit()
        else:
            messagebox.showerror("Error", "No se pudo generar el archivo Excel.")
            self.info_label_3.config(text="Error: No se encontró el archivo Excel.")

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazApp(root)
    root.mainloop()