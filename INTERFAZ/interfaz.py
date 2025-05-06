import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from INTERFAZ.PROCESAMIENTO_HYPER.MASCARA_INDICE import procesar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.RECORTE_IMAGEN import recortar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.SEGMENTACION_2 import procesar_shapefile_y_extraer_datos
from INTERFAZ.PROCESAMIENTO_RGB.SEGMENTADOR_HOJAS import SegmentadorHojasApp
from PIL import Image, ImageTk
import os
import pandas as pd
import joblib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from INTERFAZ.PROCESAMIENTO_HYPER.RED_NEURONAL import comprobar_nuevos_datos, cargar_modelo_y_scaler


class InterfazApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TFM Alba Gómez")
        self.step = 0
        self.imagen_hdr = ""
        self.shapefile_path = ""
        self.imagen_recortada_path = ""
        self.excel_path = ""

        # Load the pre-trained model and scaler
        self.model, self.scaler = cargar_modelo_y_scaler()
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error",
                                 "No se pudieron cargar el modelo o el scaler. Verifica los archivos modelo_cnn.h5 y scaler.pkl.")
            self.root.quit()
            return

        self.container = tk.Frame(root, bg="#2E4A3D")  # Fondo verde oscuro
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

        # Cuadro blanco que engloba todo
        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        # Frame izquierdo para los textos y botones
        left_frame = tk.Frame(main_frame, bg="white")
        left_frame.pack(side="left", padx=20, pady=20, fill="y")

        # Frame derecho para la imagen
        right_frame = tk.Frame(main_frame, bg="white")
        right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

        # Contenido del frame izquierdo
        # Título principal
        tk.Label(left_frame,
                 text="Caracterización Geométrica y Espectral para la Clasificación\nde Variedades de Olivar utilizando Aprendizaje Automático",
                 font=("Helvetica", 14, "bold"), bg="white", justify="center").pack(pady=20)

        # Sección Procesamiento Hiperespectral
        tk.Label(left_frame, text="Procesamiento Hiperespectral", bg="#A9CBA4", font=("Helvetica", 12, "bold"),
                 relief="groove", bd=2, pady=5, padx=10).pack(fill="x", pady=5)
        tk.Label(left_frame,
                 text="Segmentación de hojas a partir de imagen hiperespectral (formato .hdr y .dat) y posterior predicción mediante IA.\n"
                      "Genera archivo shapefile con las hojas, archivo hiperespectral con el recorte y un Excel con los resultados.",
                 wraplength=400, justify="center", bg="white").pack(pady=5)
        tk.Button(left_frame, text="Iniciar Procesamiento", command=self.iniciar_procesamiento_hiperespectral,
                  bg="#d5f5e3", font=("Helvetica", 10), relief="groove", bd=2, width=25).pack(pady=10)

        # Sección Procesamiento RGB
        tk.Label(left_frame, text="Procesamiento RGB", bg="#A9CBA4", font=("Helvetica", 12, "bold"),
                 relief="groove", bd=2, pady=5, padx=10).pack(fill="x", pady=5)
        tk.Label(left_frame, text="Segmentación de hojas a partir de imagen JPG y posterior predicción mediante IA.\n"
                                  "Genera otra imagen con los resultados.",
                 wraplength=400, justify="center", bg="white").pack(pady=5)
        tk.Button(left_frame, text="Iniciar Procesamiento", command=self.iniciar_procesamiento_rgb,
                  bg="#d5f5e3", font=("Helvetica", 10), relief="groove", bd=2, width=25).pack(pady=10)

        # Imagen
        try:
            image_path = r"C:\Users\UJA\Desktop\programa\INTERFAZ\imagenFondo.png"
            img = Image.open(image_path)
            img = img.resize((300, 400), Image.Resampling.LANCZOS)  # Ajusta el tamaño según necesites
            photo = ImageTk.PhotoImage(img)
            image_label = tk.Label(right_frame, image=photo, bg="white")
            image_label.image = photo  # Mantener una referencia para evitar que se borre
            image_label.pack(expand=True)
        except Exception as e:
            tk.Label(right_frame, text="Error al cargar la imagen", bg="white", font=("Helvetica", 10, "italic")).pack(
                expand=True)

    def iniciar_procesamiento_hiperespectral(self):
        self.step = 1
        self.mostrar_paso_1()

    def iniciar_procesamiento_rgb(self):
        self.limpiar_contenedor()
        SegmentadorHojasApp(self.root)

    def mostrar_paso_1(self):
        self.limpiar_contenedor()
        ttk.Label(self.container, text="Paso 1: Generar Máscara de Vegetación").grid(row=0, column=0, columnspan=3,
                                                                                     pady=5)
        ttk.Label(self.container, text="Imagen hiperespectral (.hdr):").grid(row=1, column=0, pady=5)
        self.imagen_entry = ttk.Entry(self.container, width=50)
        self.imagen_entry.grid(row=1, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar",
                   command=lambda: self.imagen_entry.insert(0, self.seleccionar_archivo("hdr"))).grid(row=1, column=2,
                                                                                                      pady=5)
        ttk.Label(self.container, text="Carpeta de exportación:").grid(row=2, column=0, pady=5)
        self.exportacion_entry = ttk.Entry(self.container, width=50)
        self.exportacion_entry.grid(row=2, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar",
                   command=lambda: self.exportacion_entry.insert(0, self.seleccionar_archivo("dir"))).grid(row=2,
                                                                                                           column=2,
                                                                                                           pady=5)
        ttk.Label(self.container, text="Nombre del archivo (sin extensión):").grid(row=3, column=0, pady=5)
        self.nombre_entry = ttk.Entry(self.container, width=50)
        self.nombre_entry.grid(row=3, column=1, pady=5)
        self.var_raster = tk.BooleanVar()
        self.var_vector = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.container, text="Exportar como raster (.tiff)", variable=self.var_raster).grid(row=4,
                                                                                                            column=0,
                                                                                                            pady=5)
        ttk.Checkbutton(self.container, text="Exportar como vectorial (.shp)", variable=self.var_vector,
                        state="disabled").grid(row=4, column=1, pady=5)
        self.info_label_1 = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label_1.grid(row=5, column=0, columnspan=3, pady=5)
        ttk.Button(self.container, text="Siguiente", command=self.procesar_paso_1).grid(row=6, column=0, columnspan=3,
                                                                                        pady=10)

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
        ttk.Label(self.container, text=f"Usando shapefile: {self.shapefile_path}").grid(row=2, column=0, columnspan=3,
                                                                                        pady=5)
        ttk.Label(self.container, text="Carpeta para guardar imagen recortada:").grid(row=3, column=0, pady=5)
        self.recorte_entry = ttk.Entry(self.container, width=50)
        self.recorte_entry.grid(row=3, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar",
                   command=lambda: self.recorte_entry.insert(0, self.seleccionar_archivo("dir"))).grid(row=3, column=2,
                                                                                                       pady=5)
        self.info_label_2 = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label_2.grid(row=4, column=0, columnspan=3, pady=5)
        ttk.Button(self.container, text="Siguiente", command=self.procesar_paso_2).grid(row=5, column=0, columnspan=3,
                                                                                        pady=10)

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
        ttk.Label(self.container, text="Paso 3: Análisis Espectral y Predicción").grid(row=0, column=0, columnspan=3,
                                                                                       pady=5)
        ttk.Label(self.container, text=f"Usando imagen recortada: {self.imagen_recortada_path}").grid(row=1, column=0,
                                                                                                      columnspan=3,
                                                                                                      pady=5)
        ttk.Label(self.container, text=f"Usando shapefile: {self.shapefile_path}").grid(row=2, column=0, columnspan=3,
                                                                                        pady=5)
        ttk.Label(self.container, text="Divisiones por cuadrícula:").grid(row=3, column=0, pady=5)
        self.divisiones_entry = ttk.Entry(self.container, width=50)
        self.divisiones_entry.grid(row=3, column=1, pady=5)
        ttk.Label(self.container, text="Especie (opcional):").grid(row=4, column=0, pady=5)
        self.especie_entry = ttk.Entry(self.container, width=50)
        self.especie_entry.grid(row=4, column=1, pady=5)
        ttk.Label(self.container, text="Ruta para guardar archivo Excel:").grid(row=5, column=0, pady=5)
        self.excel_entry = ttk.Entry(self.container, width=50)
        self.excel_entry.grid(row=5, column=1, pady=5)
        ttk.Button(self.container, text="Seleccionar",
                   command=lambda: self.excel_entry.insert(0, self.seleccionar_archivo("excel"))).grid(row=5, column=2,
                                                                                                       pady=5)
        self.info_label_3 = ttk.Label(self.container, text="Esperando procesamiento...")
        self.info_label_3.grid(row=6, column=0, columnspan=3, pady=5)
        ttk.Button(self.container, text="Finalizar", command=self.procesar_paso_3).grid(row=7, column=0, columnspan=3,
                                                                                        pady=10)

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
            procesar_shapefile_y_extraer_datos(
                self.shapefile_path,
                self.imagen_recortada_path,
                divisiones,
                self.excel_path,
                especie=especie if especie else None
            )
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
                output_excel = os.path.splitext(self.excel_path)[0] + '_predicciones.xlsx'
                comprobar_nuevos_datos(self.model, data, self.scaler, output_excel=output_excel)
                messagebox.showinfo("Éxito", f"Predicciones completadas. Resultados guardados en: {output_excel}")
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