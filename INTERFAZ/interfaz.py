import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from INTERFAZ.PROCESAMIENTO_HYPER.MASCARA_INDICE import procesar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.RECORTE_IMAGEN import recortar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.SEGMENTACION_2 import procesar_shapefile_y_extraer_datos
from INTERFAZ.PROCESAMIENTO_RGB.SEGMENTADOR_HOJAS import SegmentadorHojas
from PIL import Image, ImageTk
import os
import pandas as pd
import joblib
import cv2

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

        # Variables para RGB
        self.rgb_step = 0
        self.rgb_imagen_path = ""
        self.rgb_carpeta_salida = ""
        self.rgb_hojas_guardadas = 0
        self.rgb_temp_path = ""
        self.segmentador = None

        # Load the pre-trained model and scaler for hiperespectral
        self.model, self.scaler = cargar_modelo_y_scaler()
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error",
                                 "No se pudieron cargar el modelo o el scaler. Verifica los archivos modelo_cnn.h5 y scaler.pkl.")
            self.root.quit()
            return

        # Inicializar el segmentador de hojas para RGB
        try:
            self.segmentador = SegmentadorHojas()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.root.quit()
            return

        self.container = tk.Frame(root, bg="#2E4A3D")  # Fondo verde oscuro
        self.container.pack(padx=10, pady=10, expand=True, fill="both")

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
        elif tipo == "jpg":
            return filedialog.askopenfilename(filetypes=[("Archivos JPG", "*.jpg")])
        elif tipo == "save_jpg":
            return filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])

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
            img = img.resize((300, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label = tk.Label(right_frame, image=photo, bg="white")
            image_label.image = photo
            image_label.pack(expand=True)
        except Exception as e:
            tk.Label(right_frame, text="Error al cargar la imagen", bg="white", font=("Helvetica", 10, "italic")).pack(
                expand=True)

    def iniciar_procesamiento_hiperespectral(self):
        self.step = 1
        self.mostrar_paso_1()

    def iniciar_procesamiento_rgb(self):
        self.rgb_step = 1
        self.rgb_imagen_path = ""
        self.rgb_carpeta_salida = ""
        self.rgb_hojas_guardadas = 0
        self.rgb_temp_path = ""
        self.mostrar_paso_rgb_1()

    def mostrar_paso_rgb_1(self):
        self.limpiar_contenedor()

        # Cuadro blanco que engloba todo
        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Títulos
        tk.Label(main_frame, text="Procesamiento RGB", font=("Helvetica", 14, "bold"), bg="white", justify="center").pack(pady=10)
        tk.Label(main_frame, text="Paso 1: Elección de Imagen", font=("Helvetica", 12), bg="white", justify="center").pack(pady=5)

        # Frame para los campos
        input_frame = tk.Frame(main_frame, bg="white")
        input_frame.pack(padx=20, pady=10, fill="x")

        # Campo para seleccionar imagen RGB
        tk.Label(input_frame, text="Seleccionar imagen RGB (.jpg):", bg="white").grid(row=0, column=0, pady=5, sticky="e")
        self.rgb_imagen_entry = ttk.Entry(input_frame, width=50)
        self.rgb_imagen_entry.grid(row=0, column=1, pady=5)
        tk.Button(input_frame, text="Buscar", command=self.seleccionar_imagen_rgb, bg="#d5f5e3", relief="groove", bd=2).grid(row=0, column=2, pady=5)

        # Campo para carpeta de salida
        tk.Label(input_frame, text="Carpeta de salida:", bg="white").grid(row=1, column=0, pady=5, sticky="e")
        self.rgb_salida_entry = ttk.Entry(input_frame, width=50)
        self.rgb_salida_entry.grid(row=1, column=1, pady=5)
        tk.Button(input_frame, text="Buscar", command=self.seleccionar_salida_rgb, bg="#d5f5e3", relief="groove", bd=2).grid(row=1, column=2, pady=5)

        # Campo para número inicial
        tk.Label(input_frame, text="N° inicial: (Por si hubiera más imágenes)", bg="white").grid(row=2, column=0, pady=5, sticky="e")
        self.rgb_inicio_entry = ttk.Entry(input_frame, width=10)
        self.rgb_inicio_entry.grid(row=2, column=1, pady=5, sticky="w")
        self.rgb_inicio_entry.insert(0, "1")

        # Sección de previsualización
        tk.Label(input_frame, text="Previsualización:", bg="white").grid(row=3, column=0, columnspan=3, pady=5)
        self.rgb_preview_frame = tk.Frame(input_frame, bg="white")
        self.rgb_preview_frame.grid(row=4, column=0, columnspan=3, pady=5)
        self.rgb_preview_label = tk.Label(self.rgb_preview_frame, bg="white")
        self.rgb_preview_label.pack(expand=True)

        # Botón Procesar
        self.rgb_procesar_btn = tk.Button(main_frame, text="Procesar", command=self.procesar_rgb_paso_1, bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_procesar_btn.pack(pady=20)

    def mostrar_paso_rgb_2(self):
        self.limpiar_contenedor()

        # Cuadro blanco que engloba todo
        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Títulos
        tk.Label(main_frame, text="Procesamiento RGB", font=("Helvetica", 14, "bold"), bg="white",
                 justify="center").pack(pady=10)
        tk.Label(main_frame, text="Paso 2: Resultados de Segmentación", font=("Helvetica", 12), bg="white",
                 justify="center").pack(pady=5)

        # Frame para los resultados
        result_frame = tk.Frame(main_frame, bg="white")
        result_frame.pack(padx=20, pady=10, fill="x")

        # Mostrar información
        tk.Label(result_frame, text=f"Imagen procesada: {self.rgb_imagen_path}", bg="white", wraplength=500).pack(
            pady=5)
        tk.Label(result_frame, text=f"Se han guardado {self.rgb_hojas_guardadas} hojas en: {self.rgb_carpeta_salida}",
                 bg="white", wraplength=500).pack(pady=5)

        # Sección de previsualización para los resultados
        tk.Label(result_frame, text="Previsualización de resultados:", bg="white").pack(pady=5)
        self.rgb_preview_frame = tk.Frame(result_frame, bg="white")
        self.rgb_preview_frame.pack(pady=5)
        self.rgb_preview_label = tk.Label(self.rgb_preview_frame, bg="white")  # Inicializar aquí
        self.rgb_preview_label.pack(expand=True)

        # Botón para predecir variedades
        self.rgb_predecir_btn = tk.Button(main_frame, text="Predecir Variedad", command=self.predecir_rgb_variedades,
                                          bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_predecir_btn.pack(pady=10)

        # Botón para descargar (inicialmente oculto)
        self.rgb_descargar_btn = tk.Button(main_frame, text="Descargar Imagen", command=self.descargar_rgb_imagen,
                                           bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_descargar_btn.pack(pady=5)
        self.rgb_descargar_btn.pack_forget()

        # Botón para volver a la pantalla principal
        tk.Button(main_frame, text="Volver al Inicio", command=self.volver_al_inicio_rgb, bg="#d5f5e3", relief="groove",
                  bd=2).pack(pady=10)

        # Si ya se ha predicho, mostrar la previsualización de la imagen anotada
        if self.rgb_temp_path:
            self.mostrar_previsualizacion_rgb_resultados()

    def seleccionar_imagen_rgb(self):
        ruta = self.seleccionar_archivo("jpg")
        if ruta:
            self.rgb_imagen_entry.delete(0, tk.END)
            self.rgb_imagen_entry.insert(0, ruta)
            self.mostrar_previsualizacion_rgb(ruta)

    def seleccionar_salida_rgb(self):
        ruta = self.seleccionar_archivo("dir")
        if ruta:
            self.rgb_salida_entry.delete(0, tk.END)
            self.rgb_salida_entry.insert(0, ruta)

    def mostrar_previsualizacion_rgb(self, ruta):
        try:
            img = Image.open(ruta)
            img = img.resize((300, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.rgb_preview_label.configure(image=photo)
            self.rgb_preview_label.image = photo
        except Exception as e:
            self.rgb_preview_label.configure(text="Error al cargar la previsualización")

    def mostrar_previsualizacion_rgb_resultados(self):
        if not hasattr(self, 'rgb_preview_label'):
            return  # Si no existe el label, no hacemos nada

        if self.rgb_temp_path and os.path.exists(self.rgb_temp_path):
            try:
                img = Image.open(self.rgb_temp_path)
                img = img.resize((300, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.rgb_preview_label.configure(image=photo)
                self.rgb_preview_label.image = photo
                # Asegurarse de que el botón de descarga aparezca
                if hasattr(self, 'rgb_descargar_btn'):
                    self.rgb_descargar_btn.pack()
            except Exception as e:
                self.rgb_preview_label.configure(text=f"Error al cargar la previsualización: {str(e)}")

    def procesar_rgb_paso_1(self):
        self.rgb_imagen_path = self.rgb_imagen_entry.get()
        self.rgb_carpeta_salida = self.rgb_salida_entry.get()
        try:
            inicio_num = int(self.rgb_inicio_entry.get())
        except ValueError:
            messagebox.showerror("Error", "El número inicial debe ser un entero válido.")
            return

        if not self.rgb_imagen_path or not self.rgb_carpeta_salida:
            messagebox.showerror("Error", "Debe seleccionar la imagen y la carpeta de salida.")
            return

        try:
            self.rgb_hojas_guardadas = self.segmentador.procesar_hojas(self.rgb_imagen_path, self.rgb_carpeta_salida, inicio_num)
            self.rgb_step = 2
            self.mostrar_paso_rgb_2()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predecir_rgb_variedades(self):
        try:
            self.rgb_temp_path = self.segmentador.predecir_variedades()
            # Verificar que la ruta temporal se haya establecido
            if not self.rgb_temp_path or not os.path.exists(self.rgb_temp_path):
                raise Exception("No se pudo generar la imagen anotada temporal.")

            # Leer la imagen reescalada
            imagen = cv2.imread(self.rgb_temp_path)
            if imagen is None:
                raise Exception("No se pudo cargar la imagen anotada para mostrar.")

            # Configurar la ventana con un tamaño fijo
            cv2.namedWindow("Hojas Anotadas", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Hojas Anotadas", 800, 600)  # Tamaño fijo de la ventana (800x600 píxeles)
            cv2.imshow("Hojas Anotadas", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Mostrar la previsualización y activar el botón de descarga
            self.mostrar_previsualizacion_rgb_resultados()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def descargar_rgb_imagen(self):
        ruta_guardado = self.seleccionar_archivo("save_jpg")
        if ruta_guardado:
            try:
                self.segmentador.descargar_imagen(ruta_guardado)
                messagebox.showinfo("Éxito", f"Imagen descargada en: {ruta_guardado}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def volver_al_inicio_rgb(self):
        self.rgb_step = 0
        self.rgb_imagen_path = ""
        self.rgb_carpeta_salida = ""
        self.rgb_hojas_guardadas = 0
        self.rgb_temp_path = ""
        self.mostrar_paso_0()

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