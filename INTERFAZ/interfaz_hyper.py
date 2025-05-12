import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from INTERFAZ.PROCESAMIENTO_HYPER.MASCARA_INDICE import procesar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.RECORTE_IMAGEN import recortar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.SEGMENTACION_2 import procesar_shapefile_y_extraer_datos
from INTERFAZ.PROCESAMIENTO_HYPER.RED_NEURONAL import cargar_modelo_y_scaler, comprobar_nuevos_datos
import os
import pandas as pd


def seleccionar_archivo(tipo):
    if tipo == "hdr":
        return filedialog.askopenfilename(filetypes=[("Archivos .hdr", "*.hdr")])
    elif tipo == "dir":
        return filedialog.askdirectory()
    elif tipo == "excel":
        return filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Archivos Excel", "*.xlsx")])


class InterfazHyper:
    def __init__(self, container, root, volver_inicio_callback):
        self.next_button = None
        self.var_vector = None
        self.var_raster = None
        self.nombre_entry = None
        self.exportacion_entry = None
        self.imagen_entry = None
        self.subtitle_label = None
        self.title_label = None
        self.container = container
        self.root = root
        self.volver_inicio_callback = volver_inicio_callback
        self.step = 1
        self.imagen_hdr = ""
        self.shapefile_path = ""
        self.imagen_recortada_path = ""
        self.excel_path = ""
        self.model, self.scaler = cargar_modelo_y_scaler()

        self.mostrar_paso_1()

    def limpiar_contenedor(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def mostrar_paso_1(self):
        self.limpiar_contenedor()

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        self.title_label = tk.Label(main_frame, text="Procesamiento Hiperespectral", font=("Helvetica", 14, "bold"),
                                    bg="white", justify="center")
        self.title_label.pack(pady=(10, 2), fill="x")

        self.subtitle_label = tk.Label(main_frame, text="Paso 1: Generar Máscara de Vegetación", font=("Helvetica", 10),
                                       bg="white", justify="center", fg="#2E4A3D")
        self.subtitle_label.pack(pady=(0, 5), fill="x")

        input_frame = tk.Frame(main_frame, bg="white")
        input_frame.pack(padx=20, pady=10, expand=True, fill="both")

        tk.Label(input_frame, text="Seleccionar imagen (.hdr):", bg="white", font=("Helvetica", 10, "bold")).grid(row=0,
                                                                                                                  column=0,
                                                                                                                  pady=5,
                                                                                                                  sticky="e")
        self.imagen_entry = ttk.Entry(input_frame, width=50)
        self.imagen_entry.grid(row=0, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Buscar", command=lambda: self.imagen_entry.insert(0, seleccionar_archivo("hdr")),
                  bg="#d5f5e3", relief="groove", bd=2).grid(row=0, column=2, pady=5)

        tk.Label(input_frame, text="Carpeta de salida:", bg="white", font=("Helvetica", 10, "bold")).grid(row=1,
                                                                                                          column=0,
                                                                                                          pady=5,
                                                                                                          sticky="e")
        self.exportacion_entry = ttk.Entry(input_frame, width=50)
        self.exportacion_entry.grid(row=1, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Buscar",
                  command=lambda: self.exportacion_entry.insert(0, seleccionar_archivo("dir")),
                  bg="#d5f5e3", relief="groove", bd=2).grid(row=1, column=2, pady=5)

        tk.Label(input_frame, text="Nombre exportación (sin extensión):", bg="white",
                 font=("Helvetica", 10, "bold")).grid(row=2, column=0, pady=5, sticky="e")
        self.nombre_entry = ttk.Entry(input_frame, width=30)
        self.nombre_entry.grid(row=2, column=1, columnspan=2, pady=5, sticky="ew")

        self.var_raster = tk.BooleanVar()
        self.var_vector = tk.BooleanVar(value=True)

        check_frame = tk.Frame(input_frame, bg="white")
        check_frame.grid(row=3, column=1, columnspan=2, pady=5, sticky="w")
        tk.Checkbutton(check_frame, text="Exportar como ráster (.tiff)", variable=self.var_raster,
                       bg="white", activebackground="white", anchor="w", fg="#2E4A3D").pack(anchor="w")
        tk.Checkbutton(check_frame, text="Exportar como vectorial (.shp)", variable=self.var_vector,
                       bg="white", activebackground="white", anchor="w", state="disabled").pack(anchor="w")

        for i in range(3):
            input_frame.grid_columnconfigure(i, weight=1)
        input_frame.grid_rowconfigure(4, weight=1)

        self.next_button = tk.Button(main_frame, text="Siguiente", command=self.procesar_paso_1,
                                     bg="#d5f5e3", relief="groove", bd=2)
        self.next_button.pack(pady=20)  # Sin fill="x" ni padx para no ocupar todo el ancho

        def resize_elements(event):
            try:
                main_width = main_frame.winfo_width()
                if main_width > 40:
                    new_font_size_title = max(14, int(main_width / 50))
                    new_font_size_subtitle = max(12, int(main_width / 60))
                    self.title_label.config(font=("Helvetica", new_font_size_title, "bold"))
                    self.subtitle_label.config(font=("Helvetica", new_font_size_subtitle))
            except Exception as e:
                print(f"Error al redimensionar elementos: {str(e)}")

        main_frame.bind("<Configure>", resize_elements)

    def procesar_paso_1(self):
        self.imagen_hdr = self.imagen_entry.get()
        exportacion = self.exportacion_entry.get()
        nombre = self.nombre_entry.get()
        if not self.imagen_hdr or not exportacion or not nombre:
            messagebox.showerror("Error", "Por favor, complete todos los campos.")
            return
        try:
            procesar_imagen(self.imagen_hdr, self.var_raster.get(), True, exportacion, nombre)
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar la máscara: {str(e)}")
            return
        self.shapefile_path = os.path.normpath(os.path.join(exportacion, nombre + '.shp'))
        if not os.path.exists(self.shapefile_path):
            messagebox.showerror("Error", "No se pudo generar el archivo shapefile.")
            return
        self.step = 2
        self.mostrar_paso_2()

    def mostrar_paso_2(self):
        self.limpiar_contenedor()

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        self.title_label = tk.Label(main_frame, text="Procesamiento Hiperespectral", font=("Helvetica", 14, "bold"),
                                    bg="white", justify="center")
        self.title_label.pack(pady=(10, 2), fill="x")

        self.subtitle_label = tk.Label(main_frame, text="Paso 2: Recorte de Imagen", font=("Helvetica", 10),
                                       bg="white", justify="center", fg="#2E4A3D")
        self.subtitle_label.pack(pady=(0, 10), fill="x")

        input_frame = tk.Frame(main_frame, bg="white")
        input_frame.pack(padx=20, pady=10, expand=True, fill="both")

        imagen_var = tk.StringVar(value=self.imagen_hdr)
        shapefile_var = tk.StringVar(value=self.shapefile_path)

        tk.Label(input_frame, text="Imagen (.hdr):", bg="white", font=("Helvetica", 10, "bold")).grid(row=0, column=0,
                                                                                                      pady=5,
                                                                                                      sticky="e")
        tk.Entry(input_frame, textvariable=imagen_var, state="readonly", relief="flat", bg="white",
                 fg="#2E4A3D", font=("Helvetica", 9)).grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        tk.Label(input_frame, text="Shapefile:", bg="white", font=("Helvetica", 10, "bold")).grid(row=1, column=0,
                                                                                                  pady=5, sticky="e")
        tk.Entry(input_frame, textvariable=shapefile_var, state="readonly", relief="flat", bg="white",
                 fg="#2E4A3D", font=("Helvetica", 9)).grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)

        tk.Label(input_frame, text="Carpeta de salida:", bg="white", font=("Helvetica", 10, "bold")).grid(row=2,
                                                                                                          column=0,
                                                                                                          pady=5,
                                                                                                          sticky="e")
        self.recorte_entry = ttk.Entry(input_frame)
        self.recorte_entry.grid(row=2, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Seleccionar",
                  command=lambda: self.recorte_entry.insert(0, seleccionar_archivo("dir")),
                  bg="#d5f5e3", relief="groove", bd=2).grid(row=2, column=2, pady=5, padx=(5, 0))

        tk.Label(input_frame, text="Nombre del archivo de salida (sin extensión):", bg="white",
                 font=("Helvetica", 10, "bold")).grid(row=3, column=0, pady=5, sticky="e")
        self.recorte_nombre_entry = ttk.Entry(input_frame, width=30)
        self.recorte_nombre_entry.grid(row=3, column=1, columnspan=2, pady=5, sticky="ew")

        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(2, weight=0)

        # Create progress bar and status label, initially hidden
        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar", troughcolor="white", background="#2E4A3D")
        self.progress_bar = ttk.Progressbar(main_frame, mode="indeterminate", length=300,
                                            style="Green.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)
        self.status_label = ttk.Label(main_frame, text="Ejecutando recorte...", font=("Helvetica", 10),
                                      background="white", foreground="#2E4A3D", relief="flat")
        self.status_label.pack(pady=5)
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()

        self.next_button = tk.Button(main_frame, text="Siguiente", command=self.procesar_paso_2,
                                     bg="#d5f5e3", relief="groove", bd=2)
        self.next_button.pack(pady=20)

        def resize_elements(event):
            try:
                main_width = main_frame.winfo_width()
                if main_width > 40:
                    new_font_size_title = max(14, int(main_width / 50))
                    new_font_size_subtitle = max(12, int(main_width / 60))
                    self.title_label.config(font=("Helvetica", new_font_size_title, "bold"))
                    self.subtitle_label.config(font=("Helvetica", new_font_size_subtitle))
            except Exception as e:
                print(f"Error al redimensionar elementos: {str(e)}")

        main_frame.bind("<Configure>", resize_elements)

    def procesar_paso_2(self):
        import threading

        recorte_dir = self.recorte_entry.get()
        recorte_nombre = self.recorte_nombre_entry.get()
        if not recorte_dir or not recorte_nombre:
            messagebox.showerror("Error", "Seleccione la carpeta y el nombre para guardar la imagen recortada.")
            return

        self.next_button.config(state="disabled")
        self.progress_bar.pack(pady=5)
        self.status_label.pack(pady=5)
        self.progress_bar.start()
        self.root.update_idletasks()

        def run_recorte():
            try:
                recortar_imagen(self.shapefile_path, self.imagen_hdr, recorte_dir, output_filename=recorte_nombre)
                self.root.after(0, self._complete_recorte)
            except Exception as e:
                self.root.after(0, lambda: self._error_recorte(str(e)))

        def thread_task():
            thread = threading.Thread(target=run_recorte)
            thread.daemon = True
            thread.start()

        self._complete_recorte = lambda: (
            self.progress_bar.stop(),
            self.progress_bar.pack_forget(),
            self.status_label.pack_forget(),
            self.next_button.config(state="normal"),
            setattr(self, 'imagen_recortada_path',
                    os.path.normpath(os.path.join(recorte_dir, recorte_nombre + ".dat"))),
            messagebox.showerror("Error", "No se pudo generar la imagen recortada.") if not os.path.exists(
                self.imagen_recortada_path) else (
                setattr(self, 'step', 3),
                self.mostrar_paso_3()
            )
        )

        self._error_recorte = lambda error: (
            self.progress_bar.stop(),
            self.progress_bar.pack_forget(),
            self.status_label.pack_forget(),
            self.next_button.config(state="normal"),
            messagebox.showerror("Error", f"Error al recortar la imagen: {error}")
        )

        thread_task()

    def mostrar_paso_3(self):
        self.limpiar_contenedor()

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        self.title_label = tk.Label(main_frame, text="Procesamiento Hiperespectral", font=("Helvetica", 14, "bold"),
                                    bg="white", justify="center")
        self.title_label.pack(pady=(10, 2), fill="x")

        self.subtitle_label = tk.Label(main_frame, text="Paso 3: Análisis Espectral y Predicción",
                                       font=("Helvetica", 10),
                                       bg="white", justify="center", fg="#2E4A3D")
        self.subtitle_label.pack(pady=(0, 10), fill="x")

        input_frame = tk.Frame(main_frame, bg="white")
        input_frame.pack(padx=20, pady=10, expand=True, fill="both")

        imagen_var = tk.StringVar(value=self.imagen_recortada_path)
        shapefile_var = tk.StringVar(value=self.shapefile_path)

        tk.Label(input_frame, text="Imagen recortada:", bg="white", font=("Helvetica", 10, "bold")).grid(row=0,
                                                                                                         column=0,
                                                                                                         pady=5,
                                                                                                         sticky="e")
        tk.Entry(input_frame, textvariable=imagen_var, state="readonly", relief="flat", bg="white",
                 fg="#2E4A3D", font=("Helvetica", 9)).grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        tk.Label(input_frame, text="Shapefile:", bg="white", font=("Helvetica", 10, "bold")).grid(row=1, column=0,
                                                                                                  pady=5, sticky="e")
        tk.Entry(input_frame, textvariable=shapefile_var, state="readonly", relief="flat", bg="white",
                 fg="#2E4A3D", font=("Helvetica", 9)).grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)

        tk.Label(input_frame, text="Divisiones por cuadrícula:", bg="white", font=("Helvetica", 10, "bold")).grid(row=2,
                                                                                                                  column=0,
                                                                                                                  pady=5,
                                                                                                                  sticky="e")
        self.divisiones_entry = ttk.Entry(input_frame, width=50)
        self.divisiones_entry.grid(row=2, column=1, columnspan=2, pady=5, sticky="ew")

        tk.Label(input_frame, text="Especie (opcional):", bg="white", font=("Helvetica", 10, "bold")).grid(row=3,
                                                                                                           column=0,
                                                                                                           pady=5,
                                                                                                           sticky="e")
        self.especie_entry = ttk.Entry(input_frame, width=50)
        self.especie_entry.grid(row=3, column=1, columnspan=2, pady=5, sticky="ew")

        tk.Label(input_frame, text="Ruta para guardar archivo Excel:", bg="white", font=("Helvetica", 10, "bold")).grid(
            row=4, column=0, pady=5, sticky="e")
        self.excel_entry = ttk.Entry(input_frame, width=50)
        self.excel_entry.grid(row=4, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Seleccionar",
                  command=lambda: self.excel_entry.insert(0, seleccionar_archivo("excel")),
                  bg="#d5f5e3", relief="groove", bd=2).grid(row=4, column=2, pady=5, padx=(5, 0))

        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(2, weight=0)

        self.info_label_3 = ttk.Label(main_frame, text="Esperando procesamiento...")
        self.info_label_3.pack(pady=5)

        self.next_button = tk.Button(main_frame, text="Finalizar", command=self.procesar_paso_3,
                                     bg="#d5f5e3", relief="groove", bd=2)
        self.next_button.pack(pady=20)

        def resize_elements(event):
            try:
                main_width = main_frame.winfo_width()
                if main_width > 40:
                    new_font_size_title = max(14, int(main_width / 50))
                    new_font_size_subtitle = max(12, int(main_width / 60))
                    self.title_label.config(font=("Helvetica", new_font_size_title, "bold"))
                    self.subtitle_label.config(font=("Helvetica", new_font_size_subtitle))
            except Exception as e:
                print(f"Error al redimensionar elementos: {str(e)}")

        main_frame.bind("<Configure>", resize_elements)

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
            return

        if os.path.exists(self.excel_path):
            self.info_label_3.config(text=f"Archivo Excel generado en: {self.excel_path}")
            self.info_label_3.config(text="Realizando predicción con el modelo preentrenado...")
            self.root.update_idletasks()
            try:
                data = pd.read_excel(self.excel_path)
                output_excel = os.path.splitext(self.excel_path)[0] + '_predicciones.xlsx'
                comprobar_nuevos_datos(self.model, data, self.scaler, output_excel=output_excel)

                predicciones_df = pd.read_excel(output_excel, sheet_name="Predicciones por Hoja")
                conteo_especies = predicciones_df['Especie Predicha'].value_counts()
                total_pic = conteo_especies.get('PIC', 0)
                total_no_pic = conteo_especies.get('No PIC', 0)

                especie_mayoritaria = 'Picual' if total_pic > total_no_pic else 'No Picual'

                resultado_texto = (
                    f"La especie predicha es: {especie_mayoritaria}\n"
                    f"Hay {total_pic} casos de Picual y {total_no_pic} casos de No Picual\n"
                    f"Archivo Excel generado en: {output_excel}"
                )
                self.info_label_3.config(text=resultado_texto)

                messagebox.showinfo("Éxito", f"Predicciones completadas. Resultados guardados en: {output_excel}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al realizar la predicción: {str(e)}")
                return

            respuesta = messagebox.askyesno("Volver al inicio", "¿Desea volver a la pantalla principal?")
            if respuesta:
                self.volver_inicio_callback()
            else:
                self.root.quit()
        else:
            messagebox.showerror("Error", "No se pudo generar el archivo Excel.")

    def mostrar_paso_3(self):
        self.limpiar_contenedor()

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        self.title_label = tk.Label(main_frame, text="Procesamiento Hiperespectral", font=("Helvetica", 14, "bold"),
                                    bg="white", justify="center")
        self.title_label.pack(pady=(10, 2), fill="x")

        self.subtitle_label = tk.Label(main_frame, text="Paso 3: Análisis Espectral y Predicción",
                                       font=("Helvetica", 10),
                                       bg="white", justify="center", fg="#2E4A3D")
        self.subtitle_label.pack(pady=(0, 10), fill="x")

        input_frame = tk.Frame(main_frame, bg="white")
        input_frame.pack(padx=20, pady=10, expand=True, fill="both")

        # Variables de texto para mostrar rutas
        imagen_var = tk.StringVar(value=self.imagen_recortada_path)
        shapefile_var = tk.StringVar(value=self.shapefile_path)

        tk.Label(input_frame, text="Imagen recortada:", bg="white", font=("Helvetica", 10, "bold")).grid(row=0,
                                                                                                         column=0,
                                                                                                         pady=5,
                                                                                                         sticky="e")
        tk.Entry(input_frame, textvariable=imagen_var, state="readonly", relief="flat", bg="white",
                 fg="#2E4A3D", font=("Helvetica", 9)).grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        tk.Label(input_frame, text="Shapefile:", bg="white", font=("Helvetica", 10, "bold")).grid(row=1, column=0,
                                                                                                  pady=5, sticky="e")
        tk.Entry(input_frame, textvariable=shapefile_var, state="readonly", relief="flat", bg="white",
                 fg="#2E4A3D", font=("Helvetica", 9)).grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)

        tk.Label(input_frame, text="Divisiones por cuadrícula:", bg="white", font=("Helvetica", 10, "bold")).grid(row=2,
                                                                                                                  column=0,
                                                                                                                  pady=5,
                                                                                                                  sticky="e")
        self.divisiones_entry = ttk.Entry(input_frame, width=50)
        self.divisiones_entry.grid(row=2, column=1, columnspan=2, pady=5, sticky="ew")

        tk.Label(input_frame, text="Especie (opcional):", bg="white", font=("Helvetica", 10, "bold")).grid(row=3,
                                                                                                           column=0,
                                                                                                           pady=5,
                                                                                                           sticky="e")
        self.especie_entry = ttk.Entry(input_frame, width=50)
        self.especie_entry.grid(row=3, column=1, columnspan=2, pady=5, sticky="ew")

        tk.Label(input_frame, text="Ruta para guardar archivo Excel:", bg="white", font=("Helvetica", 10, "bold")).grid(
            row=4, column=0, pady=5, sticky="e")
        self.excel_entry = ttk.Entry(input_frame, width=50)
        self.excel_entry.grid(row=4, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Seleccionar",
                  command=lambda: self.excel_entry.insert(0, seleccionar_archivo("excel")),
                  bg="#d5f5e3", relief="groove", bd=2).grid(row=4, column=2, pady=5, padx=(5, 0))

        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(2, weight=0)

        self.info_label_3 = ttk.Label(main_frame, text="Esperando procesamiento...")
        self.info_label_3.pack(pady=5)

        self.next_button = tk.Button(main_frame, text="Finalizar", command=self.procesar_paso_3,
                                     bg="#d5f5e3", relief="groove", bd=2)
        self.next_button.pack(pady=20)

        def resize_elements(event):
            try:
                main_width = main_frame.winfo_width()
                if main_width > 40:
                    new_font_size_title = max(14, int(main_width / 50))
                    new_font_size_subtitle = max(12, int(main_width / 60))
                    self.title_label.config(font=("Helvetica", new_font_size_title, "bold"))
                    self.subtitle_label.config(font=("Helvetica", new_font_size_subtitle))
            except Exception as e:
                print(f"Error al redimensionar elementos: {str(e)}")

        main_frame.bind("<Configure>", resize_elements)

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
            return

        if os.path.exists(self.excel_path):
            self.info_label_3.config(text=f"Archivo Excel generado en: {self.excel_path}")
            self.info_label_3.config(text="Realizando predicción con el modelo preentrenado...")
            self.root.update_idletasks()
            try:
                data = pd.read_excel(self.excel_path)
                output_excel = os.path.splitext(self.excel_path)[0] + '_predicciones.xlsx'
                comprobar_nuevos_datos(self.model, data, self.scaler, output_excel=output_excel)

                # Leer el Excel generado para contar especies
                predicciones_df = pd.read_excel(output_excel, sheet_name="Predicciones por Hoja")
                conteo_especies = predicciones_df['Especie Predicha'].value_counts()
                total_pic = conteo_especies.get('PIC', 0)
                total_no_pic = conteo_especies.get('No PIC', 0)

                # Determinar especie mayoritaria
                especie_mayoritaria = 'Picual' if total_pic > total_no_pic else 'No Picual'

                # Mostrar resultados en pantalla
                resultado_texto = (
                    f"La especie predicha es: {especie_mayoritaria}\n"
                    f"Hay {total_pic} casos de Picual y {total_no_pic} casos de No Picual\n"
                    f"Archivo Excel generado en: {output_excel}"
                )
                self.info_label_3.config(text=resultado_texto)

                messagebox.showinfo("Éxito", f"Predicciones completadas. Resultados guardados en: {output_excel}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al realizar la predicción: {str(e)}")
                return

            respuesta = messagebox.askyesno("Volver al inicio", "¿Desea volver a la pantalla principal?")
            if respuesta:
                self.volver_inicio_callback()
            else:
                self.root.quit()
        else:
            messagebox.showerror("Error", "No se pudo generar el archivo Excel.")