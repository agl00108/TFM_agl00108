import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from INTERFAZ.PROCESAMIENTO_HYPER.MASCARA_INDICE import procesar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.RECORTE_IMAGEN import recortar_imagen
from INTERFAZ.PROCESAMIENTO_HYPER.SEGMENTACION_2 import procesar_shapefile_y_extraer_datos
from INTERFAZ.PROCESAMIENTO_HYPER.RED_NEURONAL import cargar_modelo_y_scaler, comprobar_nuevos_datos
import os
import pandas as pd
import re
import threading


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
                                       font=("Helvetica", 10), bg="white", justify="center", fg="#2E4A3D")
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

        tk.Label(input_frame, text="Especie:", bg="white", font=("Helvetica", 10, "bold")).grid(row=3,
                                                                                                column=0,
                                                                                                pady=5,
                                                                                                sticky="e")
        self.especie_entry = ttk.Entry(input_frame, width=50)
        self.especie_entry.grid(row=3, column=1, columnspan=2, pady=5, sticky="ew")

        tk.Label(input_frame, text="Carpeta de salida:", bg="white", font=("Helvetica", 10, "bold")).grid(row=4,
                                                                                                          column=0,
                                                                                                          pady=5,
                                                                                                          sticky="e")
        self.excel_dir_entry = ttk.Entry(input_frame, width=50)
        self.excel_dir_entry.grid(row=4, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Seleccionar",
                  command=lambda: self.excel_dir_entry.insert(0, seleccionar_archivo("dir")),
                  bg="#d5f5e3", relief="groove", bd=2).grid(row=4, column=2, pady=5, padx=(5, 0))

        tk.Label(input_frame, text="Nombre del archivo Excel (sin extensión):", bg="white",
                 font=("Helvetica", 10, "bold")).grid(row=5, column=0, pady=5, sticky="e")
        self.excel_nombre_entry = ttk.Entry(input_frame, width=30)
        self.excel_nombre_entry.grid(row=5, column=1, columnspan=2, pady=5, sticky="ew")

        self.results_text = tk.Text(main_frame, height=6, width=50, state="disabled", wrap="word", bg="white",
                                    fg="#2E4A3D", relief="flat", font=("Helvetica", 10))
        self.results_text.pack(pady=10, fill="x", padx=20)
        self.results_text.tag_configure("center", justify="center")
        self.results_text.tag_configure("bold", font=("Helvetica", 10, "bold"))

        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(2, weight=0)

        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar", troughcolor="white", background="#2E4A3D")
        self.progress_bar_3 = ttk.Progressbar(main_frame, mode="indeterminate", length=300,
                                              style="Green.Horizontal.TProgressbar")
        self.progress_bar_3.pack(pady=2)  # Reduced pady to move progress bar higher
        self.status_label_3 = ttk.Label(main_frame, text="Ejecutando segmentación y prediciendo variedad...",
                                        font=("Helvetica", 10), background="white", foreground="#2E4A3D", relief="flat")
        self.status_label_3.pack(pady=2)  # Reduced pady for consistency
        self.progress_bar_3.pack_forget()
        self.status_label_3.pack_forget()

        button_frame = tk.Frame(main_frame, bg="white")
        button_frame.pack(pady=10)
        self.next_button = tk.Button(button_frame, text="Finalizar", command=self.procesar_paso_3,
                                     bg="#d5f5e3", relief="groove", bd=2)
        self.next_button.pack(side="left", padx=5)
        self.volver_button = tk.Button(button_frame, text="Volver al Inicio", command=self.volver_inicio_callback,
                                       bg="#d5f5e3", relief="groove", bd=2, state="disabled")
        self.volver_button.pack(side="left", padx=5)

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
        especie = self.especie_entry.get().strip()
        excel_dir = self.excel_dir_entry.get()
        excel_nombre = self.excel_nombre_entry.get()

        if not divisiones or not excel_dir or not excel_nombre:
            messagebox.showerror("Error",
                                 "Complete todos los campos obligatorios (Divisiones, Carpeta de salida y Nombre del archivo Excel).")
            return

        if not especie:
            messagebox.showerror("Error",
                                 "El campo Especie es obligatorio. Si no conoce la especie, ingrese 'No Sabe'.")
            return

        try:
            divisiones = int(divisiones)
            if divisiones <= 0:
                raise ValueError("El número de divisiones debe ser mayor que cero.")
        except ValueError:
            messagebox.showerror("Error", "El número de divisiones debe ser un número entero positivo.")
            return

        # Asignar None si especie es "None" (case-insensitive)
        especie_param = None if especie.lower() == "none" else especie
        print(f"Depuración: Pasando especie_param = {especie_param}")  # Depuración

        excel_nombre = re.sub(r'[<>:"/\\|?*]', '_', excel_nombre.strip())
        self.excel_path = os.path.normpath(os.path.join(excel_dir, excel_nombre + ".xlsx"))
        self.next_button.config(state="disabled")
        self.volver_button.config(state="disabled")
        self.progress_bar_3.pack(pady=2)
        self.status_label_3.pack(pady=2)
        self.progress_bar_3.start()
        self.root.update_idletasks()

        def run_procesamiento():
            try:
                procesar_shapefile_y_extraer_datos(
                    self.shapefile_path,
                    self.imagen_recortada_path,
                    divisiones,
                    self.excel_path,
                    especie=especie_param
                )
                if os.path.exists(self.excel_path):
                    print(f"Depuración: Excel generado en {self.excel_path}")  # Depuración
                    data = pd.read_excel(self.excel_path)
                    output_excel = os.path.splitext(self.excel_path)[0] + '_predicciones.xlsx'
                    comprobar_nuevos_datos(self.model, data, self.scaler, output_excel=output_excel)

                    try:
                        mayoritarias_df = pd.read_excel(output_excel, sheet_name="Especies Mayoritarias por Hoja")
                        if 'Especie Mayoritaria' in mayoritarias_df.columns:
                            conteo_especies = mayoritarias_df['Especie Mayoritaria'].value_counts()
                            total_pic = conteo_especies.get('PIC', 0)
                            total_no_pic = conteo_especies.get('No PIC', 0)
                            especie_mayoritaria = 'Picual' if total_pic > total_no_pic else 'No Picual'
                        else:
                            total_pic = 0
                            total_no_pic = 0
                            especie_mayoritaria = "No disponible (sin especie proporcionada)"
                            print("Depuración: No se encontró la columna 'Especie Mayoritaria', asumiendo sin especie.")
                    except (ValueError, KeyError) as e:
                        print(f"Depuración: Error al leer 'Especies Mayoritarias por Hoja': {str(e)}")
                        total_pic = 0
                        total_no_pic = 0
                        especie_mayoritaria = "No disponible (error en datos)"

                    self.root.after(0, lambda: (
                        self.results_text.config(state="normal"),
                        self.results_text.delete("1.0", "end"),
                        self.results_text.insert("end", f"Hojas de Picual: {total_pic}\n", "center"),
                        self.results_text.insert("end", f"Hojas de No Picual: {total_no_pic}\n", "center"),
                        self.results_text.insert("end", f"Especie predicha: {especie_mayoritaria}\n",
                                                 ("bold", "center")),
                        self.results_text.insert("end", f"Archivo Excel: {output_excel}\n", "center"),
                        (self.results_text.insert("end", f"Especie proporcionada: {especie_param}\n", "center")
                         if especie_param else None),
                        self.results_text.config(state="disabled"),
                        self._complete_procesamiento()
                    ))
                else:
                    self.root.after(0, lambda: self._error_procesamiento("No se pudo generar el archivo Excel."))
            except Exception as e:
                error_msg = f"Error al realizar el análisis espectral o predicción: {str(e)}"
                print(f"Depuración: Excepción capturada - {error_msg}")  # Depuración
                self.root.after(0, lambda: self._error_procesamiento(error_msg))

        def thread_task():
            thread = threading.Thread(target=run_procesamiento)
            thread.daemon = True
            thread.start()

        self._complete_procesamiento = lambda: (
            self.progress_bar_3.stop(),
            self.progress_bar_3.pack_forget(),
            self.status_label_3.pack_forget(),
            self.next_button.config(state="normal"),
            self.volver_button.config(state="normal")
        )

        self._error_procesamiento = lambda error: (
            self.progress_bar_3.stop(),
            self.progress_bar_3.pack_forget(),
            self.status_label_3.pack_forget(),
            self.next_button.config(state="normal"),
            self.volver_button.config(state="normal"),
            self.results_text.config(state="normal"),
            self.results_text.delete("1.0", "end"),
            self.results_text.insert("end", f"Error: {error}\n", "center"),
            self.results_text.config(state="disabled"),
            messagebox.showerror("Error", error)
        )

        thread_task()