import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from INTERFAZ.PROCESAMIENTO_RGB.SEGMENTADOR_HOJAS import SegmentadorHojas
from PIL import Image, ImageTk
import os

class InterfazRGB:
    def __init__(self, container, root, volver_inicio_callback):
        self.container = container
        self.root = root
        self.volver_inicio_callback = volver_inicio_callback  # Callback para volver al inicio
        self.rgb_step = 1
        self.rgb_imagen_path = ""
        self.rgb_carpeta_salida = ""
        self.rgb_hojas_guardadas = 0
        self.rgb_temp_path = ""
        self.segmentador = SegmentadorHojas()

        self.mostrar_paso_rgb_1()

    def limpiar_contenedor(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def seleccionar_archivo(self, tipo):
        if tipo == "jpg":
            return filedialog.askopenfilename(filetypes=[("Archivos JPG", "*.jpg")])
        elif tipo == "dir":
            return filedialog.askdirectory()
        elif tipo == "save_jpg":
            return filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])

    def mostrar_paso_rgb_1(self):
        self.limpiar_contenedor()

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        tk.Label(main_frame, text="Procesamiento RGB", font=("Helvetica", 14, "bold"), bg="white", justify="center").pack(pady=10)
        tk.Label(main_frame, text="Paso 1: Elección de Imagen", font=("Helvetica", 12), bg="white", justify="center").pack(pady=5)

        input_frame = tk.Frame(main_frame, bg="white")
        input_frame.pack(padx=20, pady=10, expand=True, fill="both")

        tk.Label(input_frame, text="Seleccionar imagen RGB (.jpg):", bg="white", font=("Helvetica", 10, "bold")).grid(row=0, column=0, pady=5, sticky="e")
        self.rgb_imagen_entry = ttk.Entry(input_frame, width=50)
        self.rgb_imagen_entry.grid(row=0, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Buscar", command=self.seleccionar_imagen_rgb, bg="#d5f5e3", relief="groove", bd=2).grid(row=0, column=2, pady=5)

        tk.Label(input_frame, text="Carpeta de salida:", bg="white", font=("Helvetica", 10, "bold")).grid(row=1, column=0, pady=5, sticky="e")
        self.rgb_salida_entry = ttk.Entry(input_frame, width=50)
        self.rgb_salida_entry.grid(row=1, column=1, pady=5, sticky="ew")
        tk.Button(input_frame, text="Buscar", command=self.seleccionar_salida_rgb, bg="#d5f5e3", relief="groove", bd=2).grid(row=1, column=2, pady=5)

        tk.Label(input_frame, text="N° inicial:", bg="white", font=("Helvetica", 10, "bold")).grid(row=2, column=0, pady=5, sticky="e")
        self.rgb_inicio_entry = ttk.Entry(input_frame, width=10)
        self.rgb_inicio_entry.grid(row=2, column=1, pady=5, sticky="w")
        self.rgb_inicio_entry.insert(0, "1")

        input_frame.grid_columnconfigure(1, weight=1)

        tk.Label(input_frame, text="Previsualización:", bg="white", font=("Helvetica", 10, "bold")).grid(row=3, column=0, columnspan=3, pady=5)
        self.rgb_preview_frame = tk.Frame(input_frame, bg="white")
        self.rgb_preview_frame.grid(row=4, column=0, columnspan=3, pady=5, sticky="nsew")
        self.rgb_preview_label = tk.Label(self.rgb_preview_frame, bg="white")
        self.rgb_preview_label.pack(expand=True, fill="both")

        input_frame.grid_rowconfigure(4, weight=1)

        self.rgb_procesar_btn = tk.Button(main_frame, text="Procesar", command=self.procesar_rgb_paso_1, bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_procesar_btn.pack(pady=20)

    def mostrar_paso_rgb_2(self):
        self.limpiar_contenedor()

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        tk.Label(main_frame, text="Procesamiento RGB", font=("Helvetica", 14, "bold"), bg="white", justify="center").pack(pady=10)
        tk.Label(main_frame, text="Paso 2: Resultados de Segmentación", font=("Helvetica", 12), bg="white", justify="center").pack(pady=5)

        result_frame = tk.Frame(main_frame, bg="white")
        result_frame.pack(padx=20, pady=10, expand=True, fill="both")

        tk.Label(result_frame, text=f"Imagen procesada: {self.rgb_imagen_path}", bg="white", font=("Helvetica", 10, "bold"), wraplength=500).pack(pady=5)
        tk.Label(result_frame, text=f"Se han guardado {self.rgb_hojas_guardadas} hojas en: {self.rgb_carpeta_salida}", bg="white", wraplength=500).pack(pady=5)

        tk.Label(result_frame, text="Previsualización de resultados:", bg="white", font=("Helvetica", 10, "bold")).pack(pady=5)
        self.rgb_preview_frame = tk.Frame(result_frame, bg="white")
        self.rgb_preview_frame.pack(pady=5, expand=True, fill="both")
        self.rgb_preview_label = tk.Label(self.rgb_preview_frame, bg="white")
        self.rgb_preview_label.pack(expand=True, fill="both")

        self.rgb_predecir_btn = tk.Button(main_frame, text="Predecir Variedad", command=self.predecir_rgb_variedades, bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_predecir_btn.pack(pady=10)

        self.rgb_descargar_btn = tk.Button(main_frame, text="Descargar Imagen", command=self.descargar_rgb_imagen, bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_descargar_btn.pack(pady=5)
        self.rgb_descargar_btn.pack_forget()

        self.rgb_volver_btn = tk.Button(main_frame, text="Volver al Inicio", command=self.volver_al_inicio_rgb, bg="#d5f5e3", relief="groove", bd=2)
        self.rgb_volver_btn.pack(pady=10)

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
            frame_width = self.rgb_preview_frame.winfo_width()
            frame_height = self.rgb_preview_frame.winfo_height()
            if frame_width <= 1 or frame_height <= 1:
                frame_width, frame_height = 300, 200
            img = img.resize((frame_width - 10, frame_height - 10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.rgb_preview_label.configure(image=photo)
            self.rgb_preview_label.image = photo
        except Exception as e:
            self.rgb_preview_label.configure(text="Error al cargar la previsualización")

    def mostrar_previsualizacion_rgb_resultados(self):
        if not hasattr(self, 'rgb_preview_label'):
            messagebox.showerror("Error", "No se pudo encontrar el área de previsualización.")
            return

        if self.rgb_temp_path and os.path.exists(self.rgb_temp_path):
            try:
                img = Image.open(self.rgb_temp_path)
                frame_width = self.rgb_preview_frame.winfo_width()
                frame_height = self.rgb_preview_frame.winfo_height()
                if frame_width <= 1 or frame_height <= 1:
                    frame_width, frame_height = 300, 200
                img = img.resize((frame_width - 10, frame_height - 10), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.rgb_preview_label.configure(image=photo)
                self.rgb_preview_label.image = photo
                self.rgb_preview_label.bind("<Button-1>", self.mostrar_imagen_grande)
                if hasattr(self, 'rgb_descargar_btn'):
                    self.rgb_descargar_btn.pack()
            except Exception as e:
                self.rgb_preview_label.configure(text=f"Error al cargar la previsualización: {str(e)}")

    def mostrar_imagen_grande(self, event):
        if not self.rgb_temp_path or not os.path.exists(self.rgb_temp_path):
            messagebox.showerror("Error", "No se puede cargar la imagen completa.")
            return

        try:
            ventana_grande = tk.Toplevel(self.root)
            ventana_grande.title("Imagen Anotada Completa")
            img = Image.open(self.rgb_temp_path)
            max_size = (800, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(ventana_grande, image=photo)
            label.image = photo
            label.pack(padx=10, pady=10)
            tk.Button(ventana_grande, text="Cerrar", command=ventana_grande.destroy, bg="#d5f5e3", relief="groove", bd=2).pack(pady=5)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la imagen completa: {str(e)}")

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
            if not self.rgb_temp_path or not os.path.exists(self.rgb_temp_path):
                raise Exception("No se pudo generar la imagen anotada temporal.")
            self.mostrar_previsualizacion_rgb_resultados()
            if hasattr(self, 'rgb_predecir_btn'):
                self.rgb_predecir_btn.pack_forget()
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
        self.volver_inicio_callback()  # Llamar al callback para volver al inicio