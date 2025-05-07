import tkinter as tk
from tkinter import ttk, messagebox
from interfaz_hyper import InterfazHyper
from interfaz_rgb import InterfazRGB
from PIL import Image, ImageTk
import os

class InterfazApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TFM Alba Gómez")
        self.root.geometry("900x600")
        self.root.resizable(True, True)

        self.container = tk.Frame(root, bg="#2E4A3D")
        self.container.pack(padx=10, pady=10, expand=True, fill="both")

        self.mostrar_pantalla_principal()

    def limpiar_contenedor(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def mostrar_pantalla_principal(self):
        self.limpiar_contenedor()

        title_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        title_frame.pack(fill="x", padx=20, pady=(20, 0))

        tk.Label(title_frame,
                 text="Caracterización Geométrica y Espectral para la Clasificación\nde Variedades de Olivar utilizando Aprendizaje Automático",
                 font=("Helvetica", 14, "bold"), bg="white", justify="center", wraplength=600).pack(fill="x", pady=10)

        main_frame = tk.Frame(self.container, bg="white", bd=2, relief="groove")
        main_frame.pack(padx=20, pady=(0, 20), expand=True, fill="both")

        left_frame = tk.Frame(main_frame, bg="white")
        left_frame.pack(side="left", padx=40, pady=20, fill="y", expand=True)

        right_frame = tk.Frame(main_frame, bg="white")
        right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

        tk.Label(left_frame, text="Procesamiento Hiperespectral", bg="#A9CBA4", font=("Helvetica", 12, "bold"),
                 relief="groove", bd=2, pady=5, padx=10).pack(fill="x", pady=5)
        tk.Label(left_frame,
                 text="Segmentación de hojas a partir de imagen hiperespectral (formato .hdr y .dat) y posterior predicción mediante IA.\n"
                      "Genera archivo shapefile con las hojas, archivo hiperespectral con el recorte y un Excel con los resultados.",
                 wraplength=300, justify="center", bg="white").pack(pady=5)
        tk.Button(left_frame, text="Iniciar Procesamiento", command=self.iniciar_procesamiento_hiperespectral,
                  bg="#d5f5e3", font=("Helvetica", 10), relief="groove", bd=2, width=25).pack(pady=10)

        tk.Label(left_frame, text="Procesamiento RGB", bg="#A9CBA4", font=("Helvetica", 12, "bold"),
                 relief="groove", bd=2, pady=5, padx=10).pack(fill="x", pady=5)
        tk.Label(left_frame, text="Segmentación de hojas a partir de imagen JPG y posterior predicción mediante IA.\n"
                                  "Genera otra imagen con los resultados.",
                 wraplength=300, justify="center", bg="white").pack(pady=5)
        tk.Button(left_frame, text="Iniciar Procesamiento", command=self.iniciar_procesamiento_rgb,
                  bg="#d5f5e3", font=("Helvetica", 10), relief="groove", bd=2, width=25).pack(pady=10)

        try:
            image_path = r"C:\Users\UJA\Desktop\programa\INTERFAZ\imagenFondo.png"
            img = Image.open(image_path)
            new_width = 200
            new_height = 300
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label = tk.Label(right_frame, image=photo, bg="white")
            image_label.image = photo
            image_label.pack(expand=True, fill="both", padx=10, pady=10)

            def resize_image(event):
                try:
                    frame_width = event.width
                    frame_height = event.height
                    if frame_width <= 40 or frame_height <= 40:
                        return
                    img_resized = Image.open(image_path)
                    img_ratio = img_resized.width / img_resized.height
                    new_width = min(frame_width - 40, int((frame_height - 40) * img_ratio))
                    new_height = int(new_width / img_ratio)
                    if new_width > 0 and new_height > 0:
                        img_resized = img_resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        new_photo = ImageTk.PhotoImage(img_resized)
                        image_label.configure(image=new_photo)
                        image_label.image = new_photo
                except Exception as e:
                    print(f"Error al redimensionar la imagen: {str(e)}")

            right_frame.bind("<Configure>", resize_image)

        except Exception as e:
            tk.Label(right_frame, text=f"Error al cargar la imagen: {str(e)}", bg="white",
                     font=("Helvetica", 10, "italic")).pack(expand=True, fill="both", padx=10, pady=10)

    def iniciar_procesamiento_hiperespectral(self):
        self.limpiar_contenedor()
        InterfazHyper(self.container, self.root, self.mostrar_pantalla_principal)

    def iniciar_procesamiento_rgb(self):
        self.limpiar_contenedor()
        InterfazRGB(self.container, self.root, self.mostrar_pantalla_principal)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazApp(root)
    root.mainloop()