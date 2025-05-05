import pandas as pd
import os

def merge_excel_files(folder_path, output_filename="TODOS.xlsx"):
    all_dataframes = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.xlsx', '.xls')):
            file_path = os.path.join(folder_path, filename)

            try:
                df = pd.read_excel(file_path)
                all_dataframes.append(df)
                print(f"Archivo procesado: {filename}")
            except Exception as e:
                print(f"Error al procesar {filename}: {str(e)}")

    if not all_dataframes:
        print("No se encontraron archivos Excel en la carpeta especificada")
        return

    merged_df = pd.concat(all_dataframes, ignore_index=True)
    output_path = os.path.join(folder_path, output_filename)
    merged_df.to_excel(output_path, index=False)

    print(f"\nFusion completada! Archivo guardado como: {output_filename}")
    print(f"Total de filas combinadas: {len(merged_df)}")

if __name__ == "__main__":
    folder_path = "C:\\Users\\UJA\Desktop\programa\Resultados\excel_2\TODOS"
    merge_excel_files(folder_path)