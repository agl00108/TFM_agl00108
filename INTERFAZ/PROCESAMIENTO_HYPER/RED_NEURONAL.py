import numpy as np
import openpyxl
import pandas as pd
import tensorflow as tf
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
import time
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from openpyxl.styles import Font, Alignment

# Cargar los mejores hiperparámetros desde el archivo JSON
with open('best_params.json', 'r') as f:
    best_params = json.load(f)

# Definir las constantes para los parámetros del modelo
POOL_SIZE = best_params['pool_size']
BATCH_SIZE = best_params['batch_size']
EPOCHS = 50
PATIENCE = best_params['patience']

# Función para establecer la semilla
def establecer_semilla(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Función para mostrar los resultados del modelo
def mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test):
    print(f"Evaluación del modelo {nombre_modelo}")
    print(f"Pérdida: {loss}")
    print(f"Precisión: {accuracy}")
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    # Personalización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                annot_kws={"size": 16, "weight": "bold"}, linewidths=1.5, linecolor="black")
    plt.title(f'Confusion Matrix - {nombre_modelo}', fontsize=16, weight='bold')
    plt.xlabel('Predictions', fontsize=14, weight='bold')
    plt.ylabel('Real Values', fontsize=14, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Función para preparar los datos
def preparar_datos(df):
    # Imprimir número de columnas para depuración
    print(f"Número de columnas en el DataFrame de entrenamiento: {df.shape[1]}")

    # Filtrar filas donde ninguna banda (columnas 1 a 187, índices 3 a 189) tenga un valor > 0.35
    bandas = df.iloc[:, 3:190]
    filtro = (bandas <= 0.35).all(axis=1)
    df_filtrado = df[filtro].copy()
    print(f"Filas originales: {len(df)}, Filas después de filtrar: {len(df_filtrado)}")

    # Preparar características (X) y etiquetas (y) a partir del DataFrame filtrado
    X = df_filtrado.iloc[:, 3:].values  # Usar todas las bandas desde la columna 3
    y = df_filtrado['Especie'].apply(lambda x: 0 if x == 'PIC' else 1).values

    # Verificar la forma de X
    print(f"Forma de X antes de dividir: {X.shape}")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Contamos cuántos 0 y 1 hay en y_train antes de SMOTE
    train_counts = Counter(y_train)
    print(f'En y_train antes de SMOTE, 0: {train_counts[0]}, 1: {train_counts[1]}')

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)  # Convertir a 2D para SMOTE
    X_train_smote, y_train_smote = smote.fit_resample(X_train_2d, y_train)

    # Restaurar la forma 3D después de SMOTE
    X_train = X_train_smote.reshape(X_train_smote.shape[0], X_train.shape[1], 1)

    # Actualizar y_train con los datos balanceados
    y_train = y_train_smote

    # Contamos cuántos 0 y 1 hay en y_train después de SMOTE
    train_counts_smote = Counter(y_train)
    print(f'En y_train después de SMOTE, 0: {train_counts_smote[0]}, 1: {train_counts_smote[1]}')

    # Contamos cuántos 0 y 1 hay en y_test
    test_counts = Counter(y_test)
    print(f'En y_test, 0: {test_counts[0]}, 1: {test_counts[1]}')

    # Escalar los datos
    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)

    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1]))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Imprimir formas finales
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler

# Función principal para ejecutar la red neuronal CNN
def ejecutar_cnn(df):
    establecer_semilla()
    start_time = time.time()

    X_train, X_test, y_train, y_test, scaler = preparar_datos(df)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(
        Conv1D(filters=int(best_params['filters_1']), kernel_size=int(best_params['kernel_size']), activation='elu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(
        Conv1D(filters=int(best_params['filters_2']), kernel_size=int(best_params['kernel_size']), activation='elu'))
    model.add(
        Conv1D(filters=int(best_params['filters_3']), kernel_size=int(best_params['kernel_size']), activation='elu'))
    model.add(
        Conv1D(filters=int(best_params['filters_4']), kernel_size=int(best_params['kernel_size']), activation='elu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Flatten())
    model.add(Dense(int(best_params['ff_dim']), activation='relu'))
    model.add(Dropout(best_params['dropout_rate']))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Primera gráfica: cambiar estilo y color
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='purple', linestyle='dotted')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='pink', linestyle='dashed')
    plt.title('Model Accuracy Across Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='darkblue', marker='o')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='orange', marker='s')
    plt.title('Training vs Test Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='crimson', linestyle='-.')
    plt.plot(history.history['val_loss'], label='Test Loss', color='darkgreen', linestyle='--')
    plt.title('Training vs Test Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Binary Crossentropy)', fontsize=12)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    nombre_modelo = "CNN"
    mostrar_resultados(y_test, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Tiempo de ejecución: {execution_time:.2f} segundos')

    return model, scaler

def comprobar_nuevos_datos(model, nuevos_datos, scaler, output_excel="predicciones_por_hoja.xlsx"):
    """
    Preprocesa nuevos datos, realiza predicciones y genera un Excel con la especie mayoritaria por hoja.
    """
    print(f"Número de columnas en el DataFrame de nuevos datos: {nuevos_datos.shape[1]}")

    bandas_nuevas = nuevos_datos.iloc[:, 3:190]
    filtro_nuevas = (bandas_nuevas <= 0.35).all(axis=1)
    nuevos_datos_filtrados = nuevos_datos[filtro_nuevas].copy()
    print(f"Filas originales (nuevos datos): {len(nuevos_datos)}, Filas después de filtrar: {len(nuevos_datos_filtrados)}")

    if len(nuevos_datos_filtrados) == 0:
        print("Error: No hay datos después de filtrar. Verifica los datos de entrada.")
        return

    X_nuevos = nuevos_datos_filtrados.iloc[:, 3:].values
    y_nuevos = nuevos_datos_filtrados['Especie'].apply(lambda x: 0 if x == 'PIC' else 1).values
    hojas = nuevos_datos_filtrados['Hoja'].values

    print(f"Forma de X_nuevos antes de reshape: {X_nuevos.shape}")
    X_nuevos = X_nuevos.reshape((X_nuevos.shape[0], X_nuevos.shape[1]))  # Asegurar 2D antes del escalado
    X_nuevos = scaler.transform(X_nuevos)
    X_nuevos = X_nuevos.reshape((X_nuevos.shape[0], X_nuevos.shape[1], 1)).astype(np.float32)
    print(f"Forma de X_nuevos después de escalado y reshape: {X_nuevos.shape}")
    print(f"Tipo de X_nuevos: {type(X_nuevos)}, dtype: {X_nuevos.dtype}")

    # Realizar predicciones
    y_pred_prob = model.predict(X_nuevos)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    y_pred_especie = ['PIC' if pred == 0 else 'No PIC' for pred in y_pred]

    print("\nPredicciones para los nuevos datos (especies):")
    print(y_pred_especie)

    print("\nValores reales:")
    print(y_nuevos)

    print("\nInforme de clasificación para los nuevos datos:")
    print(classification_report(y_nuevos, y_pred))

    cm = confusion_matrix(y_nuevos, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                annot_kws={"size": 16, "weight": "bold"}, linewidths=1.5, linecolor="black")
    plt.title('Confusion Matrix - New Data')
    plt.xlabel('Predictions')
    plt.ylabel('Real Values')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    predicciones_por_hoja = pd.DataFrame({
        'Hoja': hojas,
        'Especie_Predicha': y_pred_especie
    })

    especie_mayoritaria_por_hoja = {}
    for hoja in predicciones_por_hoja['Hoja'].unique():
        especies_hoja = predicciones_por_hoja[predicciones_por_hoja['Hoja'] == hoja]['Especie_Predicha']
        conteo = Counter(especies_hoja)
        especie_mayoritaria = conteo.most_common(1)[0][0]
        especie_mayoritaria_por_hoja[hoja] = especie_mayoritaria

    wb = openpyxl.Workbook()
    wb.remove(wb['Sheet'])

    for hoja, especie in especie_mayoritaria_por_hoja.items():
        ws = wb.create_sheet(title=str(hoja))
        ws['A1'] = 'Hoja'
        ws['B1'] = 'Especie Mayoritaria'
        ws['A1'].font = Font(bold=True)
        ws['B1'].font = Font(bold=True)
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['B1'].alignment = Alignment(horizontal='center')
        ws['A2'] = hoja
        ws['B2'] = especie
        ws['A2'].alignment = Alignment(horizontal='center')
        ws['B2'].alignment = Alignment(horizontal='center')
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 20

    wb.save(output_excel)
    print(f"\nArchivo Excel generado: {output_excel}")


# Ejemplo de ejecución
if __name__ == "__main__":
    # Cargar los datos
    data = pd.read_excel('../../Resultados/excel/TODOS/TODOS.xlsx')
    nuevos_datos = pd.read_excel('../../Resultados/PICUAL/Picual_4/jjj/defseefwer.xlsx')

    # Verificar número de columnas
    print(f"Número de columnas en TODOS.xlsx: {data.shape[1]}")
    print(f"Número de columnas en defseefwer.xlsx: {nuevos_datos.shape[1]}")

    # Ajustar el número de bandas si es necesario
    n_bandas = min(data.shape[1] - 3, nuevos_datos.shape[1] - 3)  # Usar el menor número de bandas
    print(f"Número de bandas a usar: {n_bandas}")

    # Recortar las columnas en ambos DataFrames para usar el mismo número de bandas
    data = data.iloc[:, :3 + n_bandas]
    nuevos_datos = nuevos_datos.iloc[:, :3 + n_bandas]

    # Ejecutar el modelo CNN
    model, scaler = ejecutar_cnn(data)

    # Comprobar nuevos datos
    comprobar_nuevos_datos(model, nuevos_datos, scaler)