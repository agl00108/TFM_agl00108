import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LayerNormalization
import time
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Definir las constantes para los parámetros del modelo
POOL_SIZE = 3
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8

# Mejores parámetros obtenidos PICUAL
best_params = {
    'dropout_rate': 0.35483380605435355,
    'ff_dim': 59.99681409968073,
    'filters_1': 14.017624178380828,
    'filters_2': 29.7391777595531,
    'filters_3': 60.67157395019049,
    'filters_4': 111.31192330114831,
    'kernel_size': 4.880006256681035
}


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
    # Filtrar filas donde ninguna banda (columnas 1 a 187, índices 3 a 189 en el DataFrame) tenga un valor > 0.35
    bandas = df.iloc[:, 3:190]
    filtro = (bandas <= 0.35).all(axis=1)
    df_filtrado = df[filtro].copy()
    print(f"Filas originales: {len(df)}, Filas después de filtrar: {len(df_filtrado)}")

    # Preparar características (X) y etiquetas (y) a partir del DataFrame filtrado
    X = df_filtrado.iloc[:, 3:].values
    y = df_filtrado['Especie'].apply(lambda x: 0 if x == 'PIC' else 1).values

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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

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
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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


# PREDICCIÓN DE DATOS
data = pd.read_excel('../../Resultados/excel/TODOS/TODOS.xlsx')

# Ejecutar el modelo CNN y obtener el modelo entrenado y el escalador
model, scaler = ejecutar_cnn(data)

# Función para preprocesar nuevos datos y realizar predicciones
def comprobar_nuevos_datos(model, nuevos_datos, scaler):
    X_nuevos = nuevos_datos.iloc[:, 3:].values
    y_nuevos = nuevos_datos['Especie'].apply(lambda x: 0 if x == 'PI' else 1).values

    X_nuevos = X_nuevos.reshape(X_nuevos.shape[0], X_nuevos.shape[1], 1)

    X_nuevos = scaler.transform(X_nuevos.reshape(-1, X_nuevos.shape[1])).reshape(X_nuevos.shape)

    y_pred_prob = model.predict(X_nuevos)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    print("\nPredicciones para los nuevos datos:")
    print(y_pred.flatten())

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

# Cargar los nuevos datos desde un archivo Excel
# nuevos_datos = pd.read_excel('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPruebaPicual.xlsx')

# Preprocesar y comprobar los nuevos datos con el modelo entrenado
# comprobar_nuevos_datos(model, nuevos_datos, scaler)