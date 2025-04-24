import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from collections import Counter
from imblearn.over_sampling import SMOTE
import keras_tuner as kt
import json

# Definir constantes
EPOCHS = 50


# Función para establecer la semilla
def establecer_semilla(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


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


# Función para crear el modelo con parámetros de KerasTuner
def crear_modelo(hp, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Optimización del tamaño de pooling
    pool_size = hp.Choice('pool_size', values=[2, 3, 4])

    # Optimización de los filtros y el tamaño del kernel para cada capa convolucional
    model.add(Conv1D(
        filters=hp.Int('filters_1', min_value=8, max_value=64, step=8),
        kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=1),
        activation='elu'
    ))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Conv1D(
        filters=hp.Int('filters_2', min_value=16, max_value=128, step=16),
        kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=1),
        activation='elu'
    ))

    model.add(Conv1D(
        filters=hp.Int('filters_3', min_value=32, max_value=256, step=32),
        kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=1),
        activation='elu'
    ))

    model.add(Conv1D(
        filters=hp.Int('filters_4', min_value=64, max_value=512, step=64),
        kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=1),
        activation='elu'
    ))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Flatten())

    # Capa densa con número de neuronas optimizado
    model.add(Dense(
        units=hp.Int('ff_dim', min_value=32, max_value=256, step=32),
        activation='relu'
    ))

    # Optimización de la tasa de dropout
    model.add(Dropout(
        rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    ))

    # Capa final de clasificación binaria
    model.add(Dense(1, activation='sigmoid'))

    # Optimización de batch_size y patience (almacenados para uso posterior)
    batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    patience = hp.Choice('patience', values=[5, 8, 10])

    # Compilación del modelo con tasa de aprendizaje optimizada
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        ),
        metrics=['accuracy']
    )

    return model


# Función para ejecutar la optimización bayesiana
def optimizar_modelo(X_train, y_train, X_test, y_test):
    establecer_semilla()

    # Crear el tunador de optimización bayesiana
    tuner = kt.BayesianOptimization(
        lambda hp: crear_modelo(hp, input_shape=(X_train.shape[1], X_train.shape[2])),
        objective='val_accuracy',  # Objetivo de la optimización
        max_trials=10,  # Número máximo de combinaciones de hiperparámetros
        executions_per_trial=1,  # Número de ejecuciones por combinación
        directory='keras_tuner_dir',
        project_name='cnn_olivos_optimization'
    )

    # Callback de parada temprana (usará el patience optimizado durante la búsqueda)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Ejecutar la búsqueda de hiperparámetros
    tuner.search(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    # Mostrar los mejores hiperparámetros encontrados
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    best_params = {
        'filters_1': best_hyperparameters['filters_1'],
        'filters_2': best_hyperparameters['filters_2'],
        'filters_3': best_hyperparameters['filters_3'],
        'filters_4': best_hyperparameters['filters_4'],
        'kernel_size': best_hyperparameters['kernel_size'],
        'ff_dim': best_hyperparameters['ff_dim'],
        'dropout_rate': best_hyperparameters['dropout_rate'],
        'learning_rate': best_hyperparameters['learning_rate'],
        'pool_size': best_hyperparameters['pool_size'],
        'batch_size': best_hyperparameters['batch_size'],
        'patience': best_hyperparameters['patience']
    }
    print("Mejores parámetros encontrados:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    # Guardar los mejores hiperparámetros en un archivo JSON
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    print("Mejores hiperparámetros guardados en 'best_params.json'")


# Ejemplo de ejecución
if __name__ == "__main__":
    # Cargar los datos
    data = pd.read_excel('../../Resultados/excel/TODOS/TODOS.xlsx')

    # Preparar los datos
    X_train, X_test, y_train, y_test, _ = preparar_datos(data)

    # Ejecutar la optimización
    optimizar_modelo(X_train, y_train, X_test, y_test)