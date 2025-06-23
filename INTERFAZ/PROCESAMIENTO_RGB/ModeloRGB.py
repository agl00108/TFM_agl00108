import numpy as np
import tensorflow as tf
import random
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.src.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parámetros del modelo
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10

best_params = {'dropout_rate': 0.3, 'kernel_size': (3, 3)}

def establecer_semilla(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def cargar_datos_variedades(directorio, img_size=(224, 224), batch_size=BATCH_SIZE, test_size=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=test_size)
    train_generator = datagen.flow_from_directory(directorio, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training', shuffle=True)
    validation_generator = datagen.flow_from_directory(directorio, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation', shuffle=True)
    return train_generator, validation_generator

def mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo):
    print(f"Evaluación del modelo {nombre_modelo}")
    print(f"Pérdida: {loss}")
    print(f"Precisión: {accuracy}")
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {nombre_modelo}')
    plt.xlabel('Predictions')
    plt.ylabel('Real Values')
    plt.show()

def ejecutar_cnn_transfer_learning(train_gen, val_gen, save_path='modelo_vgg16.h5'):
    establecer_semilla()
    start_time = time.time()

    #Definición de los modelos
    #VGG16
    #base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #ResNet50
    #base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #InceptionV3
    #base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    #Xception
    #base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    #MobileNetV2
    #base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    #DenseNet121
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False


    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(best_params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[early_stopping])

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.show()

    loss, accuracy = model.evaluate(val_gen)
    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    nombre_modelo = "DenseNet121 Transfer Learning"
    mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo)

    model.save(save_path)
    print(f"Modelo guardado en: {save_path}")

    end_time = time.time()
    print(f'Tiempo de ejecución: {end_time - start_time:.2f} segundos')
    return model

def predecir_con_modelo(model, directorio, img_size=(224, 224)):
    resultados = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            ruta_imagen = os.path.join(directorio, archivo)
            imagen = cv2.imread(ruta_imagen)
            imagen = cv2.resize(imagen, img_size)
            imagen = imagen / 255.0
            imagen = np.expand_dims(imagen, axis=0)
            prediccion = model.predict(imagen)
            prediccion_binaria = (prediccion > 0.5).astype(int)
            resultados.append({
                'Archivo': archivo,
                'Predicción_Modelo': 'Picual' if prediccion_binaria == 1 else 'No Picual',
                'Probabilidad': prediccion[0][0]
            })
    return pd.DataFrame(resultados)


if __name__ == "__main__":
    directorio = '../../RGB/Dataset_hojas/Dataset Entrenamiento'
    train_gen, val_gen = cargar_datos_variedades(directorio)
    model = ejecutar_cnn_transfer_learning(train_gen, val_gen)
    resultados = predecir_con_modelo(model, '../../RGB/Dataset_hojas/Dataset Prueba')
    print(resultados)

'''
Best val_accuracy So Far: 0.9821428656578064
Total elapsed time: 00h 23m 47s
Mejores parámetros encontrados:
Unidades: 384
Tasa de dropout: 0.2384
Tasa de aprendizaje: 0.0009937535905290506

Value             |Best Value So Far |Hyperparameter
192               |512               |units
0.4               |0.3               |dropout_rate
0.00060476        |0.0068338         |learning_rate
'''