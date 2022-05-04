import time
from os import listdir
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import os
import numpy as np
import matplotlib.pyplot as plt

#Inicio contador tiempo
timeStat = time.time()

#rutas de interes
path = os.getcwd()
pathModels = path + "\\Models"
pathDataSet = path + "\\DataSet"
pathBuenas = pathDataSet + "\\etiquetasBuenas"
pathMalas = pathDataSet + "\\etiquetasMalas"
pathModelSelected = pathModels + "\\Modelo_100_etapas"

#leer cantidad de imagenes por carpeta de interes
numberGoogImages = listdir(pathBuenas)
numberBadImages = listdir(pathMalas)
print("cantidad de imagenes sin defectos: {}".format(len(numberGoogImages)))
print("cantidad de imagenes con defectos: {}".format(len(numberBadImages)))

#Crear el dataset generador

IMAGE_SIZE = 224
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    validation_split=0.2 #20% para pruebas
)

#Generadores para sets de entrenamiento y pruebas
train_generator = datagen.flow_from_directory(
    pathDataSet,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset='training')

val_generator = datagen.flow_from_directory(
    pathDataSet,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset='validation')

image_batch, label_batch = next(val_generator)
image_batch.shape, label_batch.shape

print (train_generator.class_indices)
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('etiquetas_labels.txt', 'w') as f:
  f.write(labels)

# SE PUEDE IMPORTAR EL MODELO DE DOS FORMAS:
    # LA PRIMERA: CON LA URL DEL MODELO
    # LA SEGUNDA: CON EL COMANDO DE LA APLICACION

# PRIMERA FORMA:
#Importar modelo preentrenado de google:
"""
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))
mobilenetv2.trainable = False
"""
# SEGUNDA FORMA:
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')
base_model.trainable = False

#agregamos capa de salida con la clasificacion deseada
#(numerio de entradas, )

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units=2, activation='softmax')
])

# Entrenamiento basico
"""
modeloBasico = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2, activation='softmax')
])
"""

#Compilar como siempre
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#INFORMACION DEL MODELO:
model.summary()
print('Number of trainable weights = {}'.format(len(model.trainable_weights)))

#Entrenar el modelo
EPOCAS = 200

historial = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=EPOCAS,
                    validation_data=val_generator,
                    validation_steps=len(val_generator))

#fin de contador de tiempo
timeStop = time.time()
timeInSeg = timeStop - timeStat
timeInMin = ((timeStop - timeStat)/60)

print("Tiempo en segundos: {}".format(timeInSeg))
print("Tiempo en minutos: {}".format(timeInMin))

#Graficas de precisión
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(EPOCAS)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
plt.legend(loc='upper right')
plt.title('Pérdida de entrenamiento y pruebas')
plt.show()

#Guardar el modelo en formato SavedModel
modelName = input("nombre del modelo: \n")
if model.save(modelName):
    print("modelo guardado correctamente como: {}".format(modelName))
#Modelo en version recortada .h5
model.save(modelName+".h5")

#CONVERTIR A TFlite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224.tflite', 'wb') as f:
  f.write(tflite_model)


# A generator that provides a representative dataset
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(pathDataSet + '/*/*')
  for i in range(50):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open('mobilenet_v2_1.0_224_quant.tflite', 'wb') as f:
  f.write(tflite_quant_model)



