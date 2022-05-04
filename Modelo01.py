# direccion del proyecto
#C:\Users\marci\PycharmProjects\SelectorEtiquetas\Imagenes

#importar librerias:
import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_hub as hub
import time


#rutas de interes
path = os.getcwd()
pathModels = path + "\\Models"
pathDataSet = path + "\\DataSet"
pathBuenas = pathDataSet + "\\etiquetasBuenas"
pathMalas = pathDataSet + "\\etiquetasMalas"
#pathEstaticas = pathDataSet + "\\etiquetasEstaticas"

#leer cantidad de imagenes por carpeta de interes
numberGoogImages = listdir(pathBuenas)
numberBadImages = listdir(pathMalas)
#numberStaticImages = listdir(pathEstaticas)

print("cantidad de imagenes sin defectos: {}".format(len(numberGoogImages)))
print("cantidad de imagenes con defectos: {}".format(len(numberBadImages)))
#print("cantidad de imagenes estaticas: {}".format(len(numberStaticImages)))



#mostrar algunas figuras:

plt.figure(figsize=(15,15))
for i, nombreimg in enumerate(numberGoogImages[:25]):
  plt.subplot(5,5,i+1)
  imagen = mpimg.imread(pathBuenas + '/' + nombreimg)
  plt.imshow(imagen)

##Aumento de datos

#Crear el dataset generador
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    validation_split=0.2 #20% para pruebas
)

#Generadores para sets de entrenamiento y pruebas
data_gen_entrenamiento = datagen.flow_from_directory(pathDataSet, target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory(pathDataSet, target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='validation')

#Imprimir 10 imagenes del generador de entrenamiento
for imagen, etiqueta in data_gen_entrenamiento:
  for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i])
  break
plt.show()

#Inicio contador tiempo
timeStat = time.time()

#Importar modelo preentrenado de google:
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))

#Congelar el modelo descargado
mobilenetv2.trainable = False

#agregamos capa de salida con la clasificacion deseada
#(numerio de entradas, )


modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2, activation='softmax')
])

modelo.summary()

#Compilar como siempre
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Entrenar el modelo
EPOCAS = 1000

historial = modelo.fit(
    data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas
)


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
if modelo.save(modelName):
    print("modelo guardado correctamente como: {}".format(modelName))

modelo.save(modelName+".h5")


"""
#Categorizar una imagen de internet
from PIL import Image
import requests
from io import BytesIO
import cv2

def categorizar(url):
  respuesta = requests.get(url)
  img = Image.open(BytesIO(respuesta.content))
  img = np.array(img).astype(float)/255

  img = cv2.resize(img, (224,224))
  prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
  return np.argmax(prediccion[0], axis=-1)
  
  #0 = cuchara, 1 = cuchillo, 2 = tenedor
url = 'https://th.bing.com/th/id/R.e44940120b7b67680af246c3b3e936f2?rik=XZPLfxf4nHlzyw&pid=ImgRaw&r=0' #debe ser 2
prediccion = categorizar (url)
print(prediccion)

#Crear la carpeta para exportarla a TF Serving
!mkdir -p carpeta_salida/modelo_cocina/1

#Guardar el modelo en formato SavedModel
modelo.save('carpeta_salida/modelo_cocina/1')

"""

