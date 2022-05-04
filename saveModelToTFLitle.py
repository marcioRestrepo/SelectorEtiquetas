import tensorflow as tf
import os
# Direccion del modelo
path = os.getcwd()
pathModels = path + "\\Models"
pathModelSelected = pathModels + "\\Modelo_100_etapas"
print(pathModelSelected)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(pathModelSelected) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)