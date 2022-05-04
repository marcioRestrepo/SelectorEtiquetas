
# al final cuando tenfas el modelo compulado con el nombre "model"
# basta con agregar estas lineas de codigo para obtener una version guardada del modelo.

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)