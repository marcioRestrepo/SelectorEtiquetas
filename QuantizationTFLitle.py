# Load TensorFlow
import tensorflow as tf
import os

# Direccion del modelo
path = os.getcwd()
pathModels = path + "\\Models"
pathModelSelected = pathModels + "\\Modelo_100_etapas"
pathDataSet = path + "\\DataSet"
print(pathModelSelected)

# A generator that provides a representative dataset

IMAGE_SIZE = 224

def representative_dataset_gen():
  dataset_list = tf.data.Dataset.list_files(pathDataSet + '/*/*')
  for i in range(10):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]


# Set up the converter
converter = tf.lite.TFLiteConverter.from_saved_model(pathModelSelected)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

open("quantized.tflite", "wb").write(tflite_quant_model)

