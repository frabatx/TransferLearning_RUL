import tensorflow as tf
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)