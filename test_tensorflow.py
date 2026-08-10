import tensorflow as tf

print("=" * 60)
print("TensorFlow Test")
print("=" * 60)

print("TensorFlow Version :", tf.__version__)

print("\nKeras Version :", tf.keras.__version__)

print("\nAvailable Devices")

for device in tf.config.list_physical_devices():
    print(device)

print("\nGPU Devices")

print(tf.config.list_physical_devices('GPU'))

print("\nTensorFlow Loaded Successfully!")