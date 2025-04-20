import tensorflow as tf
from tensorflow.python.keras.models import load_model

# Load the Keras model
model = load_model("chatbot_model.h5", compile=False)

# Save the model in SavedModel format (this will create a directory)
model.save("saved_model_dir")  # Note: No .h5 or any extension, just the directory name

# Now convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")
tflite_model = converter.convert()

# Save the converted TFLite model
with open("chatbot_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model conversion successful!")
