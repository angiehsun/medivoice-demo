"""
Run this once to convert the best Keras model to TensorFlow.js format.

  pip3 install tensorflow tensorflowjs
  python3 convert_model.py
"""
import pathlib, sys

try:
    import tensorflow as tf
    import tensorflowjs as tfjs
except ImportError:
    sys.exit("Run:  pip3 install tensorflow tensorflowjs")

MODEL_PATH = pathlib.Path(__file__).parent.parent / "Scripts/models/asl_lstm_3T_80S.h5"
OUT_PATH   = pathlib.Path(__file__).parent / "tfjs_model"

if not MODEL_PATH.exists():
    sys.exit(f"Model not found: {MODEL_PATH}")

print(f"Loading {MODEL_PATH} ...")
model = tf.keras.models.load_model(str(MODEL_PATH))
model.summary()

print(f"\nConverting to TF.js → {OUT_PATH} ...")
tfjs.converters.save_keras_model(model, str(OUT_PATH))
print("Done! Upload the web_demo/ folder to GitHub Pages.")
