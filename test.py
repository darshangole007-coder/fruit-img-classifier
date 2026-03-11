import tensorflow as tf
import numpy as np
from PIL import Image
import sys, os, json

MODEL_PATH = "model/fruit_model.h5"
CLASS_PATH = "model/class_names.json"
IMG_SIZE = 224
CONF_THRESHOLD = 65  # %

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

if len(sys.argv) < 2:
    print("❌ Usage: python test.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print("❌ Image file not found")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

img = Image.open(image_path).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))
img = np.array(img)
img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
img = np.expand_dims(img, axis=0)

preds = model.predict(img)[0]
idx = int(np.argmax(preds))
confidence = preds[idx] * 100

print("✅ Prediction Complete")

if confidence < CONF_THRESHOLD:
    print("⚠️ Unable to confidently classify this image")
else:
    print(f"🐾 Fruit      : {CLASS_NAMES[idx]}")
    print(f"📊 Confidence  : {confidence:.2f}%")
