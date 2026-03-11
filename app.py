from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = tf.keras.models.load_model("model/fruit_model.h5")
with open("model/class_names.json") as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = 224
CONF_THRESHOLD = 65

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img = Image.open(request.files["image"]).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))
    
    # FIX: Convert the numpy float32 to a standard Python float
    confidence = float(preds[idx] * 100) 

    if confidence < CONF_THRESHOLD:
        return jsonify({
            "fruit": "Unknown", 
            "confidence": round(confidence, 2)
        })

    return jsonify({
        "fruit": CLASS_NAMES[idx],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
