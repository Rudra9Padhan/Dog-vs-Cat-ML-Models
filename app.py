from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("dog_vs_cat_classifier.h5")

app = Flask(__name__)

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to model's input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp.jpg"
    try:
        # Save the uploaded file
        file.save(file_path)

        # Preprocess the image
        img = preprocess_image(file_path)

        # Make a prediction
        prediction = model.predict(img)
        result = "Dog" if prediction[0][0] > 0.5 else "Cat"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
