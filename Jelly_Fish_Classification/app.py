from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "C:\\Users\\chaha\\Desktop\\JellyFish\\VGG16_Jellyfish_Classifier_Finetuned.h5"
model = load_model(MODEL_PATH)

# Define class names
class_names = {
    0: "Moon Jellyfish",
    1: "Barrel Jellyfish",
    2: "Blue Jellyfish",
    3: "Compass Jellyfish",
    4: "Lions Mane Jellyfish",
    5: "Mauve Stinger Jellyfish"
}

# Create an uploads folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route to the main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image was uploaded
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Save the uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Adjust size if needed
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        probabilities = model.predict(img_array)
        class_idx = np.argmax(probabilities, axis=1)[0]
        predicted_class = class_names.get(class_idx, "Unknown")

        return render_template("index.html", prediction=predicted_class, image_path=filepath)

    return render_template("index.html", prediction=None)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, use_reloader=False)
