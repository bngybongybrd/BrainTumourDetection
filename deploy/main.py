import keras, os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
from flask import Flask, render_template, request

app = Flask(__name__)

model = keras.models.load_model('BrainTumourDetector.h5')


def preprocess_and_predict(path, contrast_factor=1.4, resize=(256, 256)):
    # Open Image and convert to RGB
    img = Image.open(path)
    img = img.convert("RGB")

    # Adjust Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Resize image
    img = img.resize(resize)

    # Convert image to format suitable for model input
    input_array = np.array(img) / 255
    input_array = np.expand_dims(input_array, axis=0)

    prediction = model.predict(input_array)

    if prediction < 0.5:
        return "No tumour detected"
    else:
        return "Tumour detected"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None

    if request.method == "POST" and "mri" in request.files:
        if request.files["mri"].filename:
            photo = request.files['mri']
            file_name = secure_filename(photo.filename)
            
            # form and save path
            path = os.path.join('uploads', file_name)
            photo.save(path)

            # Preprocess and predict given MRI scan
            prediction_result = preprocess_and_predict(path)

    return render_template('index.html', prediction_result=prediction_result)


if __name__ == "__main__":
    app.run()