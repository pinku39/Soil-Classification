from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

model = load_model('my_model.h5')

classes = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

def preprocess_image(image_path):
    img = Image.open(image_path).resize((220, 220))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_label(img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)
    predicted_label = classes[np.argmax(prediction)]
    return predicted_label

# Flask routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index3.html")

@app.route("/about")
def about_page():
    return "Please "

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/upload/" + img.filename
        img.save(img_path)
        predicted_label = predict_label(img_path)
        return render_template("index3.html", prediction=predicted_label, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
