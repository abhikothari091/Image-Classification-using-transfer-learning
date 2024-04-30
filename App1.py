import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model_path = 'imagenet.h5'
model = load_model(model_path)
model.make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        file_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        pred_class = decode_predictions(preds, top=1)[0][0][1]
        return pred_class

if __name__ == "__main__":
    app.run(debug=True)
