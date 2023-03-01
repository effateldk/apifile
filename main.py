from __future__ import division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import joblib as joblib


# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify

# Define a flask app
app = Flask(__name__)

def preprocessing(img ):
    img = Image.open(img)
    img = img.resize((224,224))

    # Preprocessing the image
    image = np.array(img.convert('RGB'))

    image.shape = (1, 224, 224, 3)

    return image


classes = ['Coccidiosis', 'Healthy', 'New Castle Disease','Salmonella']

model = load_model('efficientnetb3-Chicken Disease-98.27.h5')

@app.route('/predictApi', methods=['POST'])
def predict_api():
    #try:
        if 'fileup' not in request.files:
             return "Please try again, the image doesn't exit "
        img = request.files.get('fileup')
        image = preprocessing(img)

        result = model.predict(image)
        ind = np.argmax(result)
        prediction = classes[ind]

        return jsonify({'prediction': prediction})
#except:
        # return jsonify({'ERROR': 'error occured try again'})
if __name__ == '_main_':
    app.run(debug=True)

