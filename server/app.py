from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, json, jsonify, render_template
from flask_cors import CORS
import cv2
# from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


def generate_occlusion():
    np_img = cv2.imread('../images/vankok.jpg')
    resized_img = cv2.resize(np_img, (224, 224))
    print(resized_img.shape)
    alphas = 255 * np.ones((224, 224, 1))
    resized_img = np.dstack((resized_img, alphas))
    return resized_img.flatten().tolist()

@app.route('/model/predict/', methods=['POST'])
def predict():
    # Decoding and pre-processing base64 image
    payload = request.json
    # print(payload)

    if payload['signature_name'] == 'occlusion':
        content = {"outputs": {0: generate_occlusion()}}
    else:
        # Making POST request 
        r = requests.post('http://localhost:8501/v1/models/' + payload['signature_name']  + ':predict', json=payload)
        content = json.loads(r.content.decode('utf-8'))
    
    # Decoding results from TensorFlow Serving server
    # output = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(content)
