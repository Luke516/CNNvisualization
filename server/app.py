import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify
# from flask_cors import CORS

app = Flask(__name__)


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/model/predict/', methods=['POST'])
def predict():
    # Decoding and pre-processing base64 image
    payload = request.json
    print(payload)

    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/model:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    output = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return r