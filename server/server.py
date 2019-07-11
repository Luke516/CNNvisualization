from flask import Flask, request, Response, redirect
import jsonpickle
import numpy as np
import cv2

# Initialize the Flask application
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# route http posts to this method
@app.route('/api/method1', methods=['POST'])
def test():
    print(request.headers)
    print(request.files)
    print(request.data)
    print(request.form)

    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)

    #read image file string data
    filestr = request.files['file'].read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)