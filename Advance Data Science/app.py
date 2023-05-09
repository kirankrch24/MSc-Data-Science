import numpy as np
import cv2
import json
from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = load_model('my_cnn_model.h5')  # Load the saved model

# Define a function to preprocess the image data
def preprocess_image(image, target_size=(25, 25)):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
    processed_image = preprocess_image(decoded)

    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': prediction
    }
    return json.dumps(response)

if __name__ == '__main__':
    app.run(debug=True)
