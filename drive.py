import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
from base64 import b64decode
from io import BytesIO
from PIL import Image
import cv2
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow import keras
from keras.saving import register_keras_serializable

# Initialize Socket.IO server and Flask app
sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10

# Image processing function
def img_processes(img):
    img = img[60:135, :, :]  # Corrected slice
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# Handle connection events
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Send control commands
def send_control(steering_angle, throttle):
    print(f"Sending control: Steering Angle = {steering_angle}, Throttle = {throttle}")
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

# Handle telemetry events
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(b64decode(data['image'])))
    image = np.asarray(image)
    image = img_processes(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    


# Register the custom metric mae and mse functions
@register_keras_serializable()
def mae(y_true, y_pred):
    return MeanAbsoluteError()(y_true, y_pred)

@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)




if __name__ == '__main__':
 # Load model with custom objects for mae and mse
    model = keras.models.load_model('/Users/mac/Desktop/Berlin Workshop/model.h5', custom_objects={'mae': mae, 'mse': mse})
    
    # Recompile the model to ensure it's properly configured
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)
