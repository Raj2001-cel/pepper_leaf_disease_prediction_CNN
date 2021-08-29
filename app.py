from fastapi import FastAPI, File, UploadFile
# import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask import Flask
from flask import Flask, redirect, url_for, request, render_template
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app = Flask(__name__)

MODEL = tf.keras.models.load_model("//models//1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route("/ping")
async def ping():
    return "Hello, I am alive"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    # x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased pepper leaf"
    else:
        preds="The leaf is fresh pepper plant"
        
    
    
    return preds

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, MODEL)
        result=preds
        return result
    return "ERRORR RAJ"



# @app.route("/predict",methods=['GET', 'POST'])
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }

if __name__ == "__main__":
    app.run(port=5001,debug=True)