from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn
import logging
import time
import interpreter_cls
from interpreter_cls import javaemployees

app = FastAPI()


# creating object for javaemployees class
obj = interpreter_cls.javaemployees(
    "/home/ubuntu/Desktop/train_data_facesagain/quatized_model/model.tflite")
interpreter, output_details, input_details = obj.inter_cls()


# it is the basic configuration it will create stream handler
logging.basicConfig(filename='test1.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((224, 224))
    # image= np.expand_dims(image, 0)
    image = np.array(image).astype("float32")
    image = image/255

    return image


# all the images names
CLASS_NAMES = ["feraz", "ganesh", "lakshmee", "pooja",
               "roshan", "sathish", "shrikant", "shruthi"]


@app.get('/')
def home():
    return {'welcome to face recognition quantized model'}


@app.post('/prediction')
async def predict(file: UploadFile = File(...)):
    if not file.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        raise HTTPException(
            status_code=400, detail=f'File \'{file.filename}\' is not an right extension of image.')

    logging.info('Got the image from user')
    start = time.perf_counter()

    image = read_file_as_image(await file.read())

    logging.info(f'Successfully preprocessed {image}')

    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]['index'], [image])

    interpreter.invoke()

    # Gets the value of the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # np.max function returns the maximum array element
    confidence = np.max(output_data)

    # the argmax function returns the index of the maximum value of a Numpy array
    index = np.argmax(output_data)

    # threshold setting
    if confidence >= 0.8:
        name = CLASS_NAMES[index]
    else:
        name = 'match not found'

    end = time.perf_counter()

    logging.info(f'finished time is {end}-{start}')

    return {'name': name, 'confidence': float(confidence)}
