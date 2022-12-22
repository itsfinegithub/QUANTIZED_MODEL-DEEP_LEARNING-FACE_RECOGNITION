import tensorflow 



interpreter = tensorflow.lite.Interpreter(model_path = "/home/ubuntu/Desktop/train_data_facesagain/model.tflite")

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
print(input_details)
