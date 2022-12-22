import tensorflow as tf



class javaemployees:
    def __init__(self,model_path):
        self.model_path = model_path
    
    def inter_cls(self):
        interpreter = tf.lite.Interpreter(self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter,output_details,input_details

# obj = javaemployees()
# obj.inter_cls()
