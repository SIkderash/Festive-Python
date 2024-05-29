import numpy as np
import onnxruntime as ort

class MyTensor:
    def __init__(self, tensor, shape, buffer=None):
        self.tensor = tensor
        self.shape = shape
        self.buffer = buffer

    def get_buffer(self):
        return self.buffer

    def set_buffer(self, buffer):
        self.buffer = buffer

    def get_tensor(self):
        return self.tensor

    def set_tensor(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape
