import numpy as np
import onnxruntime as ort

from MyTensor import MyTensor

class TensorHelper:
    @staticmethod
    def duplicate(data, dimensions):
        floats = np.concatenate([data, data])
        tensor = ort.OrtValue.ortvalue_from_numpy(floats.reshape(dimensions))
        return MyTensor(tensor, dimensions)

    @staticmethod
    def multiply_tensor_by_float(data, value, dimensions):
        data = data * value
        tensor = ort.OrtValue.ortvalue_from_numpy(data.reshape(dimensions))
        return MyTensor(tensor, dimensions)

    @staticmethod
    def sum_tensors(tensor_array, dimensions):
        sum_array = np.sum([tensor.numpy() for tensor in tensor_array], axis=0)
        tensor = ort.OrtValue.ortvalue_from_numpy(sum_array.reshape(dimensions))
        return MyTensor(tensor, dimensions)

    @staticmethod
    def add_tensors(sample, sum_tensor, dimensions):
        result = sample + sum_tensor
        tensor = ort.OrtValue.ortvalue_from_numpy(result.reshape(dimensions))
        return MyTensor(tensor, dimensions)

    @staticmethod
    def divide_tensor_by_float(data, value, dimensions):
        data = data / value
        tensor = ort.OrtValue.ortvalue_from_numpy(data.reshape(dimensions))
        return MyTensor(tensor, dimensions)

    @staticmethod
    def split_tensor(tensor_to_split, dimensions):
        tensor1 = tensor_to_split[0:1]
        tensor2 = tensor_to_split[1:2]
        return tensor1, tensor2
