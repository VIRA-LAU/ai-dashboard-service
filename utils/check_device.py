import torch

import tensorflow as tf
from tensorflow.python.client import device_lib

def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #To detect if there is a GPU available
    print(device,"is detected")

    tf.test.gpu_device_name()
    device_lib.list_local_devices()