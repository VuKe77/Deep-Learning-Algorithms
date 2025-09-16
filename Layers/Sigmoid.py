import numpy as np
from Layers.Base import *


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self._activation = None
    
    def forward(self,input_tensor):
        out = 1/(1+np.exp(-input_tensor))
        self.activation = out
        return out
    def backward(self, error_tensor):
        deriv = self.activation*(1-self.activation)
        return error_tensor*deriv
    
    @property
    def activation(self):
        return self._activation
    @activation.setter
    def activation(self,new_activation):
        self._activation = new_activation

    