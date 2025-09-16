import numpy as np
from Layers.Base import *


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self._activation = None
    
    def forward(self,input_tensor):
        out = np.tanh(input_tensor)
        self.activation = out
        return out
    def backward(self, error_tensor):
        deriv = 1-self.activation**2
        return error_tensor*deriv
    
    @property
    def activation(self):
        return self._activation
    @activation.setter
    def activation(self,new_activation):
        self._activation = new_activation
    

# %% Test
if __name__=="__main__":
    input = np.random.uniform(-10, 10, [4,5]) # batch 4, input size = 5
    out = TanH.forward(TanH, input)
    bp  = TanH.backward(TanH,out)
    print(out.shape)