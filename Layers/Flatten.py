import numpy as np
from Layers.Base import BaseLayer

# class Flatten(BaseLayer):
#     def __init__(self):
#         super().__init__()
#         self.channel = 0
#         self.width = 0
#         self.height =0
#         self.batch = 0

#     def forward(self, input_tensor):
#         """
#         connecting covolutional layer and fully connected layers
        
#         parameter:
#         -----
#         input_tensor: ndarray,
#             shape(batch, channel, height, width)
        
#         Return:
#         ------
#         ndarray,
#             shape of (batch, channel * height * width)
#         """
#         self.batch, self.channel, self.height, self.width = input_tensor.shape
#         return np.reshape(input_tensor, shape=(self.batch, -1), order="C")
#     def backward(self, error_tensor):
#         return np.reshape(error_tensor, shape= (self.batch, self.channel, self.height, self.width))
    

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self,input_tensor):
        #Reshape input tensor
        self.input_shape = input_tensor.shape #AS we go backward we need to de-flatten
        f = input_tensor.reshape(input_tensor.shape[0], -1)
        return  f

    def backward(self,error_tensor):
        #Reshape error tensor
        f =  error_tensor.reshape(self.input_shape) 
        return f
    
