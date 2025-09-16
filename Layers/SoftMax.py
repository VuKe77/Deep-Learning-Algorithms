# %% def
import numpy as np
from Layers.Base import *

#Skeleton to resolve import errors when running tests
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.yhat = []

    def forward(self, input_tensor):
        """
        softmax function: hat y_k = frac{exp(xk)}{sum_j^K exp(xj)}
        
        Here K is the number of class, 

        k is the current class position.

        This is a function to map the result into [0, 1]  

        input tensor: [batch, class]

        return: result for each batch [batch, class]   
        """

        sum = np.sum(np.exp(input_tensor - np.max(input_tensor)), axis = 1) #Avoid overflow
        self.yhat = (np.exp(input_tensor - np.max(input_tensor)).T/sum).T
        return self.yhat

    
    def backward(self, error_tensor):
        # error_tensor shape: (9, 4) (batch, class)
        # yhat shape: (9, 4), (batch, class)
        # here backpropagate the derivitive of the loss, and through back propagation, 
        # with the chain rule, the deravitive will accumulate, giving the current dev of this layer


        # check ex video
        return self.yhat*(error_tensor - np.sum(error_tensor*self.yhat, axis = 1, keepdims=True))