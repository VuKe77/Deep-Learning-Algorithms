import numpy as np
from Layers.Base import*

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__(trainable = False, testing_phase=False)
        self.probability = probability
        self.drop_map = 1

    def forward(self, input_tensor):
        """
        set activation to zero with probability 1-p

        Parameter:
        ---
        input_tensor: tuple,
            shape of (batch, input_size)"""
        if not self.testing_phase:
            batch, input_size = input_tensor.shape
            drop = np.zeros(input_size)
            drop[0:int(np.ceil(input_size*self.probability))] = 1
            drop = np.random.permutation(drop)
            #print(drop, input_size, self.probability, input_size*self.probability, int(input_size*self.probability))
            drop2d = np.tile(drop, (batch, 1))
            self.drop_map = drop2d
            return drop2d*input_tensor/self.probability

        else:
            return input_tensor
        
    def backward(self, error_tensor):
        if not self.testing_phase:
            return self.drop_map*error_tensor/self.probability
