# %% def
import numpy as np
from Layers.Base import*

class ReLU(BaseLayer):
    def __init__(self):
        """
        self.relu_result: [batch, input]
        """
        super().__init__()
        self.relu_result = []
        

    def forward(self, input_tensor):
        out = np.maximum(0, input_tensor)
        self.relu_result = input_tensor
        return out

    def backward(self, error_tensor):
# slide 03, 97
# clip 9 2021/2022 video
# clip 15 activation function back propagation

        """ Back propagation in network
        for back propagation: the used formula is
        E_{l-1} = W^T *E_l --- it should be inner product.

        Parameter:
        --------------------
        error_tensor: ndarray,
                    error propagating in layers with shape of (batch, input)
        
        Return: 
        --------
        array like, 
            the error should be propagated to next layer
        """

# help, I don't get it
# why this is simple multiplication instead of inner product:
# inner product gives scalar, we need to propagate vector back!
        w = self.relu_result>0  #Backpopagate
        #out = self.out * (1-self.out) * (error_tensor * w)
        err = error_tensor * w
        return err

# %% Test
if __name__=="__main__":
    input = np.random.uniform(-10, 10, [4, 5]) # batch 4, input size = 5
    out = ReLU.forward(ReLU, input)