import numpy as np


class Constant:
    def __init__(self, weight_init = 0.1):
        self.weight_init = weight_init

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Parameters
        -----------
        weights_shape: (fan_out, fan_in), tuple
                        Weight's shape.
        - For fully connected layers:
        fan_in: int
                input dimension of the weights.
        fan_out: int
                 output dimension of the wieghts.
        - For convolutional layers:
        fan_in: int
                input channels* kernel height* kernel width
        fan_out: int
                output channels* kernel height* kernel width
                
        Returns
        ---------
        ndarray
            initialized tensor with shape (channel, kernel height, kernel width)
        """

        return np.ones(weights_shape) * self.weight_init

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Parameters
        -----------
        weights_shape: (fan_out, fan_in, width, height), tuple
                        Weight's shape.
        - For fully connected layers:
        fan_in: int
                input dimension of the weights.
        fan_out: int
                 output dimension of the wieghts.
        - For concolutional layers:
        fan_in: int
                input channels* kernel height* kernel width
        fan_out: int
                output channels* kernel height* kernel width
                
        Returns
        ---------
        ndarray
            initialized tensor with shape (channel, kernel height, kernel width)
        """

        return np.random.uniform(0,1,weights_shape)
class Xavier:

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2/(fan_out+fan_in)), weights_shape)
    
class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2/fan_in), weights_shape)
