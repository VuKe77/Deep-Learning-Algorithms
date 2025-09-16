# %% def
import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    """
    Normalize within the network over batch.
    calculate generaly the mean and variance over batch and spatial domain

    constructors
    ---
    bias: shape of (channel, ), initialize to zero
    gamma: shape of (channel, ), initialize to zero, weights"""
    def __init__(self, channels):
        super().__init__(trainable = True, testing_phase = False)
        self.channels = channels
        self.bias = None
        self.weights = None
        self.initialize()
        self.mean_test = 0
        self.var_test = 0
        self.mean_train = 0
        self.var_train = 0
        self.batch = 0
        self.alpha = 0.8 # wrong but I don't know how to get alpha
        self.input_tensor = None
        self.x_tilde = None
        self.first_batch = True
        self._optimizer = None
        self.conv_shape = None
        self.gradient_weights = None
        self.gradient_bias = None

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    def moving_average_estimation(self):
        self.mean_test = self.mean_test*self.alpha + (1-self.alpha)*self.mean_train
        self.var_test = self.var_test*self.alpha + (1-self.alpha)*self.var_train
        
    def forward(self, input_tensor):
        """Normalization along batch
        if input.ndim == 4: convolution,
        else: FCL
        
        Note:
        ---
        xtilde = \frac{X - \mu_B}{\sqrt{\sigma^2_B + \varepsilon}}
        \hat Y = \gamma X_tilde + \beta
        
        \gamma, \beta has the same shape of input X
        """
        eps = 1e-11
        self.input_tensor = input_tensor
        is_conv = False

        if input_tensor.ndim == 4:
            is_conv = True
            input_tensor = self.reformat(input_tensor)
            # mean = np.mean(input_tensor, axis = 0)
            # var = np.var(input_tensor, axis=0)
            
            # #initialize test mean and var 
            # if self.first_batch:
            #     self.mean_test = mean
            #     self.var_test = var
            #     self.first_batch = False


            # self.mean_train = mean
            # self.var_train = var
            # gamma = np.expand_dims(self.gamma, (0, 2, 3))
            # bias = np.expand_dims(self.bias, (0, 2, 3))

    
        mean = np.mean(input_tensor, axis=0)
        var = np.var(input_tensor, axis= 0)
        if self.first_batch:
            self.mean_test = mean
            self.var_test = var
            self.first_batch = False
        self.mean_train = mean
        self.var_train = var
        gamma = np.expand_dims(self.weights, 0)
        bias = np.expand_dims(self.bias, 0)

        if not self.testing_phase:
            x_tilde = (input_tensor - mean)/np.sqrt((var + eps))
            self.moving_average_estimation()
        else:
            x_tilde = (input_tensor-self.mean_test)/np.sqrt(self.var_test+ eps)

        
        y = gamma * x_tilde+bias
        self.x_tilde = x_tilde
        
        if is_conv:
            y = self.reformat(y)
            
        return y
    
    def backward(self, error_tensor):
        """
        compute_bn_gradient parameters: 
        -----
        input_tensor: input of the batch normalization
        weight tensor: diravative respect to the batch input
        mean: mean over batch
        var: variation over batch"""
        is_conv = False
        if error_tensor.ndim == 4:
            is_conv = True
            error_tensor = self.reformat(error_tensor)
            self.input_tensor = self.reformat(self.input_tensor)

        self.gradient_weights = np.sum(error_tensor*self.x_tilde, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        error = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean_train, self.var_train)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        if is_conv:
            error = self.reformat(error)
        return error
    
    def reformat(self, tensor):
        if tensor.ndim == 4:
            b, c, h, w = tensor.shape
            self.conv_shape = tensor.shape
            # print("original", tensor.shape)

            tensor = np.reshape(tensor, (b, c, h*w), "C")
            tensor = np.permute_dims(tensor, [0, 2, 1])
            tensor = np.reshape(tensor, (b*h*w, c), "F")
            
        
        else:
            b, c, h, w = self.conv_shape
            tensor = np.reshape(tensor, (b, h*w, c), "F")
            tensor = np.permute_dims(tensor, [0, 2, 1])
            tensor = np.reshape(tensor, (b, c, h, w), "C")
            # print("after", tensor.shape)

        return tensor