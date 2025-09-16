import numpy as np
from  Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self,input_size, output_size):
        super().__init__(trainable=True)
        self.input_size = input_size
        self.output_size = output_size
        self._weights = np.random.rand(input_size+1,output_size) #We need to add bias #TODO: delete this line
        self._optimizer=None #Default: no optimizer

    def initialize(self,weights_initializer,bias_initializer):
        #Shapes
        w_shape = (self.input_size, self.output_size)
        bias_shape = (1,self.output_size)
        #Weights
        _weights = weights_initializer.initialize(w_shape,self.input_size,self.output_size)
        bias = bias_initializer.initialize(bias_shape,1,self.output_size)
        #Concatenate
        self._weights = np.concatenate((_weights,bias),0)

        
    def forward(self,input_tensor):
        #input tensor is (batch_size,data_size)
        input_tensor = np.concatenate((input_tensor,np.ones((input_tensor.shape[0],1))),axis=1) #Adding bias term

        #Get forward pass value and input sensor to use in backward pass
        # print("fcl input_tensor shape: ", input_tensor.shape)
        # print("fcl weight shape: ", self.weights.shape)
        self.forward_pass = input_tensor @ self._weights
        self._input_tensor = np.array(input_tensor) 

        return self.forward_pass
    
    @property
    def forward_pass(self):
        return self._forward_pass 
    @forward_pass.setter
    def forward_pass(self,new_forward_pass):
        self._forward_pass = new_forward_pass
    
    #Defining getter and setter for optimizer
    @property
    def optimizer(self):
        return self._optimizer 
    @optimizer.setter
    def optimizer(self,optimizer_instance):
        self._optimizer = optimizer_instance
    @optimizer.deleter
    def optimzier(self):
        print("Delete optimizer")
        del self._optimizer

    #Defining getter and setter for gradient_weights
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self,gradient_weights_tensor):
        self._gradient_weights = gradient_weights_tensor

    #Defining getter for weights
    @property
    def weights(self):
        return self._weights
    @weights.setter
    def weights(self,new_weights):
        self._weights= new_weights
    
    




    def backward(self,error_tensor):
        #Error tensor comes from the front layer
        #We need to provide back layer with error_tensor*weight
        #This comes from chain rule(If I am right)

        #Calculate error to propagate back
        error_to_previous = error_tensor @ self._weights.T
        error_to_previous = error_to_previous[:, :-1]  # exclude bias

        
        #Calculate gradient w.r.t weights and update weights
        #self._gradient_weights = error_tensor @ self._input_tensor
        self.gradient_weights = self._input_tensor.T @ error_tensor

        if self.optimizer: #If optimizer is set -> update
            self._weights = self.optimizer.calculate_update(self._weights, self.gradient_weights)



        return error_to_previous

if __name__ == "__main__":

    a = np.ones((10,2))
    print(a.shape)
    b = np.ones((1,2))
    print(b.shape)
    c = np.concatenate((a,b),0)
    print(c.shape)

        
