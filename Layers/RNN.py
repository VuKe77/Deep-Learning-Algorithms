#%%Debugging
print("DELETE ME FOR RUNNING TESTS")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#%%
import numpy as np
from Layers.Base import*
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid




class RNN(BaseLayer):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__(trainable=True)

        self._hidden_size = hidden_size
        self._hidden_state = np.zeros(shape=hidden_size)
        self._input_size = input_size
        self._output_size = output_size

        self._memorize = False

        #Define matrix transfomation at input(Wxh,Whh) as one Fully connected layer
        x1_shape = input_size+hidden_size #Concatenated x and h vectors
        self._fc1 = FullyConnected(x1_shape,hidden_size) #We want output to be size of hidden state

        #Define matrix transformation at output(Why)
        self._fc2 = FullyConnected(hidden_size,output_size)

        #Define tanh activation
        self._tnh_activation = TanH()
        
        #Define sigmoid activation
        self._sigmoid_activation = Sigmoid()

        #Arrays for saving values during forward pass
        self._fc1_outs = []
        self._fc2_outs = []
        self._tanh_outs=[] #This are also old states
        self._sigmoid_outs = []
        self.y_outputs =[]
        self.x_inputs = [] #Inputs to RNN

        #
        self._gradient_weights = None #Corresponds to fc1

        #Optimizer
        self._optimizer = None

    
    #Defining getter and setter for optimizer
    @property
    def optimizer(self):
        return self._optimizer 
    @optimizer.setter
    def optimizer(self,optimizer_instance):
        self._optimizer = optimizer_instance
    #Defining getter and setter for weights
    @property
    def weights(self):
        return self._fc1.weights
    @weights.setter
    def weights(self,new_weights):
        self._fc1.weights = new_weights
    #Defining getter for gradient_weights
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self,new):
        self._gradient_weights = new
    #Defining getter and setter for memorize
    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self,new_m):
        self._memorize = new_m

    def initialize(self,weights_initializer,bias_initializer):
        self._fc1.initialize(weights_initializer,bias_initializer)
        self._fc2.initialize(weights_initializer,bias_initializer)


    def forward(self, input_tensor):
        #Batch dimension is considered as time dimension

        #if memorize is True we want to perserve hidden state from last batch, else reinitialize to zeros
        if not self.memorize:
            self._hidden_state = np.zeros(shape=self._hidden_state.shape)


        #Clear previously used saved forward passes
        self._tanh_outs=[] 
        self._fc1_outs =[]
        self._fc2_outs=[]
        self.y_outputs=[]
        self.x_inputs = []
        self._fc1_inputs = []
        self._fc2_inputs = []
        
        self.T,L = input_tensor.shape #T-sequence length, L-features length

        #Iterate over timesteps
        for t in range(0,self.T):
            xin = input_tensor[t] #Extract input for current time step

            #Passing through FC layer and TanH
            x1 = np.concatenate((xin[np.newaxis,:],self._hidden_state[np.newaxis,:]),axis=1)
            fc1_out = self._fc1.forward(x1)
            #Save the input (with bias) for fc1 to use in backward pass
            self._fc1_inputs.append(self._fc1._input_tensor)
            tanh_out = self._tnh_activation.forward(fc1_out) #This is also new state
            self._hidden_state = tanh_out[0]
            #Passing through output FC layer and sigmoid
            fc2_out = self._fc2.forward(self._hidden_state[np.newaxis,:])
            self._fc2_inputs.append(self._fc2._input_tensor)
            y = self._sigmoid_activation.forward(fc2_out)[0]

            #Save forward pass values in order to use for backward pass
            self._tanh_outs.append(tanh_out) #Old states
            self._fc1_outs.append(fc1_out) 
            self._fc2_outs.append(fc2_out)
            self.y_outputs.append(y)
            self.x_inputs.append(xin)

        
        return np.array(self.y_outputs)

    
    def backward(self,error_tensor):

        #error_tensor is shape (B,M), each entry (i,M) corresponds to output value yi!

        #Accumulate weights
        fc2_gradient = None #Will be initialized
        fc1_gradient = None
        #Gradient of state h
        error_h = np.zeros(self._hidden_size) #Hidden state error backpropagates also, at the beginning it is zero
    
        #We need to collect all errors first in order to send them to previous layer
        error_x = []  #Error with respect to input

    
        T,L  = error_tensor.shape #Sequence length and error vector length

        for t in range(T-1,-1,-1): #Go backward
            error_y = error_tensor[t,:] 
            error_y = error_y[np.newaxis,:] #work with batches

            #Backward through sigmoid
            self._sigmoid_activation.activation = self.y_outputs[t]
            error_sigmoid = self._sigmoid_activation.backward(error_y)
            #error_sigmoid = error_sigmoid[np.newaxis,:] #Needed as batch to be compatible
            #Backward through fc2
            self._fc2.forward_pass = self._fc2_outs[t]
            self._fc2._input_tensor = self._fc2_inputs[t]
            error_fc2 = self._fc2.backward(error_sigmoid)
            if fc2_gradient is None:#Accumulate gradient w.rt weights
                fc2_gradient = self._fc2.gradient_weights.copy()
            else:
                fc2_gradient+=self._fc2.gradient_weights

            #Combine gradients as sum, because we have branching 
            error_combined = error_fc2 + error_h
            #Backward through tanh
            self._tnh_activation.activation = self._tanh_outs[t]
            error_tanh = self._tnh_activation.backward(error_combined)
            #Backward through fc1
            self._fc1.forward_pass = self._fc1_outs[t]
            self._fc1._input_tensor = self._fc1_inputs[t]
            error_fc1 = self._fc1.backward(error_tanh)
            if fc1_gradient is None:#Accumulate gradient w.rt weights
                fc1_gradient = self._fc1.gradient_weights.copy()
            else:
                fc1_gradient+=self._fc1.gradient_weights

            #---Extract gradients---#
            #Get gradients w.r.t state
            error_h = error_fc1[:,self._input_size:]
            
            #Get gradients w.r.t input
            error_x.append(error_fc1[:,:self._input_size])

        #Set gradient weights
        self.gradient_weights = fc1_gradient
        assert fc1_gradient.shape == self.weights.shape

        
        #Combine gradients w.r.t input and backpro *pagate as error to previous layer
        error_to_previous = np.stack(error_x, axis=1)#?
        error_to_previous = error_to_previous[:,::-1]
        error_to_previous = error_to_previous.squeeze(0)
        #Perform weight update
        if self.optimizer:
            self._fc1.weights = self.optimizer.calculate_update(self._fc1.weights, fc1_gradient)
            self._fc2.weights = self.optimizer.calculate_update(self._fc2.weights, fc2_gradient)

        return error_to_previous
    


            

            

            






if __name__ =="__main__":
    in_shape = 12
    hidden_shape = 10
    out_shape = 2
    T = 4

    model = RNN(12,10,2)
    x_in = np.ones((T,in_shape))
    y = model.forward(x_in)
    print(y.shape)

    model.backward(y)
    print("Ende")

# %%
