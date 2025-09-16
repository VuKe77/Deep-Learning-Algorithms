# %% def
import numpy as np
import copy
from Layers.FullyConnected import *
from Layers.SoftMax import*
from Layers.ReLU import*
from Optimization.Optimizers import*
from Layers.Helpers import*
from Optimization.Loss import*

class NeuralNetwork():
    def __init__(self, optimizer,weights_initializer,bias_initializer):
        """
        Parameters: 
        - self.optimizer: sgd or None
        - self.loss -> list: stores loss after every iteration
        - self.layers -> list: layer structure of neural network
        - self.data_layer: dataloader
        - self.loss_layer: cross entropy loss
        - self._data_used -> boolean: flag of whether this batch of data has been used or not
        - self._current_data -> ndarray: data from this batch
        - self._current_label -> ndarray: labels from this batch
        """
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.current_data = None
        self.current_label = None
        

    def forward(self):
        # reason why it's not the same answer: I draw different datasets from dataloader
        # if self._data_used:
        #     self._current_data, self._current_label = self.data_layer.next()
        #     self._data_used = False

        self.current_data, self.current_label = self.data_layer.next()

        out = self.current_data
        for layer in self.layers:
            out = layer.forward(out)

        out = self.loss_layer.forward(out, self.current_label)
        self.loss.append(out)
                
        return out
    
    def backward(self):
        loss = self.loss_layer.backward(self.current_label)
        i = 0
        for layer in self.layers[::-1]:
            # print(layer, i)
            loss = layer.backward(loss)
            # i+=1
        

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)    
            layer.initialize(self.weights_initializer,self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        """
        iterations: number of iteration, integer
        """

        for iter in range(iterations):
            loss = self.forward()
            self.backward()
            self.data_used = True
    
    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

# %% test
if __name__ == "__main__":
    net = NeuralNetwork(optimizer=Sgd(1e-3))
    print(net.optimizer)
    net.data_layer = IrisData(20)
    fcl1 = FullyConnected(4, 3)
    # fcl1.optimizer = FullyConnected.optimizer.setter(copy.deepcopy(net.optimizer))
    # print(fcl1.optimizer)
    fcl2 = FullyConnected(3, 3)
    print(fcl2._optimizer)
    net.append_layer(fcl1)
    net.append_layer(ReLU())
    net.append_layer(fcl2)
    net.append_layer(SoftMax())

    # print(net.layers)
    # for layer in net.layers:
    #     data = layer.forward(data)
    #     print(layer)
    #     #print(data)

    out1 = net.forward()
    out2 = net.forward()

