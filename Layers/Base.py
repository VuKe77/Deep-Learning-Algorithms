import numpy as np


class BaseLayer:
    def __init__(self,trainable=False, testing_phase = False):
        self.trainable = trainable
        #TODO: Could add other parameters, like default weights...
        self.testing_phase = testing_phase
        