import numpy as np

class Optimizer():
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):

    def __init__(self,learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate 
    def calculate_update(self,weight_tensor,gradient_tensor):
        if self.regularizer:
            shrinked_weight = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weight = weight_tensor

        return shrinked_weight-self.learning_rate*gradient_tensor
    

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.step = 0.

    def calculate_update(self, weight_tensor, gradient_tensor):
        """By using equation
        v^k = momentum * v^(k-1) - eta(learning rate) * gradient, v is step
        weight = weight + v

            
        Parameter
        ------------
        weight_tensor: tuple, 
            weights stored in kernel, shape of (out_chan, in_chan, spatial)
        gradient_tensor: tuple,
            gradient respect to weights, shape of (out_chan, in_chan, spatial)

        Return
        ----------
        float,
            updated weight
        """
        if self.regularizer:
            shrinked_weight = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weight = weight_tensor 
        self.step = self.momentum_rate*self.step - self.learning_rate*gradient_tensor
        return shrinked_weight + self.step

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.step = 0.0
        self.r = 0.
        self.iter = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            shrinked_weight = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weight = weight_tensor
        # print(self.step)
        # print(gradient_tensor.shape)
        self.step = self.mu*self.step + (1-self.mu)*gradient_tensor
        # print(self.step)
        self.r = self.rho*self.r + (1-self.rho)*np.multiply(gradient_tensor, gradient_tensor)

        #bias correction
        step = self.step/(1-self.mu**self.iter)
        r = self.r / (1-self.rho**self.iter)
        self.iter +=1

        return shrinked_weight - self.learning_rate * step / (np.sqrt(r) + 1e-8)



























    
# class SgdWithMomentum():
#     def __init__(self,learning_rate=0.1,momentum=0.95):
#         self.learning_rate = learning_rate
#         self.momentum = momentum
#         self.v = 0

#     def calculate_update(self,weight_tensor,gradient_tensor):
#         self.v = self.momentum*self.v - self.learning_rate*gradient_tensor
#         w = weight_tensor + self.v
#         return w
# class Adam():
#     def __init__(self,learning_rate=0.001,mu=0.95,rho=0.999):
#         self.learning_rate = learning_rate
#         self.mu = mu
#         self.rho = rho
#         self.v = 0
#         self.r = 0
#         # self.iter = 1

#     def calculate_update(self,weight_tensor,gradient_tensor):
#         if isinstance(self.v, int) or self.v == 0:  # means self.v has not been initialized
#             self.v = np.zeros_like(weight_tensor)
#             self.r = np.zeros_like(weight_tensor)
#         self.v = self.mu*self.v + (1-self.mu)*gradient_tensor
#         self.r = self.rho*self.r + (1-self.rho)*(gradient_tensor**2) #Should be fine, if cant find error check it
#         #Bias correction
#         # self.v = self.v/(1-self.mu**self.iter)
#         # self.r = self.v/(1-self.rho**self.iter)
#         # self.iter+=1
#         w = weight_tensor  - self.learning_rate*self.v/(np.sqrt(self.r)+1e-5)
#         #!!!There is small difference between desired value and calculated cant figure out what is wrong
#         return w
    
    
    
    

# if __name__=="__main__":
#     #import sys
#     #sys.path.append('/home/vule/Documents/ASC/II semester/Deep Learning/Excersises/Excersise 1/exercise1_material/src_to_implement')
#     #from NeuralNetworkTests import TestOptimizers1
#     #test = TestOptimizers1()
#     a = Sgd()