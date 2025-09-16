#Skeleton to resolve import errors when running tests
import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        super().__init__()
        self.yhat = []

    def forward(self, prediction_tensor, label_tensor):
        """
        prediction_tensor: [batch, class]

        label_tensor: [batch, class] one-hot encoded
        """
        eps = np.finfo(float).eps

        self.yhat = prediction_tensor + eps
        out = -np.sum(label_tensor*np.log(self.yhat))
        #print(out)
        return out

    def backward(self, label_tensor):
        """
        label_tensor: [batch, class]
        """

        error_tensor = -label_tensor/(self.yhat)

        return error_tensor
