import numpy as np
from Layers.Base import*


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape, trainable=False):
        """
        Implement max-pooling
        
        Parameter
        ---------
        stride_shape: tuple,
            2d tuple for 2d stride, (sh, sw)
        pooling_shape: array_like,
            pooling kernel size, (height, width)
        """
        super().__init__(trainable)
        self.stride_shape = stride_shape
        
        self.pooling_shape = pooling_shape
        self.location = None
        self.input = None
        self.forward_out = None

    def forward(self, input_tensor):
        """
        forward path
        Parameter
        ---------
        input_tensor: array_like,
            shape of (batch, channel, height, width)
            
        Return:
        ---------
        array_like,
            array after pooling, shape of (batch, channel, shrink_h, shrink_w)"""
        # dims and inits
        sh, sw = self.stride_shape
        kh, kw = self.pooling_shape
        batch, chan, ih, iw = input_tensor.shape
        outshape = (batch, chan, int((ih-kh)/sh)+1, int((iw-kw)/sw)+1)
        output = np.zeros(outshape)
        location = np.zeros((*outshape, 2))
        self.input = input_tensor


        # "valid" padding
        min = np.min(input_tensor)
        input_pad_upsamp = np.zeros((batch, chan, int(ih + kh -1), int(iw + kw -1)))*min
        start_h = int(kh/2)
        start_w = int(kw/2)
        input_pad_upsamp[:, :, start_h: start_h+ih, start_w: start_w +iw] = input_tensor

        # find max in FOV
        for b in range(batch):
            for c in range(chan):
                for h in range(outshape[-2]):
                    for w in range(outshape[-1]):
                        x = input_tensor[b, c, h*sh:h*sh+kh, w*sw:w*sw+kw]
                        # print(x, h, w)
                        output[b, c, h, w] = np.max(x)
                        location[b, c, h, w, :] = np.unravel_index(np.argmax(x), x.shape)

        self.location = location
        self.forward_out = output
        return output
    
    def backward(self, error_tensor):
        output = np.zeros((self.input.shape))
        batch, chan, height, width = self.input.shape
        kh, kw = self.pooling_shape
        sh, sw = self.stride_shape
        
        #print(error_tensor.shape, self.location.shape)

        for b in range(batch):
            for c in range(chan):
                for h in range(error_tensor.shape[-2]):
                    for w in range(error_tensor.shape[-1]):
                        lh, lw = self.location[b, c, h, w, :]
                        output[b, c, int(h*sh+lh), int(w*sw+lw)] += error_tensor[b, c, h, w]
        

        return output
