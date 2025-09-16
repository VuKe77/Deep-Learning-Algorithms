# %% Libs
import numpy as np
from Layers.Base import BaseLayer
from scipy.signal import convolve, correlate
from Optimization.Optimizers import *
from Layers.Initializers import *
from copy import copy

# %% Class conv
class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """ Convolution Layer, performing convolution operation
        Forward path: correlation 
        Backward path: correlation


        Parameter
        -------
        stride_shape: tuple or list,
            the number of pixel move along convolution. 
            Float is the number shift along all axial, 
            tuple is the shift along (horizontal, vertial)
        convolution_shape: tuple,
            colvolution shape, NOT KERNEL SHAPE!!!!
            1D conv with shape(input_channel, length); 
            for 2D conv, shape(input_channel, height, width)
        num_kernels: int,
            Kernel number, also equals to the number of channels in output
        
            

        Instance
        ----
        w_shape: 1d tuple,
            shape of weight, 1D tuple, (out_chan, in_chan, spatial), 
            weights are three or four dimensions
        self.weights: array_like,
            uniform initialized weights, 
            shape of (out_chan, in_chan, h, w) or (out_chan, in_chan, y)
        self.bias: array_like,
            bias term of network. Shape of (out_chan)
        self._gradient_weights: array_like, 
            default None, updated by properties
        self._gradient_bias: array_like,
            default None, updated by properties
        self._optimizer: optimizer, default None.
            the optimizer of back propagation. 
            Can be "Sgd", "SgdWithMomentum" or "Adam"
        self.forward_out: tuple,
            the output of this layer, before sampling, shape of (batch, out_chan, spatial)
            spatial dimension is the same as input
        self.forward_in: tuple,
            the input of this layer, shape of (batch, in_chan, spatial)

        Methods:
        ---
        forward(input_tensor), 
            return input for next layer.
        backward(error_tensor), 
            return error tensor for last layer.
        initializer(something)
        """
        super().__init__(trainable=True)

        #assert isinstance(stride_shape,int) or isinstance(stride_shape,tuple) , "Stride is not right shape"
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self._is2d = 1 if len(convolution_shape)==3 else 0

        self.num_kernels = num_kernels
        
        w_shape = (self.num_kernels,) + self.convolution_shape #I think weights should have size (num,channel,m,n)
        self.weights = np.random.rand(*w_shape)
        self.bias = np.random.rand(num_kernels) #I think one bias term for weight tensor 

        #Gradients
        self._gradient_weights = None
        self._gradient_bias = None

        self._optimizer = None
        self._opt_initialized= False #Seperate optimizer for weights and bias
        self.forward_out = ()
        self.forward_in = ()


    @property
    def gradient_weights(self):
        """
        Return:
        ---
        the gradient respect to the weights"""
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    
    @property
    def gradient_bias(self):
        """
        Return:
        ---
        the gradient respect to the bias"""
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
    
    @property
    def optimizer(self):
        """
        return
        ---
        optimizer,
            optimizer for weight 
        optimizer,
            optimizer for bias
        """
        # print("return all optimizers")
        if self._optimizer:
            return self._optimizer[0],self._optimizer[1]
        else:
            return None
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = (optimizer,copy(optimizer))
        # print("set optimizers")

    def forward(self,input_tensor):
        """convolution layer,
        use correlation instead of covolution to make it simpler (to be honest I have no idea why)

        Parameter 
        ---
        input_tensor: tuple,
            shape of (batch, chan, height, width)
            
        Return
        ---
        tuple,
            the result after one covolution, shape of (batch, chan, height width)
            serves as input_tensor for the next layer, shape of (batch, channel, spatial resolution).
            spatial resolution = (FOV - kernel_size) / stride_in_corresponing_direction + 1"""
        
        input_shape = input_tensor.shape
        self.forward_in = input_tensor

        #1D convolution
        if not self._is2d: 
            #Extract shapes
            k_o,k_c,k_h = self.weights.shape
            s_h = self.stride_shape[0]
            pad_h1 = int(k_h//2)
            pad_h2 = int(k_h - pad_h1 -1)

            #Output shape
            output_shape = (input_shape[0],
                            self.num_kernels,
                            int(np.floor((input_shape[2]-k_h+(pad_h1+pad_h2))/self.stride_shape[0])+1))
            output = np.zeros(output_shape)
            output_upsamp = np.zeros((input_shape[0],
                                    self.num_kernels,
                                    input_shape[2]))

            #Padded inout
            padded_input = np.pad(input_tensor,((0,0),(0,0),(pad_h1,pad_h2)),mode='constant') #dont padd batches and channels
            
            # #Convolve
            # for h in range(0,output_shape[2]):
            #     _x = padded_input[:,:,h*s_h:h*s_h+k_h]
            #     output[:,:,h]  = np.sum(_x[:,np.newaxis,:,:]*self.weights[np.newaxis,:,:,:],
            #                             axis=(2,3))
                
            # convolution without stride, later sampling
            for h in range(output_shape[2]):
                x = padded_input[:,:,h: h+k_h]
                output_upsamp[:,:,h]  = np.sum(x[:,np.newaxis,:,:]*self.weights[np.newaxis,:,:,:],
                                        axis=(2,3))
            output = output_upsamp[:, :, ::s_h]

            self.forward_out = output_upsamp
            output[:,:,:] += self.bias[np.newaxis,:,np.newaxis]   
            

        #2D convolution
        if self._is2d: 
            #Extract shapes
            k_o,k_c,k_h,k_w = self.weights.shape
            s_h,s_w = self.stride_shape
            batch, chan, h, w = input_tensor.shape

            #Assymetric padding considered, convolution shape is (C,M,N)
            pad_h1 = int(k_h//2)
            pad_h2 = int(k_h-1-pad_h1)
            pad_w1 = int(k_w//2)
            pad_w2 = int(k_w-1-pad_w1)

            #Output shape, padding is same
            output_shape = (input_shape[0],
                            self.num_kernels,
                            int(np.floor((input_shape[2]-k_h+(pad_h1+pad_h2))/self.stride_shape[0])+1),
                            int(np.floor((input_shape[3]-k_w+(pad_w1+pad_w2))/self.stride_shape[1])+1))
            output = np.zeros(output_shape)
            output_upsamp = np.zeros((input_shape[0], 
                                    self.num_kernels,
                                    input_shape[2],
                                    input_shape[3]))
            
            #Ignore multiline comments, I was trying some things during implementation
            # output_shape_no_stride = (input_shape[0].item(),
            #                 self.num_kernels,
            #                 int(np.floor(input_shape[2])),
            #                 int(np.floor(input_shape[3])))
            # output_no_stride = np.zeros(output_shape_no_stride)

            padded_input = np.pad(input_tensor,((0,0),(0,0),(pad_h1,pad_h2),(pad_w1,pad_w2)),
                                  mode='constant') #dont padd batches and channels
            
            """
            Problem with scipy package:
            the convolve function will return a 3D array, and i need to select the one i need
            I need to perform mode"same" in spatial domain and "valid" in channel"""
            #Try scipy's package
            # b_in,c_in,m_in,n_in = output_shape_no_stride
            # k_o,k_c,k_h,k_w = self.weights.shape
            # for b_i in range(b_in): #Go over every image
            #     for kernel_i in range(k_o): #Convolve every filter
            #         #print(output_no_stride.shape)
            #         #print(output_no_stride[b_i,kernel_i,:,:].shape)
            #         #print(padded_input[b_i,:,:,:].shape)
            #         #print(self.weights[kernel_i,:,:,:].shape)
            #         output_no_stride[b_i,kernel_i,:,:] = convolve(padded_input[b_i,:,:,:],self.weights[kernel_i,:,:,:],mode='valid')
            #         output_no_stride[b_i,kernel_i,:,:] += self.bias[kernel_i]
            # #Downsample with stride
            # output = output_no_stride[:,:,::self.stride_shape[0],::self.stride_shape[1]]


            # #Convolve by iterating over spatial dimension and broadcasting to channels and batches
            # for h in range(0,output_shape[2]):
            #     for w in range(0,output_shape[3]):
            #         _x = padded_input[:,:,h*s_h:h*s_h+k_h,w*s_w:w*s_w+k_w]
            #         output[:,:,h,w]  = np.sum(_x[:,np.newaxis,:,:,:]*self.weights[np.newaxis,:,:,:,:],
            #                                   axis=(2,3,4))

            # convolve without stride, shape of (batch, out_chan, h, w)    
            for h in range(input_shape[2]):
                for w in range(input_shape[3]):
                    x = padded_input[:, :, h:h+k_h, w: w+k_w]
                    output_upsamp[:, :, h, w] = np.sum(x[:, np.newaxis, :, :, :] * self.weights[np.newaxis, :, :, :, :], 
                                                       axis = (2, 3, 4))
                    
            output = output_upsamp[:, :, ::s_h, ::s_w]
            self.forward_out = output_upsamp

            # broadcasting the same bias to the whole channel
            # print(self.bias.shape, self.num_kernels)
            # print(output.shape)
            output[:,:,:,:] += self.bias[np.newaxis,:,np.newaxis,np.newaxis]   
            
        return output

    def backward(self, error_tensor):
        """Porpagating the error tensor to last layer
        multiplying error tensor to fliped weight tensor to conduct convolution operation
        
        Tips of padding and up sampling when introduce stride
        input layer X uses: input_pad  = np.ceil(input/stride)*kernel
        error tensor E uses: 

        Parameter:
        ---
        error_tensor: tuple,
            shape of (batch, chan, height, width)
        
        Return:
        ---
        tuple, 
            shape of last layer, should be also (batch, chan, height, width), same as input"""
        
        # 2D situation
        if error_tensor.ndim == 4:
            # dims & init
            batch, in_chan, height, width = self.forward_in.shape
            out_chan, in_chan, kh, kw = self.weights.shape
            s_h, s_w = self.stride_shape
            output = np.zeros(self.forward_in.shape) #output is X(input in forward pass)
            error_upsamp = np.zeros(self.forward_out.shape) #As if there is no stride 
            error_zeropad = np.zeros((batch, out_chan, height + kh-1, width+kw-1))
            input_zeropad = np.zeros((batch, in_chan, height+kh -1, width +kw-1))
            grad_weight = np.zeros(self.weights.shape)

            # rearrage kernel:
            kernel = np.zeros((in_chan, out_chan, kh, kw))
            kernel = np.swapaxes(self.weights, 0, 1)
            kernel = kernel[:, :, ::-1, ::-1]
            #kernel = np.flip(kernel, axis = 1)

            # enlarge error_tensor, so that error_upsamp.shape = input.shape
            error_upsamp[:, :, ::s_h, ::s_w] = error_tensor
            # zeropad, so that back propagation can achieve the same size
            start_h = int(kh/2)
            start_w = int(kw/2)
            error_zeropad[:, :, start_h: start_h+height, start_w: start_w+width] = error_upsamp
            input_zeropad[:, :, start_h:start_h+height, start_w: start_w+width] = self.forward_in

            # error correlate with weight, gradient respect to input
            for h in range(height):
                for w in range(width):
                    x = error_zeropad[:, :, h:h+kh, w: w+kw]
                    output[:, :, h, w] = np.sum(x[:, np.newaxis, :, :, :] * kernel[np.newaxis, :, :, :, :], axis=(2, 3, 4))
            
            # input correlate with weights, calculate gradient respect to weights
            # for h in range(kh):
            #     for w in range(kw):
            #         for i in range(in_chan):
            #             for out in range(out_chan):
            #                 x = input_zeropad[0, i, h:h+height, w:w+width]
            #                 grad_weight[out, i, h, w] = np.sum(x*error_upsamp[0, out, :, :])

            
            for kernel_i in range(self.weights.shape[0]): #For every kernel
                for h in range(error_tensor.shape[2]): #Go with kernel as in forward pass(error tensor shape is shape of output of forward pass)
                    for w in range(error_tensor.shape[3]):
                        _x = input_zeropad[:, :, h*s_h:h*s_h+kh, w*s_w: w*s_w+kw]
                        _e = error_tensor[:,kernel_i,h,w] #error corresponding to our kernel
                        a = _x*_e[:,np.newaxis,np.newaxis,np.newaxis]
                        b  = np.sum(a,axis=0)
                        grad_weight[kernel_i,:,:,:]+= b
            #Bias gradients
            #error shape: (B,out,h,w).For bias  add contributions for whole spatial domain(error*1->sum all errors), than sum over batches
            grad_bias = np.sum(error_tensor, axis=(0, 2, 3)) 



        #1D situation ()
        if error_tensor.ndim == 3:
            # dims & init
            batch, in_chan, length = self.forward_in.shape
            out_chan, in_chan, ky = self.weights.shape
            sy = self.stride_shape[0]
            output = np.zeros(self.forward_in.shape)
            error_upsamp = np.zeros(self.forward_out.shape)
            error_zeropad = np.zeros((batch, out_chan, length + ky-1))
            input_zeropad = np.zeros((batch, in_chan, length+ky-1))
            grad_weight = np.zeros(self.weights.shape)

            # rearrage kernel:
            kernel = np.zeros((in_chan, out_chan, ky))
            kernel = np.swapaxes(self.weights, 0, 1)
            kernel = kernel[:, :, ::-1]
            #kernel = np.flip(kernel, axis = 1)

            # enlarge error_tensor, so that error_upsamp.shape = input.shape
            error_upsamp[:, :, ::sy] = error_tensor
            # zeropad, so that back propagation can achieve the same size
            start = int(ky/2)
            error_zeropad[:, :, start: start+length] = error_upsamp
            input_zeropad[:, :, start:start+length] = self.forward_in

            # error correlate with weight, gradient respect to input
            for y in range(length):
                x = error_zeropad[:, :, y: y+ky]
                output[:, :, y] = np.sum(x[:, np.newaxis, :, :] * kernel[np.newaxis, :, :, :], axis=(2, 3))
        
            # input correlate with weights, calculate gradient respect to weights
            # for y in range(ky):
            #     for i in range(in_chan):
            #         for out in range(out_chan):
            #             x = input_zeropad[0, i, y: y+length]
            #             grad_weight[out, i, y] = np.sum(x*error_upsamp[0, out, :])

            #gradient weights
            for kernel_i in range(self.weights.shape[0]): #For every kernel
                for y in range(error_tensor.shape[2]): #Go with kernel as in forward pass(error tensor shape is shape of output of forward pass)
                        _x = input_zeropad[:, :, y*sy: y*sy+ky]
                        _e = error_tensor[:,kernel_i,y] #error corresponding to our kernel
                        a = _x*_e[:,np.newaxis,np.newaxis]
                        b  = np.sum(a,axis=0)
                        grad_weight[kernel_i,:,:]+= b
            #bias weights
            grad_bias = np.sum(error_tensor, axis=(0, 2)) 

        # update weight
        self.gradient_weights = grad_weight
        self.gradient_bias = grad_bias

        if np.any(self.optimizer):

            self.weights = self.optimizer[0].calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer[1].calculate_update(self.bias, self.gradient_bias)

        return output
    def initialize(self, weight_initializer, bias_initializer):
        """
        Define the initializers for weight and bias

        Parameter
        ----
        weight_initializer: Initializer type,
            Can be zero initializer, const, uniform, Xavier and He
        bias_initializer: Initializer type,
            same to weight_initializer

        Return
        ---
        None
        """
        if self._is2d:
            spatial_num = self.weights.shape[3]*self.weights.shape[2]
        else:
            spatial_num = self.weights.shape[2]
        self.weights = weight_initializer.initialize(self.weights.shape, self.weights.shape[1]*spatial_num, self.weights.shape[0]*spatial_num)
        self.bias = bias_initializer.initialize(self.bias.shape, self.weights.shape[1]*spatial_num, self.weights.shape[0]*spatial_num)
    


# %% test
if __name__ =="__main__":
    kernel_shape = (3, 2, 2)
    num_kernels = 4
    a = Conv((1, 1), kernel_shape, num_kernels)
    input_shape = (2,3,4, 4)
    input = np.ones(input_shape)
    padded_input = a.forward(input)
    kernel = a.weights
    #e = convolve(padded_input[0,:,:,:],kernel[0,:,:,:],mode='valid')


