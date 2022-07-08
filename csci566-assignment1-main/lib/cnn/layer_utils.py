from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        output_shape[0] = input_size[0]
        output_shape[1] = int((input_size[1]+2*self.padding - self.kernel_size)/self.stride +1)
        output_shape[2] = int((input_size[2]+2*self.padding - self.kernel_size)/self.stride +1)
        output_shape [3] = self.number_filters 
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        """Non-Vectorized implementation:
        self.meta = img
        if self.padding !=0:
            img = np.pad(img,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)),'constant',constant_values=0)
        #print(img.shape)
        pass
        output = np.zeros(output_shape)
        #iterate over output dimensions, moving by self.stride to create the output
        for i in range(img.shape[0]):
            #print(i)
            itemp = img[i]
            for h in range(output_height):
                for w in range(output_width):
                    hstrt = h * self.stride
                    hend = hstrt + self.kernel_size
                    wstrt = w*self.stride
                    wend = wstrt+self.kernel_size
                    
                    for c in range(output_shape[-1]):
                        islice = itemp[hstrt:hend,wstrt:wend]
                        weights = self.params[self.w_name][:,:,:,c]
                        biases = self.params[self.b_name][c]
                        s = islice*weights
                        z = np.sum(s)
                        z = z+float(biases)
                        output[i,h,w,c] = z
        """
        #Vectorized implementation
        self.meta = img
        #pad img
        img = np.pad(img, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), 'constant', constant_values=0)
        output = np.zeros(output_shape)
        for h in range(output_height):
            for w in range(output_width):
                ih = h * self.stride
                iw = w * self.stride
                weights = self.params[self.w_name]
                bias = self.params[self.b_name]
                islice = img[:, ih : ih + self.kernel_size, iw : iw + self.kernel_size, :]
                islice = islice.reshape(*islice.shape, 1)
                output[:, h, w, :] = np.sum(np.multiply(weights, islice) , axis = (1,2,3)) + bias
        
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        """Non-Vectorized implementation:
        (m,nhprev,nwprev,ncprev) = self.meta.shape
        (f,f,ncprev,nc) = self.params[self.w_name].shape
        (m,nh,nw,nc) = dprev.shape
        
        daprev = np.zeros_like(self.meta)
        #print(daprev.shape)
        dw = np.zeros_like(self.params[self.w_name])
        db = np.zeros_like(self.params[self.b_name])
        
        padmeta = self.meta
        daprevpad = daprev
        if self.padding !=0:
            padmeta = np.pad(self.meta,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)),'constant',constant_values=0)
            daprevpad = np.pad(daprev,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)),'constant',constant_values=0)
        
        for i in range(m):
            
            atemp = padmeta[i,:,:,:]
            datemp = daprevpad[i,:,:,:]
            
            for h in range(nh):
                for w in range(nw):
                    for c in range(nc):
                        vst = h*self.stride
                        vend=vst+self.kernel_size
                        hst=w*self.stride
                        hend=hst+self.kernel_size
                        
                        aslice = atemp[vst:vend,hst:hend]
                        
                        datemp[vst:vend,hst:hend,:] += self.params[self.w_name][:,:,:,c]*dprev[i,h,w,c]
                        dw[:,:,:,c] += aslice*dprev[i,h,w,c]
                        db[c]+=dprev[i,h,w,c]
            #print(daprev.shape)
            #print(daprevpad.shape)
            if self.padding!=0:
                daprev[i,:,:,:]=daprevpad[i,self.padding:-self.padding,self.padding:-self.padding,:]
            else:
                daprev[i,:,:,:]=daprevpad[i,:,:,:]
        self.grads[self.w_name] = dw
        self.grads[self.b_name] = db
        dimg=daprev
        pass
        """
    
        
        output_shape = self.get_output_size(img.shape)
        _, output_height, output_width, _ = output_shape

        self.grads[self.w_name] = np.zeros_like(self.params[self.w_name])
        self.grads[self.b_name] = np.zeros_like(self.params[self.b_name])

        dimg = np.zeros_like(img)
        img = np.pad(img, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)), 'constant', constant_values = (0,0))
        dpad = np.zeros_like(img)
        batch_size = img.shape[0]
        num_filters = self.number_filters

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size

                islice = img[:, h_start:h_end, w_start:w_end, :].reshape(batch_size, self.kernel_size, self.kernel_size, img.shape[-1], 1) #N,k,k,C # 1, k, k, C || k, k, C
                dprev_reshaped = dprev[:, i, j, :].reshape(batch_size, 1, 1, 1, num_filters)
                #Vectorized for all weights
                dpad[:, h_start:h_end, w_start:w_end, :] += np.sum(self.params[self.w_name] * dprev_reshaped, axis = 4) # k, k, C, F || k, k, C
                self.grads[self.w_name] += np.sum(islice * dprev_reshaped, axis = 0)
                self.grads[self.b_name] += np.sum(dprev_reshaped, axis = (0,1,2,3))
        if self.padding > 0:
            #Resorting back to the OG input image size
            dimg = dpad[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dimg = dpad
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        """Non-vectorized implementations
        self.meta = img
        (m,nhprev,nwprev,ncprev) = img.shape
        nh = int(1 + (nhprev-self.pool_size)/self.stride)
        nw = int(1+ (nwprev-self.pool_size)/self.stride)
        nc=ncprev
        
        yolo = np.zeros((m,nh,nw,nc))
        for i in range(m):
            for h in range(nh):
                for w in range(nw):
                    for c in range(nc):
                        
                        vst = h*self.stride
                        vend=vst+self.pool_size
                        hst=w*self.stride
                        hend=hst+self.pool_size
                        
                        sliceo=img[i,vst:vend,hst:hend,c]
                        yolo[i,h,w,c] = np.max(sliceo)
        output=yolo"""
        self.meta = img
        batch_size, hin, win, c = img.shape
        hout = int( (hin - self.pool_size) / self.stride) + 1
        wout = int( (win - self.pool_size) / self.stride) + 1
        output = np.zeros((batch_size, hout, wout, c))
        for h in range(hout):
            for w in range(wout):
                h_in_1 = h * self.stride
                w_in_1 = w * self.stride
                img_slice = img[:, h_in_1 : h_in_1 + self.pool_size, w_in_1 : w_in_1 + self.pool_size, :]
                output[:, h, w, :] = np.max(img_slice, axis = (1, 2))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        """Non Vectorized implementation
        (m,nhprev,nwprev,ncprev) = img.shape
        m,nh,nw,nc=dprev.shape
        
        daprev = np.zeros_like(img)
        for i in range(m):
            itemp = img[i,:,:,:]
            for h in range(nh):
                for w in range(nw):
                    for c in range(nc):
                        vst = h*self.stride
                        vend=vst+self.pool_size
                        hst=w*self.stride
                        hend=hst+self.pool_size
                        
                        islice = itemp[vst:vend,hst:hend,c]
                        #print(islice.shape)
                        #print(np.max(islice))
                        #print(islice==np.max(islice))
                        mask = (islice==np.max(islice))
                        #print(mask.shape)
                        daprev[i,vst:vend,hst:hend,c] += mask*dprev[i,h,w,c]
        dimg = daprev"""
        batch_size , height_out, width_out, channels = dprev.shape
        for i in range(height_out):
            for j in range(width_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size
                slice = img[:, h_start:h_end, w_start:w_end, :] #N, k, k, C
                mask = slice == np.max(slice, axis = (1, 2)).reshape((batch_size, 1, 1, channels))
                dimg[:, h_start:h_end, w_start:w_end, :] += mask * dprev[:, i, j, :].reshape(batch_size, 1, 1, channels)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
