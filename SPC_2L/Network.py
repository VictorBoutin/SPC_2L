import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
from SPC_2L.DataTools import norm
from torch.autograd import Function

class LayerPC(nn.Module):

    def __init__(self, dico_shape=None, stride=1, pad=0, out_pad=0, a=1, b=1, v=1, v_size=None, bias=False,
                 transform=None, drop=None, normalize_dico=True, dico_load=None, seed=None, groups=1):

        super(LayerPC,self).__init__()

        self.params={'dico_shape': dico_shape,
                     'stride': stride,
                     'out_pad': out_pad,
                     'pad':pad,
                     'a': a,
                     'b': b,
                     'v': v,
                     'normalize_dico': normalize_dico,
                     'groups': groups}

        if seed is not None:
            torch.manual_seed(seed)
        self.dico_shape = dico_shape
        self.stride = stride
        self.out_pad = out_pad
        self.pad = pad
        self.a = a
        self.b = b

        if bias:
            self.bias = nn.Parameter(torch.zeros(1,dico_shape[0],1,1))
        else:
            self.bias = 0

        self.v_size = v_size
        if self.v_size is None:
            self.v = v
        else:
            self.v = v * torch.ones(1,v_size,1,1)
            self.v = nn.Parameter(self.v)

        self.groups = groups

        self.transform = transform
        self.drop = drop
        self.normalize_dico = normalize_dico
        self.dico = self.init_dico(dico_load)

    def init_dico(self, dico_load):
        if dico_load is None:
            if self.dico_shape is None:
                raise('you need either to define a Dictionary size (N x C x H x W) or to load an existing Dictionary')
            else:
                dico = torch.randn(self.dico_shape)
                if self.normalize_dico:
                    dico /= norm(dico)
                else:
                    dico *= np.sqrt(2/(self.dico_shape[-1]*self.dico_shape[-2]*self.dico_shape[-3]))
                return nn.Parameter(dico)
        else:
            return nn.Parameter(dico_load)

    def forward(self, x):

        return f.conv2d(x, self.dico, stride=self.stride, groups=self.groups, padding=self.pad) + self.bias


    def backward(self, x):
        if self.drop is not None:
            x = self.drop(self,x)
        x = f.conv_transpose2d(x, self.dico, stride=self.stride, padding=self.pad, output_padding=self.out_pad, groups=self.groups)
        return x
    def to_next(self, x):
        if self.transform is None:
            return self.v * x
        else:
            return self.v * self.transform.to_next(self, x)

    def to_previous(self, x):
        if self.transform is None:
            return x
        else:
            return self.transform.to_previous(self, x)


class Network(object):

    def __init__(self, layers, input_size, verbose=True):

        self.nb_layers = len(layers)
        self.layers = layers
        self.sparse_map_size = [None] * self.nb_layers
        self.input_size = input_size

        self.param_transform = []
        for i in range(self.nb_layers):
            if self.layers[i].transform is not None:
                to_append = (self.layers[i].transform.kernel_size, self.layers[i].transform.stride)
            else :
                to_append = (1,1)
            self.param_transform.append(to_append)


        if input_size is not None:

            input = torch.rand(input_size)
            network_structure = 'NETWORK STRUCTURE : \n Input : {0}'.format(input_size)

            for i in range(self.nb_layers):
                if i == 0:
                    sparse_map = self.layers[i].forward(input)
                else:
                    sparse_map = self.layers[i].forward(self.layers[i-1].to_next(sparse_map))
                self.sparse_map_size[i] = sparse_map.size()
                network_structure += '\n Layer {0} : {1}'.format(i + 1, list(sparse_map.size()))
                if self.layers[i].transform is not None:
                    network_structure += '\n Layer {0} transformed : {1}'.format(i + 1, list((self.layers[i].to_next(sparse_map).size())))

            if verbose:
                print(network_structure)

        if torch.cuda.is_available():
            self.layers = [self.layers[i].cuda() for i in range(self.nb_layers)]

    def project_dico(self, j, cpu=False):

        if cpu:
            dico = self.layers[j].dico.data.detach().cpu()
        else:
            dico = self.layers[j].dico.data.detach()

        for i in range(j-1,-1,-1):
            dico = self.layers[i].to_previous(dico)
            dico = self.layers[i].backward(dico)

        return dico

    def turn_off(self, i=None):

        if i is None:
            for i in range(self.nb_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = False
        else:
            for param in self.layers[i].parameters():
                param.requires_grad = False

    def turn_on(self, i=None):

        if i is None:
            for i in range(self.nb_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = True
        else:
            for param in self.layers[i].parameters():
                param.requires_grad = True

'''
class MaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(MaxPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
        if torch.cuda.is_available():
            self.pool = self.pool.cuda()
            self.unpool = self.unpool.cuda()

        #self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        #self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    def to_next(self, layer, x):

        x, _ = self.pool(x)
        return x

    def to_previous(self, layer, x):

        w = (x.size()[-2] - 1) * self.stride + self.kernel_size
        h = (x.size()[-1] - 1) * self.stride + self.kernel_size
        c_size = x.size()[0]
        d_size = layer.dico.data.size()[0]

        _, idx = self.pool(torch.randn(c_size, d_size, w, h).cuda())

        #if torch.cuda.is_available:
        #    _, idx = self.pool(torch.randn(c_size, d_size, w, h).cuda())
        #else:
        #    _, idx = self.pool(torch.randn(c_size, d_size, w, h))

        return self.unpool(x, idx)


## with padding
class AvgPool2d(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(AvgPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.AvgPool2d(kernel_size, stride=stride)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
        if torch.cuda.is_available():
            self.pool = self.pool.cuda()
            self.unpool = self.unpool.cuda()
        self.init = True

        # self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        # self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    def to_next(self, layer, x):

        if self.init == True:
            self.input_size = x.size()
            self.init = False
        x = self.pool(x)

        return x

    def to_previous(self, layer, x):
        x = f.interpolate(x, size=self.input_size[-1], mode='bilinear', align_corners=True)
        return x


class RescaleFeedback(Function):

    @staticmethod
    def forward(ctx, input, v):
        ctx.save_for_backward(input, v)
        return  v*input

    @staticmethod
    def backward(ctx, grad_output):
        input, v = ctx.saved_tensors
        grad_input = grad_v =  None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.div(v**2)
        if ctx.needs_input_grad[1]:
            grad_v = None#(grad_output*input).sum()


        return grad_input, grad_v


class MaxPool2d_b(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super(MaxPool2d_b, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
        if torch.cuda.is_available():
            self.pool = self.pool.cuda()
            self.unpool = self.unpool.cuda()
        self.init = True

        # self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        # self.unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    def to_next(self, layer, x):

        if self.init == True:
            self.input_size = x.size()
            self.init = False
        x, _ = self.pool(x)

        return x

    def to_previous(self, layer, x):
        x = f.interpolate(x, size=self.input_size[-1], mode='bilinear', align_corners=True)
        return x

'''