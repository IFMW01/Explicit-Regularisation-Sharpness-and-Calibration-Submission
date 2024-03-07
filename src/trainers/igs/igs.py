import numpy as np
from numpy import linalg as LA
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad, Variable
from torchvision import datasets, transforms
from pyhessian.hessian import hessian # Hessian computation

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

###### cj added
"""
def get_weight_tensors(model, reparam=False, bias=True):
    if reparam:
        if bias:
            return [model[0].th_weight, model[2].th_weight, 
                    model[0].th_bias, model[2].th_bias]
        return [model[0].th_weight, model[2].th_weight]
    if bias:
        return [model[0].weight, model[2].weight, model[0].bias, model[2].bias]
    return [model[0].weight, model[2].weight]
"""
###############

class QuotientManifoldTangentVector(object):
    """
    Container class for neural network parameter vectors represented
    on the Quotient Manifold
    """
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_components = len(layer_sizes)
        self.vec = [np.zeros(self.layer_sizes[i]) for i in range(self.n_components)]

    def set_vector(self, values, overwrite=False):
        if len(values) != self.n_components:
            if overwrite:
                self.vec = values
                self.n_components = len(values)
            else:
                raise Exception('This vector has been initialized with %d components \
                                 and %d components have been provided' % (self.n_components, len(values)))
        else:
            self.vec = values

    def get_vector(self):
        return self.vec

    def dot(self, b, weights):
        if not isinstance(b, QuotientManifoldTangentVector):
            raise Exception('Cannot find dot product with non QuotientManifoldTangentVector quantity')
        if not isinstance(weights, QuotientManifoldTangentVector):
            raise Exception('Weight vector is not a QuotientManifoldTangentVector')
        if b.n_components != self.n_components:
            raise Exception('Both QuotientManifoldTangentVectors need to have same number of components')
        if weights.n_components != self.n_components:
            raise Exception('Weight QuotientManifoldTangentVector needs to have same number of components')

        dot_prod = np.sum([np.dot(self.vec[i].ravel(),b.vec[i].ravel())/(np.linalg.norm(weights.vec[i].ravel())**2) for i in range(self.n_components)])
        return dot_prod

    def norm(self, weights):
        return np.sqrt(self.dot(self, weights))

    def normalize(self, weights):
        N = self.norm(weights)
        normed_vec = [z/N for z in self.vec]
        self.set_vector(normed_vec)

    def riemannian_hess_vec_prod(self, func, weight_tensors):
        if len(weight_tensors) != self.n_components:
            raise Exception('Mismatch between number of tangent vector components and weight tensors provided')

        grads = torch.autograd.grad(func, weight_tensors, create_graph=True)
        if torch.cuda.is_available():
            g_v_prod = sum([torch.dot(grads[i].view(-1), torch.Tensor(self.vec[i]).cuda().view(-1)) for i in range(self.n_components)])
        else:
            g_v_prod = sum([torch.dot(grads[i].view(-1), torch.Tensor(self.vec[i]).view(-1)) for i in range(self.n_components)])
        hess_vec_prod = torch.autograd.grad(g_v_prod, weight_tensors)
        norms = [torch.norm(var)**2 for var in weight_tensors]
        r_hess_vec_prod = [np.copy((hess_vec_prod[i]*norms[i]).data.cpu().numpy()) for i in range(len(norms))]
        return r_hess_vec_prod

def riemannian_hess_quadratic_form(tgt_vec, net, criterion, weights, data, labels):
    V = QuotientManifoldTangentVector(weights.layer_sizes)
    V.set_vector(tgt_vec.vec)

    if torch.cuda.is_available():
        inputs, targets = Variable(torch.Tensor(data).cuda()), Variable(torch.Tensor(labels).type(torch.LongTensor).cuda())
        net = net.cuda()
        criterion = criterion.cuda()
    else:
        inputs, targets = Variable(torch.Tensor(data)), Variable(torch.Tensor(labels).type(torch.LongTensor))
    loss = criterion(net(inputs), targets)
    
    #hv_prod = V.riemannian_hess_vec_prod(loss, get_weight_tensors(net, reparam, bias))
    hv_prod = V.riemannian_hess_vec_prod(loss, net.get_weight_tensors())
    HV = QuotientManifoldTangentVector(weights.layer_sizes)
    HV.set_vector(hv_prod)
    return V.dot(HV, weights)

def riemannian_power_method(v_init, max_iter, net, criterion, weights, data, labels, tol=1e-8):
    V_T = QuotientManifoldTangentVector(weights.layer_sizes)
    V_T.set_vector(v_init)
    V_Tp1 = QuotientManifoldTangentVector(weights.layer_sizes)
    errs = np.zeros(max_iter)
    for i in range(max_iter):
        if torch.cuda.is_available():
            inputs, targets = Variable(torch.Tensor(data).cuda()), Variable(torch.Tensor(labels).type(torch.LongTensor).cuda())
            net = net.cuda()
            criterion = criterion.cuda()
        else:
            inputs, targets = Variable(torch.Tensor(data)), Variable(torch.Tensor(labels).type(torch.LongTensor))
        loss = criterion(net(inputs), targets)
        net.zero_grad()
        #v_tp1 = V_T.riemannian_hess_vec_prod(loss, get_weight_tensors(net, reparam, bias))
        v_tp1 = V_T.riemannian_hess_vec_prod(loss, net.get_weight_tensors())
        V_Tp1.set_vector(v_tp1)
        V_Tp1.normalize(weights)

        err = np.sqrt(sum([np.linalg.norm(a.ravel() - b.ravel())**2 for a,b in zip(V_Tp1.vec,V_T.vec)]))/np.sqrt(sum([np.linalg.norm(z.ravel())**2 for z in V_T.vec]))
        V_T.set_vector(V_Tp1.vec)
        errs[i] = err
        if err < tol:
            break
    return V_T, errs


def vectorize(params_list):
    vecs = []
    for params in params_list:
        vecs.append(torch.cat([param.reshape(-1,1) for param in params]))
    return torch.cat(vecs, axis=1)

def gather_grads(network):
    dV = []
    for p in network.parameters():
        if not p.requires_grad:
            continue
        dV.append(p.grad.data)
    return dV

### reparametrization functions from Dinh et al. 2017
def get_eta(theta, a, b, theta_hat, eta_hat = 0):
    diff = theta - theta_hat
    if b == 0:
        return torch.abs(diff)**(1+2*a) * torch.sign(diff) + eta_hat
    return (diff**2 + b)**a * diff + eta_hat

def get_deta_dtheta(theta, a, b, theta_hat):
    diff = theta - theta_hat
    diff_sq = diff**2
    if b == 0:
        return (2*a+1)*diff_sq**a
    return (diff_sq + b)**(a-1) * ((2*a+1)*diff_sq + b)

def get_dtheta_deta_b0(eta, theta, a, theta_hat, eta_hat = 0):
    return (theta - theta_hat)/(eta - eta_hat)/(2*a+1)

def get_d2eta_dtheta2(theta, a, b, theta_hat):
    diff = theta - theta_hat
    diff_sq = diff**2
    return 2*diff*(diff_sq + b)**(a-2) * (a*(2*a+1)*diff_sq + 3*a*b)

def get_theta(eta, a, b, theta_hat, eta_hat = 0, eps=1e-6, maxiter = 1000):
    theta0 = theta_hat + 0.1
    dtheta_norm = 1.
    i=0
    while dtheta_norm > eps and i < maxiter:
        deta = get_eta(theta0, a, b, theta_hat, eta_hat) - eta
        deta_dtheta = get_deta_dtheta(theta0, a, b, theta_hat)
        #print(deta_dtheta)
        eta_diff = eta-eta_hat
        dtheta = -deta[eta_diff.abs()>eps]/deta_dtheta[eta_diff.abs()>eps]
        theta0[eta_diff.abs()>eps] += dtheta
        dtheta_norm = dtheta.norm()
        i+=1
    if i == maxiter:
        print("dtheta_norm: ", dtheta_norm)
    eta_diff = eta-eta_hat
    theta0[eta_diff.abs()<=eps] = theta_hat[eta_diff.abs()<=eps]
    return theta0

# some special analytic cases
def get_theta_a05(eta, b, theta_hat, eta_hat = 0):
    # assume a = 0.5
    assert b > 0
    eta_diff = eta - eta_hat
    return theta_hat + (0.5*(-b + (b*b + 4*eta_diff**2).sqrt())).sqrt() * torch.sign(eta_diff)

def get_theta_am05(eta, b, theta_hat, eta_hat = 0):
    # assume a = -0.5
    assert b > 0
    eta_diff = eta - eta_hat
    return theta_hat + (b*eta_diff**2/(1-eta_diff**2)).sqrt() * torch.sign(eta_diff)

def get_theta_b0(eta, a, theta_hat, eta_hat = 0, eps=1e-6):
    # assume b = 0
    assert a > -0.5
    eta_diff = eta - eta_hat
    ans = theta_hat + torch.abs(eta_diff)**(1/(2*a+1)) * torch.sign(eta_diff)
    ans[torch.abs(eta_diff)<eps] = theta_hat[torch.abs(eta_diff)<eps]
    return ans

### custom layer for reparametrization experiments
# Inherit from Function
class LinearFunctionReparamTheta(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, th_weight, th_weight_hat, a, b, th_bias=None, th_bias_hat=None):
        weight = get_eta(th_weight, a, b, th_weight_hat)
        if th_bias is not None and th_bias_hat is not None:
            bias = get_eta(th_bias, a, b, th_bias_hat)
        else:
            bias = None
        ctx.a = a
        ctx.b = b
        ctx.save_for_backward(input, weight, bias, th_weight, th_weight_hat, 
                              th_bias, th_bias_hat)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, th_weight, th_weight_hat, \
                              th_bias, th_bias_hat = ctx.saved_tensors
        grad_input = grad_th_weight = grad_th_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            deta_dth_weight = get_deta_dtheta(th_weight, ctx.a, ctx.b, th_weight_hat)
            grad_th_weight = grad_output.t().mm(input) * deta_dth_weight
        if bias is not None and ctx.needs_input_grad[5]:
            deta_dth_bias = get_deta_dtheta(th_bias, ctx.a, ctx.b, th_bias_hat)
            grad_th_bias = grad_output.sum(0) * deta_dth_bias

        return grad_input, grad_th_weight, None, None, None, grad_th_bias, None

# Note: .cuda() operation is not perfect (class variables are not loaded to cuda)
class LinearReparamTheta(nn.Module):
    def __init__(self, input_features, output_features, th_weight_hat, a, b, 
                 th_bias_hat=None, bias=True, cuda=True):
        super(LinearReparamTheta, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.th_weight_hat = th_weight_hat
        self.a = a
        self.b = b
        self.th_bias_hat = th_bias_hat
        if cuda:
            self.th_weight_hat = self.th_weight_hat.cuda()
            if bias:
                self.th_bias_hat = self.th_bias_hat.cuda()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.th_weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.th_bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('th_bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.th_weight, -0.1, 0.1)
        if self.th_bias is not None:
            nn.init.uniform_(self.th_bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunctionReparamTheta.apply(input, self.th_weight, 
                                               self.th_weight_hat, self.a, self.b, 
                                                self.th_bias, self.th_bias_hat)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, a={}, b={}, th_weight_hat={}, bias={}, th_bias_hat={}'.format(
            self.input_features, self.output_features, self.a, self.b, self.th_weight_hat, self.th_bias is not None, self.th_bias_hat
        )

# Inherit from Function
class LinearFunctionReparamEta_b0(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, eta_weight, th_weight_hat, a, eta_bias=None, th_bias_hat=None):
        b = 0
        weight = get_theta_b0(eta_weight, a, th_weight_hat)
        if eta_bias is not None and th_bias_hat is not None:
            bias = get_theta_b0(eta_bias, a, th_bias_hat)
        else:
            bias = None
        ctx.a = a
        ctx.b = b
        ctx.save_for_backward(input, weight, bias, eta_weight, th_weight_hat, 
                              eta_bias, th_bias_hat)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, eta_weight, th_weight_hat, \
                              eta_bias, th_bias_hat = ctx.saved_tensors
        grad_input = grad_eta_weight = grad_eta_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            #deta_dth_weight = get_deta_dtheta(weight, ctx.a, ctx.b, th_weight_hat)
            #grad_eta_weight = grad_output.t().mm(input) / deta_dth_weight
            dth_weight_deta = get_dtheta_deta_b0(eta_weight, weight, ctx.a, th_weight_hat)
            grad_eta_weight = grad_output.t().mm(input) * dth_weight_deta
        if bias is not None and ctx.needs_input_grad[4]:
            #deta_dth_bias = get_deta_dtheta(bias, ctx.a, ctx.b, th_bias_hat)
            #grad_eta_bias = grad_output.sum(0) / deta_dth_bias
            dth_bias_deta = get_dtheta_deta_b0(eta_bias, bias, ctx.a, th_bias_hat)
            grad_eta_bias = grad_output.sum(0) * dth_bias_deta

        return grad_input, grad_eta_weight, None, None, grad_eta_bias, None
    
class LinearReparamEta_b0(nn.Module):
    def __init__(self, input_features, output_features, th_weight_hat, a,  
                 th_bias_hat=None, bias=True, cuda=True):
        super(LinearReparamEta_b0, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.th_weight_hat = th_weight_hat
        self.a = a
        self.b = 0
        self.th_bias_hat = th_bias_hat
        if cuda:
            self.th_weight_hat = self.th_weight_hat.cuda()
            if bias:
                self.th_bias_hat = self.th_bias_hat.cuda()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.eta_weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.eta_bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('eta_bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.eta_weight, -0.1, 0.1)
        if self.eta_bias is not None:
            nn.init.uniform_(self.eta_bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunctionReparamEta_b0.apply(input, self.eta_weight, 
                                               self.th_weight_hat, self.a,  
                                                self.eta_bias, self.th_bias_hat)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, a={}, b={}, th_weight_hat={}, bias={}, th_bias_hat={}'.format(
            self.input_features, self.output_features, self.a, self.b, self.th_weight_hat, self.eta_bias is not None, self.th_bias_hat
        )

    
# Inherit from Function
class LinearFunctionReparamEta_a_05s(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, eta_weight, th_weight_hat, a, b, eta_weight_hat=0, 
                eta_bias=None, th_bias_hat=None, eta_bias_hat=0):
        if a == 0.5:
            weight = get_theta_a05(eta_weight, b, th_weight_hat, eta_weight_hat)
        elif a == -0.5:
            weight = get_theta_am05(eta_weight, b, th_weight_hat, eta_weight_hat)
        else:
            print("a is not 0.5 or -0.5")
        if eta_bias is not None and th_bias_hat is not None:
            if a == 0.5:
                bias = get_theta_a05(eta_bias, b, th_bias_hat, eta_bias_hat)
            elif a == -0.5:
                bias = get_theta_am05(eta_bias, b, th_bias_hat, eta_bias_hat)
            else:
                print("a is not 0.5 or -0.5")
        else:
            bias = None
        ctx.a = a
        ctx.b = b
        ctx.save_for_backward(input, weight, bias, th_weight_hat, 
                              th_bias_hat)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, th_weight_hat, \
                              th_bias_hat = ctx.saved_tensors
        grad_input = grad_eta_weight = grad_eta_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            deta_dth_weight = get_deta_dtheta(weight, ctx.a, ctx.b, th_weight_hat)
            grad_eta_weight = grad_output.t().mm(input) / deta_dth_weight
        if bias is not None and ctx.needs_input_grad[6]:
            deta_dth_bias = get_deta_dtheta(bias, ctx.a, ctx.b, th_bias_hat)
            grad_eta_bias = grad_output.sum(0) / deta_dth_bias

        return grad_input, grad_eta_weight, None, None, None, None, grad_eta_bias, None, None
    

class LinearReparamEta_a_05s(nn.Module):
    def __init__(self, input_features, output_features, th_weight_hat, a, b, eta_weight_hat=0, 
                 th_bias_hat=None, eta_bias_hat=0, bias=True, cuda=True):
        super(LinearReparamEta_a_05s, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.th_weight_hat = th_weight_hat
        self.a = a
        self.b = b
        self.th_bias_hat = th_bias_hat
        self.eta_weight_hat = eta_weight_hat
        self.eta_bias_hat = eta_bias_hat
        if cuda:
            self.th_weight_hat = self.th_weight_hat.cuda()
            if type(eta_weight_hat) != int:
                self.eta_weight_hat = self.eta_weight_hat.cuda()
            if bias:
                self.th_bias_hat = self.th_bias_hat.cuda()
                if type(eta_bias_hat) != int:
                    self.eta_bias_hat = self.eta_bias_hat.cuda()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.eta_weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.eta_bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('eta_bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.eta_weight, -0.1, 0.1)
        if self.eta_bias is not None:
            nn.init.uniform_(self.eta_bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunctionReparamEta_a_05s.apply(input, self.eta_weight, 
                                               self.th_weight_hat, self.a, self.b, self.eta_weight_hat, 
                                                self.eta_bias, self.th_bias_hat, self.eta_bias_hat)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, a={}, b={}, th_weight_hat={}, bias={}, th_bias_hat={}, eta_weight_hat={}, eta_bias_hat={}'.format(
            self.input_features, self.output_features, self.a, self.b, self.th_weight_hat, self.eta_bias is not None, self.th_bias_hat, 
            self.eta_weight_hat, self.eta_bias_hat
        )
    
### make identical reparametrized neural network
def set_network_param(model, model_reparam):
    # set linear network parameters (eta) from reparam network parameters (theta)
    isnan = False
    for layer2, layer in zip(model, model_reparam):
        if type(layer2) == nn.Linear and type(layer) == LinearReparamTheta:
            layer2.weight.data = get_eta(layer.th_weight.data, layer.a, layer.b, 
                                         layer.th_weight_hat)
            if torch.isnan(layer2.weight.data).sum() > 0:
                isnan = True
            if layer2.bias is not None and layer.th_bias is not None:
                layer2.bias.data = get_eta(layer.th_bias.data, layer.a, layer.b, 
                                       layer.th_bias_hat)
                if torch.isnan(layer2.bias.data).sum() > 0:
                    isnan = True
    return isnan

def set_network_reparamEta_a_05s(model_reparam, model):
    # set reparam network parameters (eta) from linear network parameters (theta)
    for layer, layer2 in zip(model, model_reparam):
        if type(layer) == nn.Linear and type(layer2) == LinearReparamEta_a_05s:
            # set th_weight_hat to be near to layer.weight 
            # (cannot be equal due to inf problem when calculating gradients)
            layer2.th_weight_hat = layer.weight.data.clone().detach()
            layer2.eta_weight.data = get_eta(layer.weight.data, layer2.a, layer2.b,
                                             layer2.th_weight_hat, 
                                             layer2.eta_weight_hat)
            if layer.bias is not None and layer2.eta_bias is not None:
                layer2.th_bias_hat = layer.bias.data.clone().detach()
                layer2.eta_bias.data = get_eta(layer.bias.data, layer2.a, layer2.b, 
                                       layer2.th_bias_hat, layer2.eta_bias_hat)
    return

### scale transformations
def layerwise_reparam(model, scale, param_id):
    for i, (name, param) in enumerate(model.named_parameters()):
        cur_layer = model[int(name.split('.')[0])]
        if i == param_id - 2 and cur_layer.bias is not None:
            param.data = param.data / scale
        if i == param_id - 1:
            param.data = param.data / scale
        if i == param_id:
            param.data = param.data * scale
    return

def nodewise_reparam(model, scale_scalar, param_id):
    # apply random positive scaling for all nodes in a specified layer
    for i, (name, param) in enumerate(model.named_parameters()):
        if i == param_id:
            scale = (torch.FloatTensor(param.data.shape[1]).uniform_() * scale_scalar).exp()
            if param.data.is_cuda:
                scale = scale.cuda()
    for i, (name, param) in enumerate(model.named_parameters()):
        cur_layer = model[int(name.split('.')[0])]
        if (i == param_id - 2 and cur_layer.bias is not None) or i == param_id - 1:
            if len(param.data.shape) > 1:
                param.data = param.data / scale.view(-1,1)
            else:
                param.data = param.data / scale
        if i == param_id:
            param.data = param.data * scale.view(1,-1)
    return

### jacobian, hessian, Fisher, gradient covariance calculation functions
def jacobian(y, x, create_graph=False, initial_reshape=True):          
    with torch.autograd.set_detect_anomaly(True):
        jac = []                                                                                          
        flat_y = y.reshape(-1)    
        grad_y = torch.zeros_like(flat_y)                                                                 
        for i in range(len(flat_y)):                                                                      
            grad_y[i] = 1.                                                                                
            grad_x, = torch.autograd.grad(flat_y, x, grad_y.clone(), retain_graph=True, create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def calculate_jacobian_wrt_params(network, X, y, criterion, create_graph=False):
    output = network(X)
    loss = criterion(output, y)
    J = []
    for param in network.parameters():
        Jtemp = jacobian(loss, param, create_graph=create_graph).reshape(-1)
        #print(Jtemp.shape)
        J.append(Jtemp)
    return torch.cat(J)

def calculate_jacobian_wrt_params_alldata(network, X, y, criterion_alldata, create_graph=False):
    output = network(X)
    loss = criterion_alldata(output, y)
    J = []
    N = X.shape[0]
    for param in network.parameters():
        Jtemp = jacobian(loss, param, create_graph=create_graph)
        #print(Jtemp.shape)
        J.append(Jtemp.reshape(N,-1))
    return torch.cat(J, axis=1)

def calculate_hessian_wrt_params(network, X, y, criterion):
    J = calculate_jacobian_wrt_params(network, X, y, criterion, create_graph=True)
    H = []
    len_J = len(J)
    for param in network.parameters():
        Htemp = jacobian(J, param).reshape(len_J,-1)
        #print(Jtemp.shape)
        H.append(Htemp)
    return torch.cat(H, axis=1)

def calculate_Fisher_by_sampling(network, X, criterion_alldata):
    output = network(X).detach()
    K = torch.distributions.categorical.Categorical(logits = output)
    y = K.sample()
    J_alldata = calculate_jacobian_wrt_params_alldata(network, X, y, criterion_alldata)
    
    return torch.mm(J_alldata.permute(1,0), J_alldata)

def calculate_Fisher(network, X, criterion_alldata):
    output = network(X).detach()
    p = nn.Softmax(dim=1)(output)
    for i in range(output.shape[-1]):
        y = torch.cuda.LongTensor([i]*output.shape[0])
        J_alldata = calculate_jacobian_wrt_params_alldata(network, X, y, criterion_alldata)
        if i == 0:
            F = torch.mm(J_alldata.permute(1,0), p[:,i:i+1]*J_alldata)
        else:
            F += torch.mm(J_alldata.permute(1,0), p[:,i:i+1]*J_alldata)
    return F

def calculate_Cov(network, X, y, criterion_alldata):
    # calculate gradient of the loss at each data point
    J_alldata = calculate_jacobian_wrt_params_alldata(network, X, y, criterion_alldata)
    Cov = torch.mm(J_alldata.permute(1,0), J_alldata)
    #LCov, VCov = torch.symeig(Cov, eigenvectors=True)
    return Cov

def calculate_Trace_Hess_pseudoinv_Cov(L_Hess, V_Hess, Cov, n=1, eps=0):
    # n: the number of eigenvalues to inverse (assume others to be zero)
    L_Hess_pinv = 1./(L_Hess+eps)
    L_Hess_pinv[:-n] = 0
    VTCV = torch.matmul(V_Hess.transpose(-2, -1), torch.matmul(Cov, V_Hess))
    return (L_Hess_pinv*VTCV.diag()).sum()

def calculate_Trace_Hess_pseudoinv_Cov2(L_Hess, V_Hess, Cov, n=1, eps=0):
    # n: the number of eigenvalues to inverse (assume others to be zero)
    L_Hess_pinv = 1./(L_Hess+eps)
    L_Hess_pinv[:-n] = 0
    Hpinv = torch.matmul(V_Hess, torch.matmul(L_Hess_pinv.diag_embed(), V_Hess.transpose(-2, -1)))
    return torch.trace(torch.matmul(Hpinv, Cov))

def get_information_matrices(model, X, y, criterion, criterion_alldata):
    H1 = calculate_hessian_wrt_params(model, X, y, criterion)
    C1 = calculate_Cov(model, X, y, criterion_alldata)
    F1 = calculate_Fisher(model, X, criterion_alldata)
    L_F1, V_F1 = torch.symeig(F1, eigenvectors=True)
    IGS1 = calculate_Trace_Hess_pseudoinv_Cov(L_F1, V_F1, C1, n=len(L_F1))
    #IGS1 = calculate_Trace_Hess_pseudoinv_Cov2(F1, C1, n=1)
    return H1, C1, F1, L_F1, V_F1, IGS1

def get_sqrt_diag_metric(model):
    sqrt_diag_metric = []
    for p in model.parameters():
        sqrt_diag_metric.append(p.data.norm() * torch.ones(p.data.numel()).cuda())
    return torch.cat(sqrt_diag_metric)

### Fisher-Rao norm (Liang et al. 2019)
def gather_params(network):
    dV = []
    for p in network.parameters():
        if not p.requires_grad:
            continue
        dV.append(p.data)
    return dV

def FisherRaoNorm1(model, F):
    param_vec = vectorize([gather_params(model)])
    return param_vec.t().mm(F).mm(param_vec).item()

def FisherRaoNorm2(model, X):
    m = torch.nn.Softmax(dim=1)
    output = model(X)
    p = m(output)
    L = 0
    #for layer in model:
    #for layer in model.network:
    for layer in model.module:
        if type(layer) == nn.Linear or type(layer) == LinearReparamTheta \
        or type(layer) == LinearReparamEta_b0 or type(layer) == LinearReparamEta_a_05s:
            L += 1
    return L**2 * (p*((output * p).sum(1).unsqueeze(-1) - output)**2).sum(1).mean().item()

def FisherRaoNorm2_emp(model, X, y):
    m = torch.nn.Softmax(dim=1)
    output = model(X)
    L = 0
    #for layer in model:
    #for layer in model.network:
    for layer in model.module:
        if type(layer) == nn.Linear or type(layer) == LinearReparamTheta \
        or type(layer) == LinearReparamEta_b0 or type(layer) == LinearReparamEta_a_05s:
            L += 1
    return L**2 * (((output * m(output)).sum(1) - output[range(len(output)), y])**2).mean().item()

### Petzka et al. 2021
# adapted from https://github.com/kampmichael/RelativeFlatnessAndGeneralization
def calculate_looutputss_on_data(model, loss, x, y):
    output = model(x)
    loss_value = loss(output, y)
    return output, np.sum(loss_value.data.cpu().numpy())

def calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha, normalize = False):
    shape = feature_layer.shape

    layer_jacobian = torch.autograd.grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    layer_jacobian_out = layer_jacobian[0]
    drv2 = Variable(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).cuda()
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = torch.autograd.grad(n_grd[neuron_j].cuda(), feature_layer, retain_graph=True)[0].cuda()
    print("got hessian")

    trace_neuron_measure = 0.0
    maxeigen_neuron_measure = 0.0
    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :].data.cpu().numpy()
            hessian = drv2[:,neuron_j,neuron_i,:]
            trace = np.trace(hessian.data.cpu().numpy())
            if normalize:
                trace /= 1.0*hessian.shape[0]
            trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * trace
            if neuron_j == neuron_i:
                eigenvalues = LA.eigvalsh(hessian.data.cpu().numpy())
                maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * eigenvalues[-1]
                # adding regularization term
                if alpha:
                    trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha
                    maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha

    return trace_neuron_measure, maxeigen_neuron_measure

#element_loss = torch.nn.CrossEntropyLoss(reduction='none')
#avg_loss = torch.nn.CrossEntropyLoss()

def calculate_relative_flatness(model, x_train, y_train,  
                                feature_layer_id, alpha, element_loss, avg_loss,
                                reparam_model = False, eps=1e-6):
    # hessian calculation for the layer of interest
    i = 0
    for p in model.parameters():
        if i == feature_layer_id:
            feature_layer = p
        i += 1

    # normalization of feature layer
    activation = model.feature_layer(x_train).data.cpu().numpy()
    #activation = model[:feature_layer_num+1](x_train).data.cpu().numpy()
    activation = np.squeeze(activation)
    sigma = np.std(activation, axis=0)

    j = 0
    #for name, p in model.named_parameters():
    #for name, p in model.network.named_parameters():
    for name, p in model.module.named_parameters():
        #print(name)
        #cur_layer = model[int(name.split('.')[0])]
        #cur_layer = model.network[int(name.split('.')[0])]
        cur_layer = model.module[int(name.split('.')[0])]
        if feature_layer_id - 2 == j or feature_layer_id - 1 == j:
            for i, sigma_i in enumerate(sigma):
                if sigma_i != 0.0:
                    if reparam_model:
                        if name.split('.')[1] == 'th_weight':
                            p.data[i] = get_theta(
                            get_eta(p.data[i], cur_layer.a, cur_layer.b, 
                                             cur_layer.th_weight_hat[i]) / sigma_i,
                            cur_layer.a, cur_layer.b, cur_layer.th_weight_hat[i], eps=eps)
                        elif name.split('.')[1] == 'th_bias':
                            p.data[i] = get_theta(
                            get_eta(p.data[i], cur_layer.a, cur_layer.b, 
                                             cur_layer.th_bias_hat[i]) / sigma_i,
                            cur_layer.a, cur_layer.b, cur_layer.th_bias_hat[i], eps=eps)
                        elif name.split('.')[1] == 'eta_weight':
                            if hasattr(model, 'use_eta_hat') and model.use_eta_hat:
                                cur_eta_hat = cur_layer.eta_weight_hat[i]
                            else:
                                cur_eta_hat = 0
                            # convert to theta
                            theta = get_theta(p.data[i], cur_layer.a, cur_layer.b, 
                                             cur_layer.th_weight_hat[i], 
                                          eta_hat = cur_eta_hat, eps=eps)
                            # apply normalization
                            theta = theta / sigma_i
                            cur_layer.th_weight_hat[i] = cur_layer.th_weight_hat[i] / sigma_i
                            # set eta
                            p.data[i] = get_eta(theta,
                            cur_layer.a, cur_layer.b, cur_layer.th_weight_hat[i], 
                                          eta_hat = cur_eta_hat)
                        elif name.split('.')[1] == 'eta_bias':
                            if hasattr(model, 'use_eta_hat') and model.use_eta_hat:
                                cur_eta_hat = cur_layer.eta_bias_hat[i]
                            else:
                                cur_eta_hat = 0
                            # convert to theta
                            theta = get_theta(p.data[i], cur_layer.a, cur_layer.b, 
                                             cur_layer.th_bias_hat[i], 
                                          eta_hat = cur_eta_hat, eps=eps)
                            # apply normalization
                            theta = theta / sigma_i
                            cur_layer.th_bias_hat[i] = cur_layer.th_bias_hat[i] / sigma_i
                            # set eta
                            p.data[i] = get_eta(theta,
                            cur_layer.a, cur_layer.b, cur_layer.th_bias_hat[i], 
                                          eta_hat = cur_eta_hat)
                    else:
                        p.data[i] = p.data[i] / sigma_i
        if feature_layer_id == j:
            for i, sigma_i in enumerate(sigma):
                if reparam_model:
                    if name.split('.')[1] == 'th_weight':
                        p.data[:,i] = get_theta(
                            get_eta(p.data[:,i], cur_layer.a, cur_layer.b, 
                                             cur_layer.th_weight_hat[:,i]) * sigma_i,
                        cur_layer.a, cur_layer.b, cur_layer.th_weight_hat[:,i], eps=eps)
                    elif name.split('.')[1] == 'eta_weight':
                        if hasattr(model, 'use_eta_hat') and model.use_eta_hat:
                            cur_eta_hat = cur_layer.eta_weight_hat[:,i]
                        else:
                            cur_eta_hat = 0
                        # convert to theta
                        theta = get_theta(p.data[:,i], cur_layer.a, cur_layer.b, 
                                             cur_layer.th_weight_hat[:,i], 
                                          eta_hat = cur_eta_hat, eps=eps)
                        # apply normalization
                        theta = theta * sigma_i
                        cur_layer.th_weight_hat[:,i] = cur_layer.th_weight_hat[:,i] * sigma_i
                        # set eta
                        p.data[:,i] = get_eta(theta,
                            cur_layer.a, cur_layer.b, cur_layer.th_weight_hat[:,i], 
                                          eta_hat = cur_eta_hat)
                    
                else:
                    p.data[:,i] = p.data[:,i] * sigma_i
            feature_layer = p
        j += 1
        
    train_output, train_loss_overall = calculate_loss_on_data(model, element_loss, 
                                                              x_train, y_train)
    train_loss = avg_loss(train_output, y_train)

    trace_nm, maxeigen_nm = calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, 
                                                                 alpha, normalize = False)
    return train_output, train_loss, trace_nm, maxeigen_nm


### code for experiments
def experiment(simple_model, reparam, X, y, criterion, criterion_alldata, eps=1e-4, print_vals=True):
    # simple_model: the model to calculate sharpness measures on
    # reparam (bool): the input model is reparametrized one or not
    # X, y: data used to calculate the sharpness measures
    # criterion: the loss function
    # criterion_alldata: the loss function that returns losses for individual data points
    # eps (float): a constant used to determine the number of eigenvalues to consider in calculating IGS
    # print_vals (bool): to print values or not
    
    N = X.shape[0]
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    start = time.time()

    H1, C1, F1, L_F1, V_F1, IGS1 = get_information_matrices(simple_model, X, y, criterion, criterion_alldata)
    L_H1, V_H1 = torch.symeig(H1, eigenvectors=True)
    print("information matrices calculated")
    print("information matrices elapsed time: ", time.time() - start)
    
    
    # Our measure
    start = time.time()
    IGS1_traj = []
    for i in range(len(L_F1)):
        IGS1_traj.append(calculate_Trace_Hess_pseudoinv_Cov(L_F1, V_F1, C1, 
                                                            n=i+1, eps=eps*L_F1[-1]).item())
    IGS2_traj = []
    for i in range(len(L_F1)):
        IGS2_traj.append(calculate_Trace_Hess_pseudoinv_Cov(L_F1, V_F1, C1, 
                                                            n=i+1, eps=0).item())
    print("IGS elapsed time: ", time.time() - start)
    if print_vals:
        print('H top eigval: {:f}'.format(L_H1[-1].item()))
        print('H trace: {:f}'.format(L_H1.sum().item()))
        print('IGS v1: {:f}'.format(IGS1_traj[-1]))
        print('IGS v2: {:f}'.format(IGS2_traj[len(L_F1[L_F1>1e-5*max(L_F1)])]))
    
    # Fisher-Rao norm
    start = time.time()
    frnorm1 = FisherRaoNorm1(simple_model, F1)/N
    frnorm2 = FisherRaoNorm2(simple_model, X)
    frnorm2_emp = FisherRaoNorm2_emp(simple_model, X, y)
    
    if print_vals:
        print('Fisher-Rao norm 1 (def): {:f}'.format(frnorm1))
        print('Fisher-Rao norm 2 (assumption): {:f}'.format(frnorm2))
        print('Fisher-Rao norm 2 empirical: {:f}'.format(frnorm2_emp))
    print("Fisher-Rao norm elapsed time: ", time.time() - start)
    
    # Rangamani et al.'s measure
    start = time.time()
    #W_tr = [np.copy(var.cpu().data.numpy()) for var in get_weight_tensors(simple_model, 
    #                                                                      reparam=reparam, bias=bias)]
    W_tr = [np.copy(var.cpu().data.numpy()) for var in simple_model.get_weight_tensors()]
    layer_sizes = [w.shape for w in W_tr]
    v_init = [np.random.normal(size=layer_sizes[i]) for i in range(len(layer_sizes))]
    
    W_trans = QuotientManifoldTangentVector(layer_sizes)
    W_trans.set_vector(W_tr)
    
    v_res_tr, errs_tr = riemannian_power_method(v_init, 1000, simple_model, criterion, 
                                            W_trans, X_np, y_np, 
                                            tol=1e-8)
    sp_norm_tr = riemannian_hess_quadratic_form(v_res_tr, simple_model, criterion, 
                                            W_trans, X_np, y_np)
    
    
    
    # Rangamani et al.'s measure using hessian
    d1_rang = get_sqrt_diag_metric(simple_model)
    H1_rang = d1_rang.view(-1,1) * H1 * d1_rang.view(1,-1)
    L_H1_rang, V_H1_rang = torch.symeig(H1_rang, eigenvectors=True)
    
    if print_vals:
        print('Rangamani et al. measure: {:f}'.format(sp_norm_tr))
        print('Rangamani et al. measure (using H): {:f}'.format(L_H1_rang[-1].item()))
    print("Rangamani et al. measure elapsed time: ", time.time() - start)
    
    
    # Petzka et al.'s measure (for a specific layer and assume all layers have bias parameters)
    # this method modifies parameter value hence should be calculated at the end
    start = time.time()
    feature_layer_id = 2  # '2.weight'
    #feature_layer_num = 1
    alpha = 0
    
    train_output, train_loss, trace_nm, maxeigen_nm = calculate_relative_flatness(simple_model, X, y, 
                                feature_layer_id, alpha, 
                                criterion_alldata, criterion,
                                reparam_model = reparam,
                                eps=1e-6)
    
    if print_vals:
        print('Petzka et al. measure (trace): {:f}'.format(trace_nm))
        print('Petzka et al. measure (max eigval): {:f}'.format(maxeigen_nm))
    print("Petzka et al. measure elapsed time: ", time.time() - start)
        
    res = {
        'frnorm1': frnorm1,
        'frnomr2': frnorm2,
        'frnorm2_emp': frnorm2_emp,
        'petzka_trace_nm': trace_nm, 
        'petzka_maxeigen_nm': maxeigen_nm,
        'rang_sp_norm_tr': sp_norm_tr,
        'rang_H': L_H1_rang[-1].item(),
        'IGS_traj': IGS1_traj,
        'IGS2_traj': IGS2_traj,
        'L_H': L_H1.cpu().numpy(),
        'L_F': L_F1.cpu().numpy()
    }
    
    return res

### code for faster experiments (do not obtain full hessian, do not calculate exact Fisher)
def get_hessian_measures(network, criterion, X, y, top_n, tol=1e-2):
    hessian_comp = hessian(network, criterion, data=(X, y), cuda=True)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(maxIter=100, tol=tol, top_n=top_n)
    traceH = hessian_comp.trace(maxIter=100, tol=tol)
    return top_eigenvalues, top_eigenvector, traceH

def get_information_matrices_fast(model, X, y, criterion, criterion_alldata):
    #H1 = calculate_hessian_wrt_params(model, X, y, criterion)
    C1 = calculate_Cov(model, X, y, criterion_alldata)
    F1 = calculate_Fisher_by_sampling(model, X, criterion_alldata)
    
    L_F1, V_F1 = torch.symeig(F1, eigenvectors=True)
    #IGS1 = calculate_Trace_Hess_pseudoinv_Cov(L_F1, V_F1, C1, n=len(L_F1))
    #IGS1 = calculate_Trace_Hess_pseudoinv_Cov2(F1, C1, n=1)
    return C1, F1, L_F1, V_F1

def IGS_approx(L_F_traj, L_C_traj, report_ranks):
    IGS_approx1 = [] # ratio of traces
    IGS_approx2 = [] # trace of ratios
    ratio = L_C_traj/L_F_traj
    for n in report_ranks:
        if n < len(L_F_traj):
            IGS_approx1.append(L_C_traj[-n-1:].sum()/L_F_traj[-n-1:].sum())
            IGS_approx2.append(ratio[-n-1:].sum())
    return IGS_approx1, IGS_approx2

def experiment_fast(simple_model, reparam, X, y, criterion, criterion_alldata, eps=1e-4, 
                    report_ranks = None, exact_fisher=False, print_vals=True, 
                    calculate_align = False):
    # report_ranks: list of integers for ranks to calculate Fisher pseudo-inverse
    # exact_fisher: True if IGS using exact Fisher required
    # calculate_align: True if eigensubspace alignment between the Covariance and Fisher required
    N = X.shape[0]
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    start = time.time()
    C1, F1, L_F1, V_F1 = get_information_matrices_fast(simple_model, X, y, 
                                                       criterion, criterion_alldata)
    
    top_n = 1
    tol = 1e-3
    top_eigenvalues, top_eigenvector, traceH = get_hessian_measures(simple_model, criterion, X, y, top_n, tol=tol)
    L_C1, V_C1 = torch.symeig(C1, eigenvectors=True)
    print("information matrices calculated")
    print("information matrices elapsed time: ", time.time() - start)
    
    L_F2 = []
    if exact_fisher:
        start = time.time()
        F2 = calculate_Fisher(simple_model, X, criterion_alldata)
        L_F2, V_F2 = torch.symeig(F2, eigenvectors=True)
        print("exact Fisher calculated")
        print("exact Fisher elapsed time: ", time.time() - start)
    
    # Our measure
    start = time.time()
    IGS1_traj = []
    FC_subspace_align_traj = []
    if report_ranks is None:
        report_ranks = []
        for const in [1e-2, 1e-3, 1e-4]:
            report_ranks.append(max(1, len(L_F1[L_F1 > const*max(L_F1)])))
    
    for i in report_ranks:
        IGS1 = calculate_Trace_Hess_pseudoinv_Cov(L_F1, V_F1, C1, 
                                                            n=i, eps=0).item()
        IGS1_traj.append(IGS1)
        
        if calculate_align:
            # calculate subspace alignment
            V1TV2 = V_F1[:,-i-1:].permute(1,0).mm(V_C1[:,-i-1:]).cpu().numpy()
            _, s, _ = np.linalg.svd(V1TV2)
            FC_subspace_align_traj.append((s**2).mean())
    IGS_approx1, IGS_approx2 = IGS_approx(L_F1, L_C1, report_ranks)
    
    print("IGS n: ", report_ranks[-1], ", elapsed time: ", time.time() - start)
    
    IGS2_traj = []
    report_ranks2 = []
    IGS2_approx1 = []
    IGS2_approx2 = []
    if exact_fisher:
        start = time.time()
        for const in [1e-2, 1e-3, 1e-4]:
            report_ranks2.append(max(1, len(L_F2[L_F2 > const*max(L_F2)])))
    
        for i in report_ranks2:
            IGS2 = calculate_Trace_Hess_pseudoinv_Cov(L_F2, V_F2, C1, 
                                                                n=i, eps=0).item()
            IGS2_traj.append(IGS2)
        IGS2_approx1, IGS2_approx2 = IGS_approx(L_F2, L_C1, report_ranks2)
        print("IGS2 n: ", report_ranks2[-1], ", elapsed time: ", time.time() - start)
        L_F2 = L_F2.cpu().numpy() # covert to numpy to save
    
    if print_vals:
        print('H top eigval: {:f}'.format(top_eigenvalues[0]))
        print('H trace: {:f}'.format(np.array(traceH).mean()))
        print('IGS: {:f}'.format(IGS1))
        if exact_fisher:
            print('IGS exact F: {:f}'.format(IGS2))
        #print('IGS v1: {:f}'.format(IGS1_))
        #print('IGS v2: {:f}'.format(IGS2_))
    
    
    # Fisher-Rao norm
    start = time.time()
    frnorm1 = FisherRaoNorm1(simple_model, F1)/N
    frnorm1_F2 = 0
    if exact_fisher:
        frnorm1_F2 = FisherRaoNorm1(simple_model, F2)/N
    frnorm2 = FisherRaoNorm2(simple_model, X)
    frnorm2_emp = FisherRaoNorm2_emp(simple_model, X, y)
    
    if print_vals:
        print('Fisher-Rao norm 1 (def): {:f}'.format(frnorm1))
        if exact_fisher:
            print('Fisher-Rao norm 1 (def, exact F): {:f}'.format(frnorm1_F2))
        print('Fisher-Rao norm 2 (assumption): {:f}'.format(frnorm2))
        print('Fisher-Rao norm 2 empirical: {:f}'.format(frnorm2_emp))
    print("Fisher-Rao norm elapsed time: ", time.time() - start)
    
    # Rangamani et al.'s measure
    start = time.time()
    #W_tr = [np.copy(var.cpu().data.numpy()) for var in get_weight_tensors(simple_model, 
    #                                                                      reparam=reparam, bias=bias)]
    W_tr = [np.copy(var.cpu().data.numpy()) for var in simple_model.get_weight_tensors()]
    layer_sizes = [w.shape for w in W_tr]
    v_init = [np.random.normal(size=layer_sizes[i]) for i in range(len(layer_sizes))]
    
    W_trans = QuotientManifoldTangentVector(layer_sizes)
    W_trans.set_vector(W_tr)
    
    v_res_tr, errs_tr = riemannian_power_method(v_init, 1000, simple_model, criterion, 
                                            W_trans, X_np, y_np, 
                                            tol=1e-8)
    sp_norm_tr = riemannian_hess_quadratic_form(v_res_tr, simple_model, criterion, 
                                            W_trans, X_np, y_np)
    
    if print_vals:
        print('Rangamani et al. measure: {:f}'.format(sp_norm_tr))
        #print('Rangamani et al. measure (using H): {:f}'.format(L_H1_rang[-1].item()))
    print("Rangamani et al. measure elapsed time: ", time.time() - start)
    
    # Petzka et al.'s measure (for a specific layer and assume all layers have bias parameters)
    # this method modifies parameter value hence should be calculated at the end
    start = time.time()
    feature_layer_id = 2  # '2.weight'
    #feature_layer_num = 1
    alpha = 0
    
    train_output, train_loss, trace_nm, maxeigen_nm = calculate_relative_flatness(simple_model, X, y, 
                                feature_layer_id, alpha, 
                                criterion_alldata, criterion,
                                reparam_model = reparam,
                                eps=1e-6)
    
    if print_vals:
        print('Petzka et al. measure (trace): {:f}'.format(trace_nm))
        print('Petzka et al. measure (max eigval): {:f}'.format(maxeigen_nm))
    print("Petzka et al. measure elapsed time: ", time.time() - start)
    
    res = {
        'frnorm1': frnorm1,
        'frnorm1_F2': frnorm1_F2,
        'frnomr2': frnorm2,
        'frnorm2_emp': frnorm2_emp,
        'petzka_trace_nm': trace_nm, 
        'petzka_maxeigen_nm': maxeigen_nm,
        'rang_sp_norm_tr': sp_norm_tr,
        #'rang_H': L_H1_rang[-1].item(),
        'IGS_traj': IGS1_traj,
        'IGSapprox1_traj': IGS_approx1, 
        'IGSapprox2_traj': IGS_approx2,
        'trace_H': np.array(traceH).mean(),
        'top_eigval_H': top_eigenvalues[0],
        'L_F': L_F1.cpu().numpy(),
        'L_C': L_C1.cpu().numpy(),
        'FC_subspace_align_traj': FC_subspace_align_traj,
        'report_ranks': report_ranks,
        'IGS2_traj': IGS2_traj,
        'IGS2approx1_traj': IGS2_approx1, 
        'IGS2approx2_traj': IGS2_approx2,
        'L_F2': L_F2,
        'report_ranks2': report_ranks2
    }
    
    return res

### codes for calculating IGS on larger models
def orthnormal_vec(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    if len(v_list) > 0:
        V = torch.cat(v_list, -1)
        w = w - V.mm(V.permute(1,0).mm(w))
    unnorm_w_norm = w.norm()
    return w/(unnorm_w_norm+1e-6), unnorm_w_norm

def eigproblem_JTJ(J, maxIter=100, tol=1e-3, top_n=1, printinfo = False):
    assert top_n >= 1

    eigenvalues = []
    eigenvectors = []

    computed_dim = 0
    unnorm_v_norms = []
    spurious_dim = []

    while computed_dim < top_n:
        eigenvalue = None
        
        v = torch.randn(J.shape[1],1).cuda()
        v = v/(v.norm()+1e-6)
        for i in range(maxIter):
            v, unnorm_v_norm = orthnormal_vec(v, eigenvectors)
            #print(v.norm())
            JTJv = J.permute(1,0).mm(J.mm(v))
            tmp_eigenvalue = (JTJv * v).sum().cpu().item()

            v = JTJv / (JTJv.norm()+1e-6)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                       1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        
        if printinfo:
            print("dim: %d, iter: %d, unnorm_v_norm: %f, error: %f"%(computed_dim+1, i, 
                                                                      unnorm_v_norm,
                                                  abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)))
        eigenvalues.append(eigenvalue/J.shape[0])
        eigenvectors.append(v)
        # if this value is far less than 1.0 or larger than 1.0, the eigvals are under numerical error...
        unnorm_v_norms.append(unnorm_v_norm)
        if unnorm_v_norm < 0.9 or unnorm_v_norm > 1.0:
            spurious_dim.append(computed_dim)
        computed_dim += 1
    return eigenvalues, eigenvectors, spurious_dim, unnorm_v_norms

def eigproblem_JTJ_multi(Js, ps, maxIter=100, tol=1e-3, top_n=1, printinfo = False):
    assert top_n >= 1

    eigenvalues = []
    eigenvectors = []

    computed_dim = 0
    unnorm_v_norms = []
    spurious_dim = []

    while computed_dim < top_n:
        eigenvalue = None
        
        v = torch.randn(Js[0].shape[1],1).cuda()
        v = v/(v.norm()+1e-6)
        for i in range(maxIter):
            v, unnorm_v_norm = orthnormal_vec(v, eigenvectors)
            #print(v.norm())
            for k in range(len(Js)):
                if k == 0:
                    JTJv = Js[k].permute(1,0).mm(ps[:,k:k+1]*Js[k].mm(v))
                else:
                    JTJv += Js[k].permute(1,0).mm(ps[:,k:k+1]*Js[k].mm(v))
            
            tmp_eigenvalue = (JTJv * v).sum().cpu().item()

            v = JTJv / (JTJv.norm()+1e-6)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                       1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        if printinfo:
            print("dim: %d, iter: %d, unnorm_v_norm: %f, error: %f"%(computed_dim+1, i, 
                                                                      unnorm_v_norm,
                                                  abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)))
        eigenvalues.append(eigenvalue/Js[0].shape[0])
        eigenvectors.append(v)
        # if this value is far less than 1.0 or larger than 1.0, the eigvals are under numerical error...
        unnorm_v_norms.append(unnorm_v_norm)
        if unnorm_v_norm < 0.9 or unnorm_v_norm > 1.0:
            spurious_dim.append(computed_dim)
        computed_dim += 1
    return eigenvalues, eigenvectors, spurious_dim, unnorm_v_norms

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def calculate_jacobian_wrt_params_alldata_dataloader(model, X_all, y_all, criterion):
    training_data = CustomDataset(X_all, y_all)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
    J_1 = []
    for k, (X, y) in enumerate(train_dataloader):
        model.zero_grad()
        X = X.cuda()
        y = y.cuda()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        J = vectorize([gather_grads(model)])
        J_1.append(J)

    #print(len(J_1),J_1[0].shape)
    return torch.cat(J_1, dim=-1).permute(1,0)

def calculate_IGS_largemodel(model, X_F, X, y, criterion, tol, top_n, exact_fisher=False, eps=0, expand=False):
    # calculate Fisher eigenproblem
    output = model(X_F).detach()
    if exact_fisher:
        ps = nn.Softmax(dim=1)(output)
        Js = []
        trF = 0
        for i in range(output.shape[-1]):
            y_F = torch.cuda.LongTensor([i]*output.shape[0])
            Js.append(calculate_jacobian_wrt_params_alldata_dataloader(model, X_F, y_F, criterion))
            trF += ((ps[:,i:i+1].sqrt()*Js[i])**2).sum().item() / X_F.shape[0]

        L, V, spurious_dim, _ = eigproblem_JTJ_multi(Js, ps, maxIter=1000, tol=tol, top_n=top_n, printinfo = False)
        L_F_gram = []
    else:
        K = torch.distributions.categorical.Categorical(logits = output)
        y_F = K.sample()
        J = calculate_jacobian_wrt_params_alldata_dataloader(model, X_F, y_F, criterion)

        L, V, spurious_dim, _ = eigproblem_JTJ(J, maxIter=1000, tol=tol, top_n=top_n, printinfo = False)
        trF = (J**2).sum().item()/X_F.shape[0]
        # exact eigvals
        #L_F_gram, _ = torch.symeig(J.mm(J.permute(1,0)) / X_F.shape[0], eigenvectors=False)
        # print(J.shape)
        # print(J.mm(J.permute(1,0)).shape)
        L_F_gram = torch.linalg.eigvalsh(J.mm(J.permute(1,0)) / X_F.shape[0])
        L_F_gram = L_F_gram.cpu().numpy()
    # calculate Cov 
    J_alldata = calculate_jacobian_wrt_params_alldata_dataloader(model, X, y, criterion)

    trC = (J_alldata**2).sum().item()/X.shape[0]
    total_grad = J_alldata.mean(dim=0, keepdim=True)
    # exact eigvals
    #print(J_alldata.shape, X.shape, vectorize([gather_grads(model)]).shape, (J_alldata.mm(J_alldata.permute(1,0)) ).shape)
    # L_C_gram, _ = torch.symeig(J_alldata.mm(J_alldata.permute(1,0)) / X.shape[0], eigenvectors=False)
    
    L_C_gram = torch.linalg.eigvalsh(J_alldata.mm(J_alldata.permute(1,0)) / X.shape[0])
        
    L_C_gram = L_C_gram.cpu().numpy()
    

    #print(J_alldata.sum(),sum(V).sum(),sum(L))

    IGS_dims = []
    trFC_dims = []
    IGS_correction_dims = []
    dims = len(L)
    if len(spurious_dim) > 0:
        dims = spurious_dim[0] - 1
    for i in range(dims):
        Jv = J_alldata.mm(V[i])
        temp = (Jv*Jv/(L[i]+eps)).sum().item() / X.shape[0]
        temp2 = (Jv*Jv*L[i]).sum().item() / X.shape[0]
        gv = total_grad.mm(V[i])
        temp3 = (gv*gv/(L[i]+eps)).sum().item()
        if i == 0:
            IGS_dims.append(temp)
            trFC_dims.append(temp2)
            IGS_correction_dims.append(temp3)
        else:
            IGS_dims.append(IGS_dims[-1] + temp)
            trFC_dims.append(trFC_dims[-1] + temp2)
            IGS_correction_dims.append(IGS_correction_dims[-1] + temp3)
    if expand:
        return IGS_dims, L, V, spurious_dim, trF, trC, trFC_dims, IGS_correction_dims, L_F_gram, L_C_gram
    return IGS_dims, L, V, spurious_dim

# calculate m-sharpness 
def calculate_m_IGS(model, train_loader, L, V, criterion, eps = 0, num_trials = 1):
    dims = len(L)
    IGS_dims_set = []
    for t in range(num_trials):
        tempIGS_dims = []
        for i in range(dims):
            tempIGS_dims.append(0)

        for k, (X, y) in enumerate(train_loader):
            model.zero_grad()
            X = X.cuda()
            y = y.cuda()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            J = vectorize([gather_grads(model)])
            for i in range(dims):
                Jv = (J*V[i]).sum()
                temp = (Jv*Jv/(L[i]+eps)).sum().item() * X.shape[0]
                tempIGS_dims[i] += temp
        IGS_dims = []
        for i in range(dims):
            if i == 0:
                IGS_dims.append(tempIGS_dims[i]/len(train_loader.dataset))
            else:
                IGS_dims.append(tempIGS_dims[i]/len(train_loader.dataset) + IGS_dims[-1])
        IGS_dims_set.append(IGS_dims)
    return IGS_dims_set