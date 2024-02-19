
import torch
from torch.autograd import grad, Variable
from numpy import linalg as LA
import numpy as np
import math
from copy import deepcopy
from contextlib import contextmanager

@torch.no_grad()
def get_weights_only(model):
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

@torch.no_grad()
def get_vec_params(weights):
    return torch.cat([p.view(-1) for p in weights], dim=0)

@torch.no_grad()
@contextmanager
def perturbed_model(
  model,
  sigma: float,
  rng,
  magnitude_eps = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

@torch.no_grad()
def pacbayes_sigma(
  model,
  dataloader,
  accuracy: float,
  seed: int,
  magnitude_eps = None,
  search_depth: int = 15,
  montecarlo_samples: int = 3, #10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
    lower, upper = 0, 2
    sigma = 1

    BIG_NUMBER = 10348628753
    device = next(model.parameters()).device
    rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
    rng.manual_seed(BIG_NUMBER + seed)

    for _ in range(search_depth):
        sigma = (lower + upper) / 2.0
        accuracy_samples = []
        for _ in range(montecarlo_samples):
            with perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
                loss_estimate = 0
                for data, target in dataloader:
                    logits = p_model(data.to(device))
                    pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
                    batch_correct = pred.eq(target.to(device).data.view_as(pred)).type(torch.FloatTensor).cpu()
                    loss_estimate += batch_correct.sum()
                loss_estimate /= len(dataloader.dataset)
                accuracy_samples.append(loss_estimate)
        displacement = abs(np.mean(accuracy_samples) - accuracy)
        if abs(displacement - accuracy_displacement) < displacement_tolerance:
            break
        elif displacement > accuracy_displacement:
            # Too much perturbation
            upper = sigma
        else:
            # Not perturbed enough to reach target displacement
            lower = sigma
    return sigma
#--------------------------------------------------------------------------


def calculate_loss_on_data(model, loss, x, y):
    output = model(x)
    loss_value = loss(output, y)
    return output, np.sum(loss_value.data.cpu().numpy())

def run_model(model,dataloader):
    element_loss = torch.nn.CrossEntropyLoss(reduction='none')
    avg_loss = torch.nn.CrossEntropyLoss()

    inputs, labels = iter(dataloader).next()
    X = inputs.cuda()
    y = labels.cuda()
    train_size = len(X)

    train_output, train_loss_overall = calculate_loss_on_data(model, element_loss, X, y)
    #print("Train loss calculated", train_loss_overall)
    
    train_loss = avg_loss(train_output, y)

    return train_loss, train_output, y

def hessian_single_layer(layer, train_loss):
    # hessian calculation for the layer of interest
    last_layer_jacobian = grad(train_loss, layer, create_graph=True, retain_graph=True)
    hessian = []
    for n_grd in last_layer_jacobian[0]:
        for w_grd in n_grd:
            drv2 = grad(w_grd, layer, retain_graph=True)
            hessian.append(drv2[0].data.cpu().numpy().flatten())
        
    return hessian

def get_feature_layer(model):
    params = list(model.parameters())
    feature_layer_idx = -1
    for i in range(len(params)):
        if params[i] is model.classifier.weight:
            feature_layer_idx = i

    assert i is not -1
    feature_layer = list(model.parameters())[feature_layer_idx]
    return feature_layer, feature_layer_idx 

def squared_euclidean_norm(model, dataloader):
    layer, _ = get_feature_layer(model)

    weights_norm = 0.0
    for n in layer.data.cpu().numpy():
        for w in n:
            weights_norm += w**2
    #print("Squared euclidian norm is calculated", weights_norm)
    return weights_norm

def max_hessian_eigenvalue(model, dataloader,hessian=None):
    if hessian is None:
        layer, _ = get_feature_layer(model)
        train_loss, _, _ = run_model(model,dataloader)
        hessian = hessian_single_layer(layer,train_loss)

    max_eignv = LA.eigvalsh(hessian)[-1]
    return max_eignv

def hessian_trace(model, dataloader, hessian=None):
    if hessian is None:
        layer, _ = get_feature_layer(model)
        train_loss, _, _ = run_model(model,dataloader)
        hessian = hessian_single_layer(layer,train_loss)

    trace = np.trace(hessian)
    return trace

def fisher_rao_norm(model,dataloader):
    train_loss, train_output, labels = run_model(model,dataloader)
    ## calculate FisherRao norm
    # analytical formula for crossentropy loss from Appendix of the original paper
    sum_derivatives = 0
    m = torch.nn.Softmax(dim=0)
    for inp in range(len(train_output)):
        sum_derivatives += \
            (np.inner(m(train_output[inp]).data.cpu().numpy(), train_output[inp].data.cpu().numpy()) -
                train_output[inp].data.cpu().numpy()[labels[inp]]) ** 2
    fr_norm = math.sqrt(((5 + 1) ** 2) * (1.0 / len(train_output)) * sum_derivatives)
    return fr_norm

def pacbayes_flatness(model,dataloader):
    # adapted from https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py
    
    train_loss, train_output, labels = run_model(model,dataloader)
    train_acc = (train_output.argmax(dim=1)==labels).float().mean()
    train_size = len(labels)

    sigma = pacbayes_sigma(model, dataloader, train_acc, 42)
    weights = get_weights_only(model)
    w_vec = get_vec_params(weights)
    pacbayes_flat = 1.0 / sigma ** 2

    return pacbayes_flat

    # def pacbayes_bound(reference_vec):
    #     return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(train_size / sigma) + 10
    # pacbayes_orig = pacbayes_bound(w_vec).data.cpu().item()
    # print("PacBayes orig", pacbayes_orig)

def calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha, normalize = False):
    shape = feature_layer.shape

    layer_jacobian = grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    layer_jacobian_out = layer_jacobian[0]
    drv2 = Variable(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).cuda()
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = grad(n_grd[neuron_j].cuda(), feature_layer, retain_graph=True)[0].cuda()
    #print("got hessian")

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

def relative_flatness(model,dataloader):
    train_loss, train_output, labels = run_model(model,dataloader)
    
    inputs, labels = iter(dataloader).next()
    x_train = inputs.cuda()
    y_train = labels.cuda()

    feature_layer, feature_layer_idx = get_feature_layer(model)

    _, activation = model(x_train,return_feat=True)[1].data.cpu().numpy()
    activation = np.squeeze(activation)
    sigma = np.std(activation, axis=0)

    j = 0
    for p in model.parameters():
        if feature_layer_idx - 2 == j or feature_layer_idx - 1 == j:
            for i, sigma_i in enumerate(sigma):
                if sigma_i != 0.0:
                    p.data[i] = p.data[i] / sigma_i
        if feature_layer_idx == j:
            for i, sigma_i in enumerate(sigma):
                p.data[:,i] = p.data[:,i] * sigma_i
            feature_layer = p
        j += 1

    element_loss = torch.nn.CrossEntropyLoss(reduction='none')
    avg_loss = torch.nn.CrossEntropyLoss()
        
    train_output, train_loss_overall = calculate_loss_on_data(model, element_loss, x_train, y_train)
    train_loss = avg_loss(train_output, y_train)

    trace_nm, maxeigen_nm = calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, None, normalize = False)
    return trace_nm

    #print("Neuronwise tracial measure is", trace_nm)
    #print("Neuronwise max eigenvalue measure is", maxeigen_nm)


def all_sharpness_measures(model,dataloader):
    res = {}
    
    res["squared_euclidean_norm"] = squared_euclidean_norm(model,dataloader)
    print("squared_euclidean_norm:", res["squared_euclidean_norm"])
    res["max_hessian_eigenvalue"] = max_hessian_eigenvalue(model,dataloader)
    print("max_hessian_eigenvalue:", res["max_hessian_eigenvalue"])
    res["hessian_trace"] = hessian_trace(model,dataloader)
    print("hessian_trace:", res["hessian_trace"])
    res["fisher_rao_norm"] = fisher_rao_norm(model,dataloader)
    print("fisher_rao_norm:", res["fisher_rao_norm"])
    res["pacbayes_flatness"] = pacbayes_flatness(model,dataloader)
    print("pacbayes_flatness:", res["pacbayes_flatness"])
    res["relative_flatness"] = relative_flatness(model,dataloader)
    print("relative_flatness:", res["relative_flatness"])

    return res
