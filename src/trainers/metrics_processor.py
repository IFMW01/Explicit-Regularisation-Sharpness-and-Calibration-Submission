import math
from contextlib import contextmanager
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from numpy import linalg as LA
from pyhessian import hessian
from torch.autograd import Variable, grad
from tqdm import tqdm

from models.temperature_scaling import _ECELoss
from trainers.igs.igs import calculate_IGS_largemodel


class MetricsProcessor:

    def __init__(self, config, model, train_dataloader, test_dataloader, device, seed, model_name, num_classes) -> None:
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.seed = seed
        self.model_name = model_name

        self.cache_max_hessian_eigenvalue = None
        self.cache_hessian_trace = None
        self.num_classes = num_classes

    def compute_metrics(self):
        """
        Compute metrics
        """

        results_dict = {}

        self.model.eval()
        for metric in self.config.metrics:
            compute_func = getattr(self, metric)
            print(f"Running metrics {str(metric)}...")
            metric_name = "eval_metrics_" + self.model_name + "/" + metric
            results_dict[metric_name] = compute_func()
            self.model.zero_grad(set_to_none=True)

        return results_dict

    @torch.no_grad()
    def get_weights_only(self):
        blacklist = {"bias", "bn"}
        return [
            p
            for name, p in self.model.named_parameters()
            if all(x not in name for x in blacklist)
        ]

    @torch.no_grad()
    def get_vec_params(self, weights):
        return torch.cat([p.view(-1) for p in weights], dim=0)

    

    @torch.no_grad()
    @contextmanager
    def perturbed_model(self, sigma: float, rng, magnitude_eps=None):
        device = next(self.model.parameters()).device
        if magnitude_eps is not None:
            noise = [
                torch.normal(
                    0, sigma**2 * torch.abs(p) ** 2 + magnitude_eps**2, generator=rng
                )
                for p in self.model.parameters()
            ]
        else:
            noise = [
                torch.normal(0, sigma**2, p.shape,
                             generator=rng).to(self.device)
                for p in self.model.parameters()
            ]
        model = deepcopy(self.model)
        try:
            [p.add_(n) for p, n in zip(self.model.parameters(), noise)]
            yield model
        finally:
            [p.sub_(n) for p, n in zip(self.model.parameters(), noise)]
            del model

    @torch.no_grad()
    def pacbayes_sigma(
        self,
        accuracy: float,
        magnitude_eps=None,
        search_depth: int = 15,
        montecarlo_samples: int = 3,  # 10,
        accuracy_displacement: float = 0.1,
        displacement_tolerance: float = 1e-2,
    ) -> float:
        lower, upper = 0, 2
        sigma = 1

        BIG_NUMBER = 10348628753
        device = next(self.model.parameters()).device
        rng = (
            torch.Generator(device=device)
            if magnitude_eps is not None
            else torch.Generator()
        )
        rng.manual_seed(BIG_NUMBER + self.seed)

        for _ in range(search_depth):
            sigma = (lower + upper) / 2.0
            accuracy_samples = []
            for _ in range(montecarlo_samples):
                with self.perturbed_model(sigma, rng, magnitude_eps) as p_model:
                    loss_estimate = 0
                    for data, target in self.train_dataloader:
                        logits = p_model(data.to(self.device))
                        pred = logits.data.max(1, keepdim=True)[
                            1
                        ]  # get the index of the max logits
                        batch_correct = (
                            pred.eq(target.to(self.device).data.view_as(pred))
                            .type(torch.FloatTensor)
                            .cpu()
                        )
                        loss_estimate += batch_correct.sum()
                    loss_estimate /= len(self.train_dataloader.dataset)
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

    # --------------------------------------------------------------------------

    def run_model(self):
        avg_loss = torch.nn.CrossEntropyLoss()

        all_outputs = []
        all_y = []
        all_feat = []

        for inputs, labels in tqdm(self.train_dataloader):
            X = inputs.to(self.device)
            y = labels.to(self.device)

            output, feat = self.model(X, return_feat=True)

            all_outputs.append(output)
            all_feat.append(feat)
            all_y.append(y)

        all_outputs = torch.cat(all_outputs)
        all_feat = torch.cat(all_feat)
        all_y = torch.cat(all_y)

        train_loss = avg_loss(all_outputs, all_y)

        return train_loss, all_outputs, all_feat, all_y
    
    @torch.no_grad()
    def sam_sharpness(self):
        
        num_params = sum(torch.numel(x) for x in self.model.parameters())
        def get_loss(model):
            loss_fn = torch.nn.CrossEntropyLoss()
            losses = []
            with torch.no_grad():
                for X,y in tqdm(self.train_dataloader):
                    losses.append(loss_fn(model(X.cuda()),y.cuda()).item())

            return np.mean(losses)
        
        original_loss = get_loss(self.model)
        new_losses = []
        for _ in range(10):
            new_state_dict = deepcopy(self.model.state_dict())

            ro = 0.05

            for param_name in new_state_dict:
                if new_state_dict[param_name].dtype != torch.int64:
                    new_state_dict[param_name] += torch.randn_like(new_state_dict[param_name]) / (num_params**0.5) *ro

            new_model = deepcopy(self.model)
            new_model.load_state_dict(new_state_dict)

            loss = get_loss(new_model)

            new_losses.append((loss-original_loss)/ro)

        return max(new_losses)

    def hessian_single_layer(self, layer, train_loss):
        # hessian calculation for the layer of interest
        last_layer_jacobian = grad(
            train_loss, layer, create_graph=True, retain_graph=True
        )
        hessian = []
        for n_grd in last_layer_jacobian[0]:
            for w_grd in n_grd:
                drv2 = grad(w_grd, layer, retain_graph=True)
                hessian.append(drv2[0].data.cpu().numpy().flatten())

        return hessian

    def get_feature_layer(self):
        params = list(self.model.parameters())
        feature_layer_idx = -1
        for i in range(len(params)):
            if params[i] is self.model.model.classifier.weight:
                feature_layer_idx = i

        assert i is not -1
        feature_layer = list(self.model.parameters())[feature_layer_idx]
        return feature_layer, feature_layer_idx

    def squared_euclidean_norm(self):
        layer, _ = self.get_feature_layer()

        weights_norm = 0.0
        for n in layer.data.cpu().numpy():
            for w in n:
                weights_norm += w**2
        # print("Squared euclidian norm is calculated", weights_norm)
        return weights_norm

    def max_hessian_eigenvalue_slow(self, hessian=None):
        if hessian is None:
            layer, _ = self.get_feature_layer()
            train_loss, _, _, _ = self.run_model()
            hessian = self.hessian_single_layer(layer, train_loss)

        max_eignv = LA.eigvalsh(hessian)[-1]
        return max_eignv

    def hessian_trace_slow(self, hessian=None):
        if hessian is None:
            layer, _ = self.get_feature_layer()
            train_loss, _, _, _ = self.run_model()
            hessian = self.hessian_single_layer(layer, train_loss)

        trace = np.trace(hessian)
        return trace

    def calc_hessian_metrics(self):
        if self.cache_max_hessian_eigenvalue is not None:
            return

        hessian_comp = hessian(self.model,
                               torch.nn.CrossEntropyLoss(),
                               dataloader=self.train_dataloader,
                               cuda=True)

        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=20, tol=0.01)
        trace = hessian_comp.trace(maxIter=20, tol=0.01)

        self.cache_max_hessian_eigenvalue = top_eigenvalues[0]
        self.cache_hessian_trace = np.mean(trace)

    def max_hessian_eigenvalue(self):
        self.calc_hessian_metrics()
        return self.cache_max_hessian_eigenvalue

    def hessian_trace(self):
        self.calc_hessian_metrics()
        return self.cache_hessian_trace

    def fisher_rao_norm(self):
        with torch.no_grad():
            train_loss, train_output, activation, labels = self.run_model()
        # calculate FisherRao norm
        # analytical formula for crossentropy loss from Appendix of the original paper
        sum_derivatives = 0
        m = torch.nn.Softmax(dim=0)
        for inp in range(len(train_output)):
            sum_derivatives += (
                np.inner(
                    m(train_output[inp]).data.cpu().numpy(),
                    train_output[inp].data.cpu().numpy(),
                )
                - train_output[inp].data.cpu().numpy()[labels[inp]]
            ) ** 2
        fr_norm = math.sqrt(
            ((5 + 1) ** 2) * (1.0 / len(train_output)) * sum_derivatives
        )
        return fr_norm

    def pacbayes_flatness(self):
        # adapted from https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py

        train_loss, train_output, activation, labels = self.run_model()
        train_acc = (train_output.argmax(dim=1) == labels).float().mean()
        train_size = len(labels)

        sigma = self.pacbayes_sigma(train_acc)
        weights = self.get_weights_only()
        w_vec = self.get_vec_params(weights)
        pacbayes_flat = 1.0 / sigma**2

        return pacbayes_flat

        # def pacbayes_bound(reference_vec):
        #     return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(train_size / sigma) + 10
        # pacbayes_orig = pacbayes_bound(w_vec).data.cpu().item()
        # print("PacBayes orig", pacbayes_orig)

    def calculateNeuronwiseHessians_fc_layer(
        self, feature_layer, train_loss, alpha, normalize=False
    ):
        shape = feature_layer.shape

        layer_jacobian = grad(
            train_loss, feature_layer, create_graph=True, retain_graph=True
        )
        layer_jacobian_out = layer_jacobian[0]
        drv2 = Variable(
            torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True
        ).to(self.device)
        for ind, n_grd in enumerate(layer_jacobian[0].T):
            for neuron_j in range(shape[0]):
                drv2[ind][neuron_j] = grad(
                    n_grd[neuron_j].to(self.device), feature_layer, retain_graph=True
                )[0].to(self.device)
        # print("got hessian")

        trace_neuron_measure = 0.0
        maxeigen_neuron_measure = 0.0
        for neuron_i in range(shape[0]):
            neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
            for neuron_j in range(shape[0]):
                neuron_j_weights = feature_layer[neuron_j, :].data.cpu(
                ).numpy()
                hessian = drv2[:, neuron_j, neuron_i, :]
                trace = np.trace(hessian.data.cpu().numpy())
                if normalize:
                    trace /= 1.0 * hessian.shape[0]
                trace_neuron_measure += neuron_i_weights.dot(
                    neuron_j_weights) * trace
                if neuron_j == neuron_i:
                    eigenvalues = LA.eigvalsh(hessian.data.cpu().numpy())
                    maxeigen_neuron_measure += (
                        neuron_i_weights.dot(
                            neuron_j_weights) * eigenvalues[-1]
                    )
                    # adding regularization term
                    if alpha:
                        trace_neuron_measure += (
                            neuron_i_weights.dot(
                                neuron_j_weights) * 2.0 * alpha
                        )
                        maxeigen_neuron_measure += (
                            neuron_i_weights.dot(
                                neuron_j_weights) * 2.0 * alpha
                        )

        return trace_neuron_measure, maxeigen_neuron_measure

    def relative_flatness(self):
        with torch.no_grad():
            train_loss, train_output, activation, labels = self.run_model()

        feature_layer, feature_layer_idx = self.get_feature_layer()

        activation = activation.detach().cpu().numpy()

        activation = np.squeeze(activation)
        sigma = np.std(activation, axis=0)

        j = 0
        for p in self.model.parameters():
            if feature_layer_idx - 2 == j or feature_layer_idx - 1 == j:
                for i, sigma_i in enumerate(sigma):
                    if sigma_i != 0.0:
                        p.data[i] = p.data[i] / sigma_i
            if feature_layer_idx == j:
                for i, sigma_i in enumerate(sigma):
                    p.data[:, i] = p.data[:, i] * sigma_i
                feature_layer = p
            j += 1

        # train_loss, _, _, _ = self.run_model()

        trace_nm = 0

        avg_loss = torch.nn.CrossEntropyLoss()

        for inputs, labels in tqdm(self.train_dataloader):
            X = inputs.to(self.device)
            y = labels.to(self.device)

            train_loss = avg_loss(self.model(X), y)

            curr_trace_nm, curr_maxeigen_nm = self.calculateNeuronwiseHessians_fc_layer(
                feature_layer, train_loss, None, normalize=False
            )

            trace_nm += curr_trace_nm

            self.model.zero_grad(set_to_none=True)

        return trace_nm

        # print("Neuronwise tracial measure is", trace_nm)
        # print("Neuronwise max eigenvalue measure is", maxeigen_nm)
    
    def IGS(self, output_all=False):
        criterion = torch.nn.CrossEntropyLoss()
        criterion_alldata = torch.nn.CrossEntropyLoss(reduction = 'none')
        
        IGS = []
        
        num_fails = 0
        for X,y in tqdm(self.train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            #experiment_fast(model, False, X,y,criterion, criterion_alldata)
            IGS_dims, L, V, spurious_dim = calculate_IGS_largemodel(self.model, X, X,y,criterion, 1e-4,3,exact_fisher=True)
            if len(IGS_dims)==3:
                IGS.append(IGS_dims[-1])
            else:
                num_fails += 1
                #print("Warning: IGS calculation incomplete, len =",len(IGS_dims))
                 
        if num_fails>len(IGS):
            print("Warning: failed IGS calculation")
        
        if output_all:
            return IGS
        return np.array(IGS).mean()

    @torch.no_grad()
    def ece(self):
        ece_criterion = _ECELoss().to(self.device)

        logits_list = []
        labels_list = []
        for input, label in self.test_dataloader:
            input = input.to(self.device)
            logits = self.model(input)
            logits_list.append(logits)
            labels_list.append(label)

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        ece = ece_criterion(logits, labels).item()

        return ece

    @torch.no_grad()
    def acc(self):
        correct = 0
        total = 0
        for input, label in self.test_dataloader:
            input = input.to(self.device)
            label = label.to(self.device)
            pred = self.model(input)
            comparison_with_gold = torch.argmax(pred, dim=-1) == label
            correct += comparison_with_gold.sum().item()
            total += len(label)

        acc = correct / total

        return acc

    @torch.no_grad()
    def reliability_plot(self):

        logits_list = []
        labels_list = []
        ece_criterion = _ECELoss().to(self.device)

        preds = []
        labels_oneh = []

        sm = nn.Softmax(dim=1)

        for inputs, labels in tqdm(self.test_dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            pred = self.model(inputs)

            logits_list.append(pred)
            labels_list.append(labels)
                        
            pred = sm(pred)

            _, predicted_cl = torch.max(pred.data, 1)
            pred = pred.cpu().detach().numpy()

            label_oneh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
            label_oneh = label_oneh.cpu().detach().numpy()

            preds.extend(pred)
            labels_oneh.extend(label_oneh)

        preds = np.array(preds).flatten()
        labels_oneh = np.array(labels_oneh).flatten()

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        ECE = ece_criterion(logits, labels).item() * 100

        bins, _, bin_accs, _, _ = self.calc_bins(preds, labels_oneh, num_bins=11)

        bins = bins[1:]
        bin_accs = bin_accs[1:]

        with open("/home/is473/rds/hpc-work/R252_Group_Project/data/reliability_plot_template.txt", "r") as f:
            template = f.read()

        data = [(bin-0.1, acc) for bin, acc in zip(bins, bin_accs)]
        data = [str(x) for x in data]
        data = " ".join(data)

        tikz_plot = template.replace("FILL-IN-ECE-HERE", str(round(ECE, 1)))
        tikz_plot = tikz_plot.replace("FILL-IN-MODEL-DATA-HERE", data)

        with open(self.config.models_dir / f"{self.model_name}_reliability_plot.tex", "w") as f:
            f.write(tikz_plot)
    
    @staticmethod
    def calc_bins(preds, labels_oneh, num_bins=11):

        bins = np.linspace(0.0, 1, num_bins)
        binned = np.digitize(preds, bins)

        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)

        for bin in range(num_bins):
            bin_sizes[bin] = len(preds[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

        print(bins)
        print(bin_accs)

        assert bin_accs[0] == 0.0


        return bins, binned, bin_accs, bin_confs, bin_sizes


# def all_sharpness_measures(dataloader):
#     res = {}

#     res["squared_euclidean_norm"] = squared_euclidean_norm(model, dataloader)
#     res["max_hessian_eigenvalue"] = max_hessian_eigenvalue(model, dataloader)
#     res["hessian_trace"] = hessian_trace(model, dataloader)
#     res["fisher_rao_norm"] = fisher_rao_norm(model, dataloader)
#     res["pacbayes_flatness"] = pacbayes_flatness(model, dataloader)
#     res["relative_flatness"] = relative_flatness(model, dataloader)

#     return res


# if __name__ == "__main__":

#     # Just baseline to test

#     transformations = transforms.Compose([transforms.ToTensor()])

#     print("Loading data")
#     testset = datasets.CIFAR10(
#         root="../data", train=False, download=True, transform=transformations
#     )
#     testloader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=256,
#         shuffle=False,
#         num_workers=2,
#         # worker_init_fn=self.seed_worker,
#         generator=torch.Generator(),
#     )

#     print("Creating model")
#     model_created = CM.CreateModels(
#         baseline=True,
#         adversarial=False,
#         model_type="VGG19",
#         aug=False,
#         dropout=0.0,
#         dataset="CIFAR10",
#         save_name=None,
#         save_directory=None,
#     )
#     model_created.set_seeds()
#     device = model_created.device()
#     model = model_created.VGG()

#     state_dict = torch.load("../models/baseline.pth")
#     model.load_state_dict(state_dict)
#     model = model.to(self.device)

#     print("Running measures")
#     res = all_sharpness_measures(model, testloader)

#     pprint(res)