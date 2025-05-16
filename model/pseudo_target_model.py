import gpytorch
import torch
from torch import nn

class ExactGpModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim):
        super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=x_dim))
        self.covar_module.base_kernel.lengthscale = (x_dim) ** 0.5

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)





class PseudoTargetModel(nn.Module):
    def __init__(self, dimension, noise_level = 1e-4) -> None:
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_level
        self.likelihood.eval()

        model = ExactGpModel(None, None, self.likelihood, x_dim=dimension)

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

    def estimate_pseudo_target(self, x, step_size=1):
        
        device = x.device

        if self.model.train_inputs == None or len(self.model.train_inputs) == 0:
            return x
        
        y_pred = []
        y_grad = []

        for xj in x:
            with torch.enable_grad():
                xj.requires_grad = True
                yj = self.likelihood(self.model(xj.unsqueeze(0)))
                grad = torch.autograd.grad(yj.mean, xj, create_graph=False, allow_unused=True)[0]
                xj.requires_grad = False

            y_pred.append(yj.mean.detach().cpu())
            y_grad.append(grad.detach().cpu())

        y_pred = torch.stack(y_pred).to(device)
        y_grad = torch.stack(y_grad).to(device)
        # print(f"y_grad {y_grad}")
        pseudo_target = x - y_grad*step_size

        return pseudo_target, y_pred

    def add_model_data(self, x, y):
        if self.model.train_inputs is not None and len(self.model.train_inputs) > 0:
            x = torch.cat([self.model.train_inputs[0], x], dim=0)
            y = torch.cat([self.model.train_targets, y], dim=0)

        self.model.set_train_data(
            inputs=x, 
            targets=y,
            strict=False,
        )

    def get_model_data(self):
        return self.model.train_inputs[0], self.model.train_targets