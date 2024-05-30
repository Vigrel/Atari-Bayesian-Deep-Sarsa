import math
import torch
from torch import nn

class VBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sig2 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sig2 = nn.Parameter(torch.Tensor(out_features))

        self.weight_mu_prior = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=False)
        self.weight_log_sig2_prior = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=False)
        self.bias_mu_prior = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bias_log_sig2_prior = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.normal_(self.weight_mu, mean=0, std=1/math.sqrt(self.weight_mu.shape[1]))
        nn.init.constant_(self.weight_log_sig2, -10)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_sig2, -10)

    '''
    def KL(self, loguniform=False):
        logsig2_w = self.logsig2_w.clamp(-11,11)
        kl = (0.5*(self.prior_prec * (self.weight_mu.pow(2)+logsig2_w.exp()) -logsig2_w-1-torch.log(self.prior_prec*torch.ones(self.n_out, self.n_in))).sum())
        
        return kl
    '''
    
    def KL(self):
        kl_weight = 0.5 * (self.weight_log_sig2_prior - self.weight_log_sig2 + (self.weight_log_sig2.exp() + (self.weight_mu_prior-self.weight_mu)**2) / self.weight_log_sig2_prior.exp() - 1.0)
        kl = kl_weight.sum()
        n = len(self.weight_mu.view(-1))

        kl_bias = 0.5 * (self.bias_log_sig2_prior - self.bias_log_sig2 + (self.bias_log_sig2.exp() + (self.bias_mu_prior-self.bias_mu)**2) / (self.bias_log_sig2_prior.exp()) - 1.0)
        kl += kl_bias.sum()
        n += len(self.bias_mu.view(-1))
        return kl
    
    def update_prior(self, newprior):
        self.weight_mu_prior.data = newprior.weight_mu.data.clone()
        self.weight_mu_prior.data.requires_grad = False
        self.weight_log_sig2_prior.data = newprior.weight_log_sig2.data.clone()
        self.weight_log_sig2_prior.data.requires_grad = False

        self.bias_mu_prior.data = newprior.bias_mu.data.clone()
        self.bias_mu_prior.data.requires_grad = False
        self.bias_log_sig2_prior.data = newprior.bias_log_sig2.data.clone()
        self.bias_log_sig2_prior.data.requires_grad = False  


    def forward(self, input):
        output_mu = nn.functional.linear(input, self.weight_mu, self.bias_mu)
        #var_out = nn.functional.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp())
        var_out = nn.functional.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp()).log()
        #return output_mu + var_out.sqrt() * torch.randn_like(output_mu)
        return torch.stack((output_mu, var_out), -1)
