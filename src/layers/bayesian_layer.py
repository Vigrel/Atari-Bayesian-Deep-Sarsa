import math
import torch
from torch import nn

class VBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.prior_prec = 10
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameter()
    
    def reset_parameter(self):
        stdv = 1.0 /math.sqrt(self.weight_mu.shape[1])
        self.weight_mu.data.normal_(0,stdv)
        self.logsig2_w.data.zero_().normal_(-9,0.001)
        self.bias.data.zero_()

    def KL(self, loguniform=False):
        logsig2_w = self.logsig2_w.clamp(-11,11)
        kl = (0.5*(self.prior_prec * (self.weight_mu.pow(2)+logsig2_w.exp()) -logsig2_w-1-torch.log(self.prior_prec*torch.ones(self.n_out, self.n_in))).sum())
        #prior = torch.normal(torch.zeros(self.n_out, self.n_in), torch.ones(self.n_out, self.n_in))
        #kl_loss = nn.KLDivLoss(reduction="sum")
        #kl = kl_loss(self.weight_mu, prior)
        return kl
    
    def forward(self, input):
        mu_out = nn.functional.linear(input, self.weight_mu, self.bias)
        s2_w = self.logsig2_w.clamp(-11,11).exp()
        var_out = nn.functional.linear(input.pow(2), s2_w) + 1e-8
        return mu_out + var_out.sqrt() * torch.randn_like(mu_out)
