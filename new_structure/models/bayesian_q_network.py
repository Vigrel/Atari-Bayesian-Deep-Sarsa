import torch
from torch import nn, distributions
from torch.nn import functional as F
from layers.bayesian_layer import VBLinear


class BayesianQNetwork(nn.Module):

    def __init__(self, action_space_n):
        super(BayesianQNetwork, self).__init__()
        self.Conv2d_1 = nn.Conv2d(4, 32, 8, stride=4)
        self.Conv2d_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.Conv2d_3 = nn.Conv2d(64, 64, 3, stride=1)
        self.Flatten_1 = nn.Flatten()
        self.Linear_1 = nn.Linear(3136, 512)
        self.VBLinear_1 = VBLinear(512, action_space_n)
        self.VBLinear_2 = VBLinear(512, action_space_n)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def update_prior_bnn(model: nn.Module, newprior: nn.Module):
        
        """ Function to update priors of bayesian neural network """
        newmodel = list(model.children())
        curmodel = list(newprior.children())

        # iterate over the nn.Module layers
        for i in range(len(newmodel)):
            if newmodel[i].__class__.__name__.startswith('VB'):
                newmodel[i].update_prior(curmodel[i])

    def KL_loss(self):
        KLsum = self.VBLinear_1.KL() + self.VBLinear_2.KL()
        return KLsum

  
    def ELBOloss(self,y_output,y_target):
        mu = y_output[:,0]
        logvar = y_output[:,1]
        #n = mu.size(dim=-1)
        elbo = (logvar + (y_target-mu).pow(2)/logvar.exp()).sum() + self.KL_loss() 
        return elbo

    def forward(self, x):
        x = F.relu(self.Conv2d_1(x))
        x = F.relu(self.Conv2d_2(x))
        x = F.relu(self.Conv2d_3(x))
        x = self.Flatten_1(x)
        x = F.relu(self.Linear_1(x))
        mu = self.VBLinear_1(x)
        logvar = self.VBLinear_2(x)
        return torch.stack((mu, logvar), -1)