import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim, distributions
from torch.nn import functional as F

from agent import Agent, EnvArgs
from layers.bayesian_layer import VBLinear

class BNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(BNN, self).__init__()
        self.Linear1 = nn.Linear(n_observations, 128)
        self.Linear2 = nn.Linear(128, 128)
        self.VBLinear1 = VBLinear(128, n_actions)
        self.VBLinear2 = VBLinear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def KL_loss(self):
        KLsum = self.VBLinear1.KL() + self.VBLinear2.KL()
        return KLsum

    def log_likelihood(self, x, y):
        """Compute the log likelihood of y given x"""
        
        # Compute mean and std predictions
        mu, logvar = self.forward(x)
        
        # Return log likelihood of true data given predictions
        return distributions.Normal(mu,logvar.exp()).log_prob(y)
    
    def ELBOloss(self,y_output,y_target):
        mu = y_output[:,0]
        logvar = y_output[:,1]
        N = mu.size(dim=-1)
        ELBO = (logvar + (y_target-mu).pow(2)/logvar.exp()).sum() + 2/N*self.KL_loss() 
        return ELBO

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        mu = self.VBLinear1(x)
        logvar = F.relu(self.VBLinear2(x))
        return torch.stack((mu, logvar), -1)



class BayesianDeepExpectedSarsa(Agent):
    def __init__(self, env: gym.Env, args: EnvArgs) -> None:
        super().__init__(env, args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.loss_fn = self.model.ELBOloss

    def _build_model(self):
        return BNN(self.num_state,self.num_actions)

    def get_target_q_values(self):
        pass

    def select_action(self, state, epsilon: float = 0) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        q_values = (
            self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            .detach()
            .numpy()[:,0]
        )
        return np.argmax(q_values)

    def training_step(self, epsilon=None) -> None:
        states, actions, rewards, next_states, dones, truncateds = (
            self.sample_experiences()
        )
        next_q_values = (
            self.model(torch.tensor(next_states, dtype=torch.float32)).detach().numpy()[:,0]
        )

        target_q_values = []
        for i in range(len(next_states)):
            next_q = next_q_values[i]
            greedy_actions = np.sum(next_q == np.max(next_q))
            non_greedy_action_probability = epsilon / self.num_actions
            greedy_action_probability = (
                1 - epsilon
            ) / greedy_actions + non_greedy_action_probability
            expected_q = np.sum(next_q * greedy_action_probability)
            target_q = (
                rewards[i]
                + (1.0 - (dones[i] or truncateds[i]))
                * self.args.discount_factor
                * expected_q
            )
            target_q_values.append(target_q)
        target_q_values = np.array(target_q_values).reshape(-1, 1)

        mask = torch.tensor(np.eye(self.num_actions)[actions], dtype=torch.float32)
        all_q_values = self.model(torch.tensor(states, dtype=torch.float32))
        mu_values = torch.sum(all_q_values[:,0] * mask, dim=1, keepdim=True)
        logvar_values = torch.sum(all_q_values[:,1] * mask, dim=1, keepdim=True)
        q_values = torch.cat((mu_values, logvar_values), -1)
        #print(all_q_values)
        #print(mask)
        #print(q_values)
        #exit()
        loss = self.loss_fn(q_values.float(),torch.tensor(target_q_values).float()) #(torch.tensor(states, dtype=torch.float32),torch.tensor(target_q_values).float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()
