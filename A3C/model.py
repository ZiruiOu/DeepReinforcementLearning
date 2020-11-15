import torch
import torch.nn as nn
from torch.nn import functional as F

class AC_Network(nn.Module):

    def __init__(self, n_state,
                       n_actions,
                       hidden=128):

        super(AC_Network, self).__init__()
        self.n_state = n_state
        self.n_actions = n_actions

        self.policy = nn.Sequential(*[nn.Linear(n_state, hidden),
                                      nn.Tanh(),
                                      nn.Linear(hidden, n_actions),
                                      nn.Softmax(dim=1)])

        self.value = nn.Sequential(*[nn.Linear(n_state, hidden),
                                     nn.Tanh(),
                                     nn.Linear(hidden, 1)])

        self.distributions = torch.distributions.Categorical

    def forward(self, states):

        logits = self.policy(states)
        values = self.value(states)

        return logits, values

    def choose_action(self, state):

        self.eval()
        prob, _ = self.forward(state)

        m = self.distributions(prob.data)
        return m.sample().numpy()[0]

    def loss_fn(self, states, actions, v_t):

        self.train()
        probs, values = self.forward(states)

        # critic loss
        advantage = v_t - values
        critic_loss = advantage.pow(2)

        # actor loss
        m = self.distributions(probs)
        actor_loss = -1 * m.log_prob(actions) * advantage.detach()

        total_loss = (actor_loss + critic_loss).mean()
        return total_loss







