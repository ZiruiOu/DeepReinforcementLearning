from abc import ABC

import gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from utils import record, network_update

UPDATE_GLOBAL_ITER = 5


class AC_Network(nn.Module, ABC):

    def __init__(self,
                 n_state,
                 n_action,
                 hidden=128):

        super(AC_Network, self).__init__()
        self.n_state = n_state
        self.n_actions = n_action

        self.policy = nn.Sequential(*[nn.Linear(n_state, hidden),
                                      nn.Tanh(),
                                      nn.Linear(hidden, n_action),
                                      nn.Softmax(dim=1)])

        self.value = nn.Sequential(*[nn.Linear(n_state, hidden),
                                     nn.Tanh(),
                                     nn.Linear(hidden, 1)])

        self.distributions = torch.distributions.Categorical

    def forward(self, states):
        prob = self.policy(states)
        values = self.value(states)

        return prob, values

    def choose_action(self, state):
        """
        :param state:  should be a 2d tensor
        :return: the action chosen by sampling
        """
        self.eval()
        prob, _ = self.forward(state)

        m = self.distributions(prob.data)
        return m.sample().numpy()[0]

    def loss_fn(self, states, actions, q_t):
        """
        :param states: 2d tensors by stacking the states in a whole batch
        :param actions: actions chosen in a batch
        :param q_t: the Q value calculated by the samples
        :return: the actor_loss + critic_loss
        """

        self.train()
        probs, values = self.forward(states)

        # critic loss
        advantage = q_t - values
        critic_loss = advantage.pow(2)

        # actor loss
        m = self.distributions(probs)
        actor_loss = -1 * m.log_prob(actions) * advantage.detach()

        total_loss = (actor_loss + critic_loss).mean()
        return total_loss


class Worker(mp.Process):

    def __init__(self,
                 n_state,
                 n_action,
                 global_net,
                 optimizer,
                 global_ep,
                 global_ep_r,
                 res_queue,
                 name,
                 max_step=4000):

        super(Worker, self).__init__()
        self.name = "w%02d" % name

        self.g_epoch, self.g_epr, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = global_net, optimizer
        self.local = AC_Network(n_state, n_action)
        self.max_step = max_step

        self.env = gym.make('CartPole-v0')

    def run(self):
        total_step = 1

        while self.g_epoch.value < self.max_step:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            epoch_reward = 0

            while True:
                action = self.local.choose_action(torch.FloatTensor(state).unsqueeze(0))
                next_state, reward, info, done = self.env.step(action)

                epoch_reward += reward
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if done or total_step % UPDATE_GLOBAL_ITER == 0:
                    # TODO : update the center network and update the local network
                    network_update(self.gnet,
                                   self.opt,
                                   self.local,
                                   next_state,
                                   done,
                                   buffer_s,
                                   buffer_a,
                                   buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(self.g_epoch,
                               self.g_epr,
                               self.res_queue,
                               epoch_reward,
                               self.name)
                        break

                state = next_state
                total_step += 1

        self.res_queue.put(None)
