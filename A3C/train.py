import gym
from model import AC_Network, Worker
from sharedAdam import SharedAdam
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
n_state, n_action = env.observation_space.shape[0], env.action_space.n

if __name__ == "__main__":
    gnet = AC_Network(n_state, n_action)
    gnet.share_memory()

    optimizer = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.99, 0.999))

    g_epoch, g_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0), mp.Queue()

    workers = [Worker(n_state,
                      n_action,
                      gnet,
                      optimizer,
                      g_epoch,
                      g_ep_r,
                      res_queue,
                      i) for i in range(mp.cpu_count())]

    [worker.start() for worker in workers]

    result = []
    while True:
        reward = res_queue.get()
        if reward is not None:
            result.append(reward)
        else:
            break

    [w.join() for w in workers]

    plt.plot(result)
    plt.ylabel("Moving average ep reward")
    plt.xlabel("step")
    plt.show()

