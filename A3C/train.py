import gym
import torch
from model import AC_Network, Worker
from sharedAdam import SharedAdam
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
n_state, n_action = env.observation_space.shape[0], env.action_space.n

def test(model, test_epoch=20):
    for i in range(test_epoch):
        state = env.reset()
        total_step = 0
        while True:
            action = model.choose_action(torch.FloatTensor(state).unsqueeze(0))
            next_state, r_, done, _ = env.step(action)

            total_step += 1
            if done:
                print("In test epoch {} : reward {}".format(i, total_step))
                break

            state = next_state

    env.close()


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

    test(gnet)

