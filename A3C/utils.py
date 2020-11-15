import torch
import numpy as np


def network_update(gnet,
                   optimizer,
                   lnet,
                   final,
                   done,
                   states,
                   actions,
                   rewards,
                   gamma=0.99):
    if done:
        run_q = 0
    else:
        final = torch.FloatTensor(final).unsqueeze(0)
        _, run_q = lnet(final)
        run_q = run_q.squeeze(0).data.numpy()[0]

    q_t = []
    N = len(states)

    for reward in rewards[::-1]:
        run_q = run_q * gamma + reward
        q_t.append(run_q)

    q_t.reverse()

    states = torch.FloatTensor(np.vstack(states))
    actions = torch.LongTensor(actions)
    q_t = torch.FloatTensor(q_t)[:, None]

    loss = lnet.loss_fn(states, actions, q_t)
    optimizer.zero_grad()
    loss.backward()

    for l_param, g_param in zip(lnet.parameters(), gnet.parameters()):
        g_param._grad = l_param.grad()

    optimizer.step()
    lnet.load_state_dict(gnet.state_dict())


def record(global_epoch, global_ep_r, res_queue, reward, name):
    with global_epoch.get_lock():
        global_epoch.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0:
            global_ep_r.value = reward
        else:
            global_ep_r.value = global_ep_r * 0.99 + reward * 0.01

    res_queue.push(global_ep_r.value)
    print(name,
          " Epoch : {}".format(global_epoch.value),
          " reward : {}".format(global_ep_r.value))

