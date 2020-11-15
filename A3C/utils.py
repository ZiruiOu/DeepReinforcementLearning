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
                   gamma=0.95):
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

    for local_param, g_param in zip(lnet.parameters(), gnet.parameters()):
        g_param._grad = local_param.grad()

    optimizer.step()
    lnet.load_state_dict(gnet.state_dict())
