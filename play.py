from robi_env import RobiEnv
from agent import DQN
import numpy as np
from collections import deque
import torch

def play(env, agent, n_plays=100, max_t=100):
    scores_window = deque(maxlen=100)  # last 100 scores

    for i in range(n_plays):
        state = env.reset()  # reset environment
        env.render(delta_t=0.5)
        score = 0  # cumulative reward of step on episode
        for t in range(max_t):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            env.render(delta_t=0.5)
            score += reward
            if done:
                print("[Info] {}".format(info))
                break

        scores_window.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))


if __name__ == '__main__':

    environ = RobiEnv()  # create robi environment
    agent_dqn = DQN(action_size=5, train=False)  # create the agent
    agent_dqn.qnetwork_local.load_state_dict(torch.load('weights/dqn.pth'))
    play(environ, agent_dqn)
    environ.close()
