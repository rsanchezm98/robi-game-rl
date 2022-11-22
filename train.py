from robi_env import RobiEnv
from agent import DQN
import numpy as np
from collections import deque
import torch


def train(env, agent, n_episodes=1000000, max_t=50):
    scores_window = deque(maxlen=50)  # last 100 scores
    for i in range(n_episodes):
        state = env.reset()  # reset environment
        # env.render()
        score = 0  # cumulative reward of step on episode
        for t in range(max_t):
            # action = env.action_space.sample()
            action = agent.act(state)
            prev_state = state
            state, reward, done, info = env.step(action)
            agent.step(prev_state, action, reward, state, done)
            # env.render(mode="gui", delta_t=0.01)
            score += reward
            if done:
                # print("[Info] {}".format(info))
                break

        scores_window.append(np.mean(score))

        if i % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

        if np.mean(scores_window) >= 200:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'weights/dqn.pth')


if __name__ == '__main__':

    environ = RobiEnv()  # create robi environment
    agent_dqn = DQN(action_size=5, train=True)  # create the agent
    agent_dqn.qnetwork_local.load_state_dict(torch.load('weights/dqn.pth'))
    agent_dqn.qnetwork_target.load_state_dict(torch.load('weights/dqn.pth'))
    train(environ, agent_dqn)
    environ.close()

