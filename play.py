from robi_env import RobiEnv
from agent import DQN
import numpy as np
from collections import deque


def play(env, agent, n_plays=100, max_t=100):
    scores_window = deque(maxlen=100)  # last 100 scores

    for i in range(n_plays):
        observation = env.reset()  # reset environment
        env.render()
        score = 0  # cumulative reward of step on episode
        for t in range(max_t):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            env.render()
            agent.step(observation, action, reward, state, done)
            observation = state
            score += reward
            if done:
                print("[Info] {}".format(info))
                break

        scores_window.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))


if __name__ == '__main__':

    environ = RobiEnv()  # create robi environment
    agent_dqn = DQN()  # create the agent
    play(environ, agent_dqn)
    environ.close()
