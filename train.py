from robi_env import RobiEnv
from agent import DQN
import numpy as np
from collections import deque
import torch


def train(env, agent, n_episodes=1000, max_t=100):
    scores_window = deque(maxlen=100)  # last 100 scores

    for i in range(n_episodes):
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

        if i % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break


if __name__ == '__main__':

    environ = RobiEnv()  # create robi environment
    agent_dqn = DQN()  # create the agent
    train(environ, agent_dqn)
    environ.close()

