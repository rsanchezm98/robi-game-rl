import gym
import numpy as np
import pygame
import time

WINDOW_SIZE = (550, 550)


class RobiEnv(gym.Env):
    def __init__(self):
        # Action space
        self.action_space = gym.spaces.Discrete(5)  # stop up down left right [0, 1, 2, 3, 4]

        # Observation space
        self._height = 8
        self._width = 8
        self._possible_positions_robi = self._width * self._height
        self._possible_positions_goal = self._height
        self._possible_positions_rocks = (self._width - 2) * self._height
        self._num_states = self._possible_positions_robi*self._possible_positions_goal*self._possible_positions_rocks
        self.observation_space = gym.spaces.Discrete(self._num_states)

        # create the map
        self.map = None
        self._robi_in_lava = False

        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self._width,
            WINDOW_SIZE[1] / self._height,
        )

    def step(self, action):
        done = False
        reward = -1  # each time here is penalised
        info = ''

        # move robot
        y, x = self._robi_pose
        delta_y, delta_x = 0, 0
        if action == 1:
            delta_y = -1
        if action == 2:
            delta_y = 1
        if action == 3:
            delta_x = -1
        if action == 4:
            delta_x = 1
        next_x = x + delta_x
        next_y = y + delta_y
        if next_y >= self._height or next_y < 0 or next_x >= self._width or next_x < 0:
            reward += -100
            info = 'robi fell to lava'
            self._robi_in_lava = True
            done = True

        if (next_y, next_x) == (y, x):
            reward += -1  # double penalisation if stays in the same place

        self._robi_pose = (next_y, next_x)

        # move rocks
        for i in range(len(self._rocks_pose)):
            y, x = self._rocks_pose[i]

            if self._robi_pose == self._rocks_pose[i]:
                info = 'robi collision with a rock'
                reward += -100
                done = True
                break

            if y == self._height - 1:
                self._rocks_pose[i] = (0, x)
            else:
                self._rocks_pose[i] = (y+1, x)

            if self._robi_pose == self._rocks_pose[i]:
                info = 'robi collision with a rock'
                reward += -100
                done = True
                break

        reward += self._robi_pose[1] * 0.5  # rewarding him with distance to goal

        if self._robi_pose == self._goal_pose:
            info = 'robi got the goal!'
            reward += 400
            done = True

        return self.to_map(), reward, done, info

    def render(self, mode="gui", delta_t=0.1):
        if mode == "gui":
            self._render_gui(delta_t=delta_t)
        else:
            print(self.to_map())

    def _render_gui(self, delta_t=0.1):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("ROBI RL GAME")
            self.window = pygame.display.set_mode(WINDOW_SIZE)

        map_ = self.to_map()

        for y in range(self._height):
            for x in range(self._width):

                polygon = [(x*self.cell_size[1], y*self.cell_size[0]),
                           ((x+1)*self.cell_size[1], y*self.cell_size[0]),
                           ((x+1)*self.cell_size[1], (y+1)*self.cell_size[0]),
                           (x*self.cell_size[1], (y+1)*self.cell_size[0])]
                if map_[y, x] == 1:
                    color = (46, 204, 113)  # robi
                elif map_[y, x] == 2:
                    color = (52, 152, 219)  # goal
                elif map_[y, x] == -1:
                    color = (52, 73, 94)  # rock
                else:
                    color = (255, 255, 255)
                pygame.draw.polygon(self.window, color, polygon, 0)

        pygame.display.flip()
        time.sleep(delta_t)

    def to_map(self):
        map_ = np.zeros((self._height, self._width))

        # rocks
        for i in range(len(self._rocks_pose)):
            map_[self._rocks_pose[i]] = -1

        # robi
        if not self._robi_in_lava:
            map_[self._robi_pose] = 1

        # goal
        map_[self._goal_pose] = 2

        return map_

    def reset(self):
        self._robi_in_lava = False

        # random position for the robi
        robi_init = np.random.randint(0, self._height-1, size=1)[0]
        self._robi_pose = (robi_init, 0)

        # random position for rocks
        rocks_init = np.random.randint(0, self._height-1, size=(self._width - 2))
        self._rocks_pose = []
        for i in range(rocks_init.shape[0]):
            self._rocks_pose.append((rocks_init[i], i + 1))

        # random position for the goal
        goal_init = np.random.randint(0, self._height-1, size=1)[0]
        self._goal_pose = (goal_init, self._width - 1)

        return self.to_map()
