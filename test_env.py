"""
Test trained model and solve environment observation black problem
"""

from env import Env, EnvBatcher
import cv2
from torchvision.utils import make_grid, save_image
import torch
from utils import lineplot, write_video

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.

ENV_NAME = 'HalfCheetah-v2'
SYMBOLIC = False
SEED = 0
MAX_EPI_LEN = 1000
ACTION_REPEAT = 2
BIT_DEPTH = 5

# env = Env(ENV_NAME, SYMBOLIC, SEED, MAX_EPI_LEN, ACTION_REPEAT, BIT_DEPTH)
# env.reset()
# for i in range(100):
#     action = env.sample_random_action()
#     next_observation, reward, done = env.step(action)
#     cv2.imshow("env", next_observation)
#     cv2.waitKey()
#     cv2.destroyWindow()
#     print(next_observation.shape)
#     env._env.render()

# # Gym env
# import gym
# env = gym.make('HalfCheetah-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()

# Test make video
NUM_ENV = 2
test_env = Env(ENV_NAME, SYMBOLIC, SEED, MAX_EPI_LEN, ACTION_REPEAT, BIT_DEPTH)
test_envs = EnvBatcher(Env, (ENV_NAME, SYMBOLIC, SEED, MAX_EPI_LEN, ACTION_REPEAT, BIT_DEPTH), {}, NUM_ENV)
test_envs.reset()
video_frames = []

for i in range(100):
    # action = test_envs.sample_random_action()
    actions = [test_env.sample_random_action()] * NUM_ENV
    next_observation, reward, done = test_envs.step(actions)
    # cv2.imshow("env", next_observation)
    # cv2.waitKey()
    # cv2.destroyAllWindow()
    video_frames.append(make_grid(next_observation + 0.5, nrow=5).numpy())  # Decentre
    save_image(torch.as_tensor(video_frames[-1]), 'test_episode{}.png'.format(i))

# write_video(video_frames, 'test_episode')  # Lossy compression
# save_image(torch.as_tensor(video_frames[-1]), 'test_episode.png')
