import gym
from PPO import PPO, Memory
from PIL import Image
import torch
import numpy as np


from os import listdir
from os.path import isfile, join


def test():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # env_name = "CartPole-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    max_timesteps = 500
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 100
    max_timesteps = 300
    render = False
    save_gif = False

    test_batch = True
    directory = "./bug_test/test/original/"
    filename = "v3_PPO_LunarLander-v2_1_259.15_2019-11-20.pth"

    if not test_batch:
        filenames = [filename]
    if test_batch:
        filenames = [f for f in listdir(
            directory) if isfile(join(directory, f))]

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var,
              lr, betas, gamma, K_epochs, eps_clip)
    for filename in filenames:
        ppo.policy_old.load_state_dict(torch.load(directory+filename))
        average_reward = []
        for ep in range(1, n_episodes+1):
            ep_reward = 0
            state = env.reset()
            for t in range(max_timesteps):
                action = ppo.policy_old.act(state, memory)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
                if done:
                    break
            if render:
                print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
            average_reward.append(ep_reward)
            ep_reward = 0
            env.close()

        print(filename)
        print("Tested {} Episode, Average Reward {:.2f}, Std {:.2f}\n".format(
            n_episodes, np.average(average_reward), np.std(average_reward)))


if __name__ == '__main__':
    test()
