
import os
import gym
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminal = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminal[:]


# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, n_latent_var):
#         super(ActorCritic, self).__init__()

#         self.action_layer = nn.Sequential(
#             nn.Linear(state_dim, n_latent_var),
#             nn.ReLU(),
#             nn.Linear(n_latent_var, n_latent_var),
#             nn.ReLU(),
#             nn.Linear(n_latent_var, n_latent_var),
#             nn.Softmax(dim=-1)
#         )

#         self.value_layer = nn.Sequential(
#             nn.Linear(state_dim, n_latent_var),
#             nn.ReLU(),
#             nn.Linear(n_latent_var, n_latent_var),
#             nn.ReLU(),
#             nn.Linear(n_latent_var, 1)
#         )

#     def forward(self):
#         raise NotImplementedError

#     def act(self, state, memory):
#         state.torch.from_numpy(state).float().to(device)
#         action_probs = self.act(state)
#         dist = Categorical(action_probs)
#         action = dist.sample()

#         memory.state.append(state)
#         memory.actions.append(action)
#         memory.logprobs.append(dist.log_prob(action))

#         return action.item()

class ParallelAgents:
    def __init__(self, num_agent, env_name, seed=None, render=False, episode=3):
        assert num_agent > 0, "Number of agents must be positive"
        self.render = render
        self.env_name = env_name
        self.seed = seed
        self.episodes = episode
        self.agents = []
        # self.reward.share_memory()
        for agent_id in range(num_agent):
            self.env = gym.make(self.env_name)
            self.env.reset()
            self.env.seed(None)
            self.agents.append((self.env, agent_id))
            print("Agent {} initialized".format(agent_id))

    def act(self, env, agent_id):
        agent_reward = []
        for i in range(self.episodes):
            env.reset()
            eps_reward = 0
            done = False
            while not done:
                # observation, reward, done, info
                _, eps_reward, done, _ = env.step(env.action_space.sample())
                eps_reward += eps_reward
                if self.render:
                    env.render()
                if done:
                    # print("Completed Episode, Reward: {0}".
                    #       format(eps_reward))
                    agent_reward.append(eps_reward)

        print("Agent {} completed {} episodes, Worker process ID {}".
              format(agent_id, self.episodes, os.getpid()))
        # print(self.reward)
        print(agent_reward)
        print(len(agent_reward))

        return agent_reward

    def parallelAct(self):
        processes = []
        for agent in self.agents:
            p = mp.Process(target=self.act, args=agent)
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        return self.reward


def main():
    start = time.perf_counter()
    env_nam = "LunarLander-v2"
    agents = ParallelAgents(
        num_agent=3, env_name=env_nam, render=True, episode=30
    )
    reward = agents.parallelAct()
    end = time.perf_counter()
    print(reward)
    print("Program done, {:.2f} sec elapsed".format(end-start))


if __name__ == "__main__":
    main()
