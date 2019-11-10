
import os
import gym
import time
from itertools import repeat
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class Memory:
    def __init__(self):
        # self.actions = torch.tensor(1).share_memory_()
        # self.states = torch.tensor(1).share_memory_()
        # self.logprobs = torch.tensor(1).share_memory_()
        # self.rewards = torch.tensor(1).share_memory_()
        # self.is_terminal = torch.tensor(1).share_memory_()
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminal = []

    # def clear_memory(self):
    #     del self.actions[:]
    #     del self.states[:]
    #     del self.logprobs[:]
    #     del self.rewards[:]
    #     del self.is_terminal[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Softmax(dim=-1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state.torch.from_numpy(state).float().to(device)
        action_probs = self.act(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.state.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma,
                 K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_dim,
            action_dim,
            n_latent_var
        ).to(device)

        self.policy_old = ActorCritic(
            state_dim,
            action_dim,
            n_latent_var
        ).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            betas=betas
        )

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        pass
    

class Agent:
    """creating a single agent, which contains the agent's gym environment
    and relevant information, such as its ID
    """

    def __init__(self, agent_id, env_name, seed=None, render=False):
        self.render = render
        self.env_name = env_name
        self.seed = seed
        self.agent_id = agent_id

        self.env = gym.make(self.env_name)
        self.env.reset()
        self.env.seed(None)
        print("Agent {} initialized".format(agent_id))


class ParallelAgents:
    """a wrapper for running parallel agents.
    """

    def __init__(self, num_agent, episode, env_name,
                 max_timestep, seed=None, render=False):
        """creates instances of agents here
        """
        assert num_agent > 0, "Number of agents must be positive"
        self.memory = Memory()
        self.max_timestep = max_timestep
        self.episodes = episode
        self.agents = []
        for agent_id in range(num_agent):
            env = Agent(agent_id, env_name=env_name, seed=seed, render=render)
            self.agents.append(env)

    def act(self, agent, action):
        """having an agent to take a step. This function is made so it can be
        used for parallel agents
        Args:
            agent (class Agent): the agent object for taking actions
        Return: agent_reward (list): a list of reward from each episode
        """
        print(action)
        for i in range(self.episodes):

            # prints episode progress
            if i+1 % 100 == 0:
                print("Agent {} at episode {}".format(agent.agent_id, i))

            state = agent.env.reset()
            timestep = 0
            for t in range(self.max_timestep):
                timestep += 1
                # observation, reward, done, info
                state, reward, done, _ = agent.env.step(
                    agent.env.action_space.sample())

                # save reward and environment state into memory
                self.memory.rewards.append(reward)
                self.memory.is_terminal.append(done)

                if agent.render:
                    agent.env.render()

                if done:
                    break

        print("Agent {} completed {} episodes, Worker process ID {}".
              format(agent.agent_id, self.episodes, os.getpid()))
        return self.memory

    def parallelAct(self):
        """run the act() function in a process pool. The result returned is a
        concatnated list in the sequence of the process starting order
        """
        p = mp.Pool()
        # TODO actually send in an action to act, the repeat('hi') is a place holder
        result = p.starmap(self.act, zip(self.agents, repeat('hi')))
        print(result)


def main():
    # configuration
    env_nam = "LunarLander-v2"
    num_agent = 3
    render = False
    episode = 5
    max_timestep = 300
    seed = None

    # run parallel agents
    start = time.perf_counter()
    agents = ParallelAgents(
        num_agent=num_agent, env_name=env_nam, max_timestep=max_timestep,
        seed=seed, render=render, episode=episode
    )
    agents.parallelAct()
    end = time.perf_counter()

    print("Program done, {:.2f} sec elapsed".format(end-start))


if __name__ == "__main__":
    main()
