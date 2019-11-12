# TODO: test if code is faster with CPU or GPU
# TODO: see how to use FP16 instead of FP32
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
# device = "cpu"

# uncomment this and add .to(device) after agent_policy
#  if sending agent_policy to GPU, it actually made it slower...

mp.set_start_method('spawn', True)
writer = SummaryWriter()


class Memory:
    def __init__(self):
        self.states = torch.Tensor().to(device)
        self.actions = torch.Tensor().to(device)
        self.logprobs = torch.Tensor().to(device)
        self.disReturn = torch.Tensor().to(device)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim),
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

    # def act(self, state, memory):
    def act(self, state):
        """pass the state observed into action_layer network to determine the action
        that the agent should take.

        Args:
            state (list): a list contatining the state observations

        Return: action (int): a number that indicates the action to be taken
                              for gym environment
                log_prob (tensor): a tensor that contains the log probability
                                   of the action taken. require_grad is true


        """
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return (action.item(), dist.log_prob(action))


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
        # TODO: implement policy updaaate
        # requires state, action, reward, old_logprob, all of which are
        # 1D tensors
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
    """facilitating running multiple agents and collect their experience into
       a shared memory.
    """

    def __init__(self, num_agent, env_name, timestep, gamma,
                 state_dim, action_dim, n_latent_var,
                 seed=None, render=False):
        """creates instances of parallel agents here for

        Args:
            num_agent (int): number of parallel agents to run
            env_name (string): the OpenAI gym environment to run
            timestep (int): the number of time steps to take for each agent
                before returning the collected memory for update. timestep
                instead of episode is used here because the paper asks for
                partial trajectories
            gamma (float): the reward discount rate
            state_dim (int): size of state observation, used for creating
                agent policy
            action_dim (int): size of action space, used for creating agent
                policy
            n_latent_var (int): size of hidden layer, used for creating agent
                policy
            seed (int): random seed for gym environment
            render (bool): whether to render the gym environment as it trains
        """
        assert num_agent > 0, "Number of agents must be positive"
        # the policy which the parallel agents will act on
        self.agent_policy = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)
        self.memory = Memory()
        self.timestep = timestep
        self.gamma = gamma
        self.agents = []
        for agent_id in range(num_agent):
            env = Agent(agent_id, env_name=env_name, seed=seed, render=render)
            self.agents.append(env)

    def experience_to_tensor(self, states, actions, rewards,
                             logprobs, is_terminal):
        """converts the experience collected by the agent into tensors

        Args:
            states (list): a list of states visited by the agent
            actions (list): a list of actions that the agent took
            rewards (list): a list of reward that the agent recieved
            logprobs (list): a list of log probabiliy of the action happening
            is_terminal (list): for each step, indicate if that the agent is in
                                the terminal state

        Return:
            stateTensor (tensor): the states converted to a 1D tensor
            actionTensor (tensor): the actions converted to a 1D tensor
            disReturnTensor (tensor): discounted return as a 1D tensor
            logprobTensor (tensor): the logprobs converted to a 1D tensor
        """

        # convert state, action and log prob into tensor
        stateTensor = torch.tensor(states).float().to(device)
        actionTensor = torch.tensor(actions).float().to(device)
        logprobTensor = torch.tensor(logprobs).float().to(device)

        # convert reward into discounted return
        discounted_reward = 0
        disReturnTensor = []
        for reward, is_terminal in zip(reversed(rewards),
                                       reversed(is_terminal)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma*discounted_reward)
            disReturnTensor.insert(0, discounted_reward)

        disReturnTensor = torch.tensor(disReturnTensor)

        return (stateTensor, actionTensor, logprobTensor, disReturnTensor)

    def combine_experience(self, result):
        pass

    def env_step(self, agent):
        """having an agent to take a step in the environment. This function is
        made so it can be used for parallel agents
        Args:
            agent (obj Agent): the agent object for taking actions
        """
        actions = []
        rewards = []
        states = []
        logprobs = []
        is_terminal = []

        state = agent.env.reset()

        for t in range(self.timestep):

            (action, logprob) = self.agent_policy.act(state)
            state, reward, done, _ = agent.env.step(action)

            # save reward and environment state into memory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            logprobs.append(logprob)
            is_terminal.append(done)

            if done:
                state = agent.env.reset()

            if agent.render:
                agent.env.render()

        # convert the experience collected into memory
        (stateTensor, actionTensor, logprobTensor, disReturnTensor) = \
            self.experience_to_tensor(
            states, actions, rewards, logprobs, is_terminal)

        del states[:]
        del rewards[:]
        del actions[:]
        del logprobs[:]

        memory = Memory()
        memory.states = stateTensor
        memory.actions = actionTensor
        memory.logprobs = logprobTensor
        memory.disReturn = disReturnTensor

        print("Agent {} took {} steps, Worker process ID {}".
              format(agent.agent_id, self.timestep, os.getpid()))
        return memory

    def parallelAct(self):
        """have each agent make an action using process pool. The result returned is a
        concatnated list in the sequence of the process starting order
        """
        # FIXME: make memory sharing work
        p = mp.Pool()
        pooledMemory = p.map(self.env_step, self.agents)
        for memory in pooledMemory:
            self.memory.states = torch.cat(
                (self.memory.states, memory.states.to(device)))
            self.memory.actions = torch.cat(
                (self.memory.actions, memory.actions.to(device)))
            self.memory.logprobs = torch.cat(
                (self.memory.logprobs, memory.logprobs.to(device)))
            self.memory.disReturn = torch.cat(
                (self.memory.disReturn, memory.disReturn.to(device)))

        return self.memory


def main():

    # Training Environment configuration
    env_name = "LunarLander-v2"
    num_agent = 4
    render = False
    timestep = 5
    seed = None
    training_iter = 1        # total number of training episodes

    # gets the parameter about the environment
    tmp_env = gym.make(env_name)
    state_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.n
    del tmp_env
    # PPO & Network Parameters
    n_latent_var = 64
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    # timer to see time it took to train
    start = time.perf_counter()

    # initialize PPO policy instance
    ppo = PPO(state_dim, action_dim, n_latent_var,
              lr, betas, gamma, K_epochs, eps_clip)

    # initialize parallel agent intance
    agents = ParallelAgents(num_agent, env_name, timestep,
                            gamma, state_dim, action_dim,
                            n_latent_var, seed, render)

    for i in range(1, training_iter+1):

        train_start = time.perf_counter()

        # making a copy of the actor for the parallel agents
        agents.agent_policy.load_state_dict(ppo.policy_old.state_dict())

        # tell all the parallel agents to act according to the policy
        # memory is the returned experience from all agents
        pooledMemory = agents.parallelAct()

        print(pooledMemory.actions)
        # update the policy with the memories collected from the agents
        ppo.update(pooledMemory)
        train_end = time.perf_counter()

        print("Training iteration {} done, {:.2f} sec elapsed".
              format(i, train_end-train_start))

    end = time.perf_counter()
    print("Training Completed, {:.2f} sec elapsed".format(end-start))


if __name__ == "__main__":
    main()
