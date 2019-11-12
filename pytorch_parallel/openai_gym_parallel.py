# TODO: test if code is faster with CPU or GPU
# TODO: see how to use FP16 instead of FP32
# TODO: remove unnecessary .to(device) to save memory
# FIXME: rewrite multiprocessing step to stop memory leak
import os
import gym
import time
from itertools import repeat
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# uncomment this and add .to(device) after agent_policy
#  if sending agent_policy to GPU, it actually made it slower...

mp.set_start_method('spawn', True)
writer = SummaryWriter()


class Memory:
    def __init__(self, timestep, num_agents, state_dim):
        self.timestep = timestep
        self.status = torch.tensor([0]).to(device).share_memory_()
        self.states = torch.zeros(
            (timestep*num_agents, state_dim)).to(device).share_memory_()
        self.actions = torch.zeros(
            timestep*num_agents).to(device).share_memory_()
        self.logprobs = torch.zeros(
            timestep*num_agents).to(device).share_memory_()
        self.disReturn = torch.zeros(
            timestep*num_agents).to(device).share_memory_()

    def clear_memory(self):
        self.status[:] = 0
        self.states[:] = 0
        self.actions[:] = 0
        self.logprobs[:] = 0
        self.disReturn[:] = 0


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

        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        # TODO verify this code if it works
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        dist_entropy = dist.entropy()

        # why are we getting the log prob here?
        # the loss function used by PPO doesn't include a log term
        action_logprobs = dist.log_prob(action)

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


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
        # TODO verify if this code works
        old_states = memory.states.detach()
        old_actions = memory.actions.detach()
        old_logprobs = memory.logprobs.detach()
        old_disReturn = memory.disReturn.detach()

        old_disReturn = (old_disReturn - old_disReturn.mean()) / \
            (old_disReturn.std()+1e-5)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta/ pi_theta_old):
            # using exponential returns the log back to non-log version
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding the surrogate loss:
            advantages = old_disReturn - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip)*advantages

            # see paper for this loss formulation; this loss function
            # need to be used if the policy and value network shares
            # parameters, however, i think the author of this code just used
            # this, even though the two network are not sharing parameters
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, old_disReturn) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


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

    def __init__(self, memory, num_agent, env_name, timestep, gamma,
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
        # TODO verify memory sharing if working
        self.memory = memory
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
        stateTensor = torch.tensor(states).float()
        actionTensor = torch.tensor(actions).float()
        logprobTensor = torch.tensor(logprobs).float()

        # convert reward into discounted return
        discounted_reward = 0
        disReturnTensor = []
        for reward, is_terminal in zip(reversed(rewards),
                                       reversed(is_terminal)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma*discounted_reward)
            disReturnTensor.insert(0, discounted_reward)

        disReturnTensor = torch.tensor(disReturnTensor).float()

        return stateTensor, actionTensor, logprobTensor, disReturnTensor

    def add_experience_to_pool(self, stateTensor, actionTensor,
                               logprobTensor, disReturnTensor):

        position = self.memory.status.item()
        increment = position + self.memory.timestep

        self.memory.states[position:increment] = stateTensor
        self.memory.actions[position:increment] = actionTensor
        self.memory.logprobs[position:increment] = logprobTensor
        self.memory.disReturn[position:increment] = disReturnTensor
        self.memory.status.add_(self.memory.timestep)

    def env_step(self, agent):
        """having an agent to take a step in the environment. This function is
        made so it can be used for parallel agents
        Args:
            agent (obj Agent): the agent object for taking actions
        """
        # TODO remove lock in the future if its not going to be used
        actions = []
        rewards = []
        states = []
        logprobs = []
        is_terminal = []

        state = agent.env.reset()

        for t in range(self.timestep):

            action, logprob = self.agent_policy.act(state)
            state, reward, done, _ = agent.env.step(action)

            # save reward and environment state into memory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            logprobs.append(logprob)
            is_terminal.append(done)

            # TODO remove this section after debugging finished
            # states.append([agent.agent_id for i in range(8)])
            # actions.append(agent.agent_id)
            # rewards.append(agent.agent_id)
            # logprobs.append(agent.agent_id)
            # is_terminal.append(done)

            if done:
                state = agent.env.reset()

            if agent.render:
                agent.env.render()

        # convert the experience collected into memory
        stateTensor, actionTensor, logprobTensor, disReturnTensor = \
            self.experience_to_tensor(
                states, actions, rewards, logprobs, is_terminal)

        self.add_experience_to_pool(
            stateTensor, actionTensor, logprobTensor, disReturnTensor)

        # free up memory
        del actions, rewards, states, logprobs, is_terminal
        del stateTensor, actionTensor, logprobTensor, disReturnTensor

        print("Agent {} took {} steps, Worker process ID {}".
              format(agent.agent_id, self.timestep, os.getpid()))
        # return memory

    def parallelAct(self):
        """have each agent make an action using process pool. The result returned is a
        concatnated list in the sequence of the process starting order
        """
        self.memory.clear_memory()
        # p = mp.Pool()
        # p.map(self.env_step, self.agents)
        # del p
        # p = mp.Pool()
        processes = []
        for agent in self.agents:
            p = mp.Process(target=self.env_step, args=(agent,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            # need to terminate process after they are done
            # since new processes are spawned every training cycle.
            # if not terminated, RAM will explode
            p.terminate()


def main():

    ######################################
    # Training Environment configuration
    env_name = "LunarLander-v2"
    num_agent = 4
    render = False
    timestep = 2000
    seed = None
    training_iter = 200      # total number of training episodes

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
    ######################################

    # timer to see time it took to train
    start = time.perf_counter()

    # initialize PPO policy instance
    ppo = PPO(state_dim, action_dim, n_latent_var,
              lr, betas, gamma, K_epochs, eps_clip)

    # preallocating shared memory object
    memory = Memory(timestep, num_agent, state_dim)

    # initialize parallel agent intance
    agents = ParallelAgents(memory, num_agent, env_name, timestep,
                            gamma, state_dim, action_dim,
                            n_latent_var, seed, render)

    for i in range(1, training_iter+1):

        train_start = time.perf_counter()

        # making a copy of the actor for the parallel agents
        agents.agent_policy.load_state_dict(ppo.policy_old.state_dict())

        # tell all the parallel agents to act according to the policy
        # memory is the returned experience from all agents
        agents.parallelAct()

        # update the policy with the memories collected from the agents
        ppo.update(memory)

        train_end = time.perf_counter()
        print("Training iteration {} done, {:.2f} sec elapsed".
              format(i, train_end-train_start))

    end = time.perf_counter()
    torch.save(ppo.policy.state_dict(),
               './Parallel_PPO_{}.pth'.format(env_name))
    print("Training Completed, {:.2f} sec elapsed".format(end-start))


if __name__ == "__main__":
    main()
