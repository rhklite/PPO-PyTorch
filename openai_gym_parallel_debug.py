# TODO: test if code is faster with CPU or GPU
# TODO: see how to use FP16 instead of FP32
# TODO: remove unnecessary .to(device) to save memory
# FIXME: rewrite multiprocessing step to stop memory leak
# MAJOR CHANGE 1: removed parallel processing, only 1 agent, directly used func: env_step(). no improvement
# Major change 2: noticed difference in the ratio calculated in the first update cycle against the baseline code (resolved)
#   fix attempt 1: pass in PPO object to act instead of load ppo state dict, didn't work
#   fix: noticed the appended logprob has 1 offset. moved append after act, before env.step(action) in the env_step function
#   big performance boost from that... wtf
# FIXME: figure out why the training saturates...
# 
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

# mp.set_start_method('spawn', True)
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

        # TODO: check if this is the right place to normalize reward
        # TODO: the normalized discounted reward is ~10 times larger than baseline code... weird might be cause of not learning?
        old_disReturn = (old_disReturn - old_disReturn.mean()) / \
            (old_disReturn.std()+1e-5)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta/ pi_theta_old):
            # using exponential returns the log back to non-log version
            # TODO this ratio looks really weird, baseline code is all 1, which should be correct. If this ratio is not 1, it means the acting policy and the current policy does not have the same network parameters
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
        # # the policy which the parallel agents will act on
        # self.agent_policy = ActorCritic(
        #     state_dim, action_dim, n_latent_var).to(device)
        # parameters
        self.timestep = timestep
        self.gamma = gamma
        self.render = render

        # for multiprocessing
        self.memory = memory
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
        logprobTensor = torch.tensor(logprobs).float().detach()

        # convert reward into discounted return
        discounted_reward = 0
        disReturnTensor = []
        for reward, done in zip(reversed(rewards),
                                reversed(is_terminal)):
            if done:
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
        # lock.acquire()
        self.memory.status.add_(self.memory.timestep)
        # lock.release()

    def log_progress(self, epoch_reward):
        pass

    def env_step(self, agent, ppo, render):
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

        # variables for logging purpose
        episode_reward = 0
        epoch_reward = []

        state = agent.env.reset()

        for t in range(self.timestep):

            action, logprob = ppo.policy_old.act(state)

            states.append(state)
            actions.append(action)
            logprobs.append(logprob)

            state, reward, done, _ = agent.env.step(action)

            # save reward and environment state into memory
            rewards.append(reward)
            is_terminal.append(done)

            # record episode reward for logging
            episode_reward += reward

            if done:
                state = agent.env.reset()

                # record episode reward for logging
                epoch_reward.append(episode_reward)
                episode_reward = 0

            if render:
                time.sleep(0.001)
                agent.env.render()
            # if agent.render:
            #     agent.env.render()

        epoch_reward.append(episode_reward)
        # convert the experience collected into memory
        stateTensor, actionTensor, logprobTensor, disReturnTensor = \
            self.experience_to_tensor(
                states, actions, rewards, logprobs, is_terminal)

        self.add_experience_to_pool(
            stateTensor, actionTensor, logprobTensor, disReturnTensor)

        if len(epoch_reward) > 0:
            epoch_reward = float(sum(epoch_reward))/float(len(epoch_reward))
        else:
            print(states[0:2])
            print(actions[0:2])
            print(rewards[0:2])
            print(is_terminal[0:2])

        # print("Agent {} took {} steps, average reward {}".
        #       format(agent.agent_id, self.timestep, epoch_reward))

        return epoch_reward

    def parallelAct(self, n_iter, n_iter_render):
        """have each agent make an action using process pool. The result returned is a
        concatnated list in the sequence of the process starting order
        """
        self.memory.clear_memory()

        if n_iter > n_iter_render:
            self.render = True

        #############################
        # pool method

        m = mp.Manager()
        lock = m.Lock()
        p = mp.Pool()
        epoch_reward = p.starmap(self.env_step, zip(self.agents, repeat(lock)))
        print("terminating process")
        p.terminate()
        return epoch_reward
        #############################

        #############################
        # Regular method
        # processes = []
        # for agent in self.agents:
        #     p = mp.Process(target=self.env_step, args=(agent,))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     print("process {} terminated".format(p))
        #     p.join()
        #     # need to terminate process after they are done
        #     # since new processes are spawned every training cycle.
        #     # if not terminated, RAM will explode
        #     p.terminate()
        #############################


def main():

    ######################################
    # Training Environment configuration
    # env_name = "LunarLander-v2"
    env_name = "CartPole-v0"
    num_agent = 1
    render = False
    timestep = 2000
    seed = None
    training_iter = 2000      # total number of training episodes

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

    train_start = time.perf_counter()
    for i in range(1, training_iter+1):

        memory.clear_memory()

        # making a copy of the actor for the parallel agents
        # agents.agent_policy.load_state_dict(ppo.policy_old.state_dict())

        # tell all the parallel agents to act according to the policy
        # memory is the returned experience from all agents
        epoch_reward = agents.env_step(agents.agents[0], ppo, render)
        # epoch_reward = agents.parallelAct(i, 20)
        # avg_reward = sum(epoch_reward)/len(epoch_reward)
        # writer.add_scalar('Average Reward ', i,
        #                   sum(epoch_reward)/len(epoch_reward))
        # update the policy with the memories collected from the agents
        ppo.update(memory)

        if i % 20 == 0:

            train_end = time.perf_counter()
            print("Training {} done, {:.2f} sec elapsed, reward {}".
                  format(i, train_end-train_start, epoch_reward))
            train_start = time.perf_counter()
        # torch.save(ppo.policy.state_dict(),
        #            './parallel_results/Step{}_R{:1f}_{}.pth'
        #            .format(i, epoch_reward, env_name))
        # if epoch_reward > 200:
        #     render = True
    end = time.perf_counter()

    print("Training Completed, {:.2f} sec elapsed".format(end-start))

    torch.save(ppo.policy.state_dict(),
                './parallel_results/Step{}_R{:1f}_{}.pth'
                .format(i, epoch_reward, env_name))


if __name__ == "__main__":
    main()
