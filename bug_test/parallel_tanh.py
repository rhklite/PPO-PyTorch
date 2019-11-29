# TODO: test if code is faster with CPU or GPU
import os
import gym
import time
from datetime import date
import collection
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
mp.set_start_method('spawn', True)
writer = SummaryWriter()


class Memory:
    def __init__(self, num_agents, update_timestep, state_dim, agent_policy):
        """a preallocated, shared memory space for each agents to pool the
        collected experience

        Args:
            num_agents (int): the number of agents that are running in parallel
                              used for calculating size of allocated memory
            update_timestep (int): number of timesteps until update, also used
                                   for calculating size of allocated memory
            state_dim (int) : the size of the state observation
            agent_policy (object): a network that contains the policy that the
                                   agents will be acting on
        """

        self.states = torch.zeros(
            (update_timestep*num_agents, state_dim)).to(device).share_memory_()
        self.actions = torch.zeros(
            update_timestep*num_agents).to(device).share_memory_()
        self.logprobs = torch.zeros(
            update_timestep*num_agents).to(device).share_memory_()
        self.disReturn = torch.zeros(
            update_timestep*num_agents).to(device).share_memory_()

        # TODO: find a way to share this, try .share_memory_() to policy_old
        self.agent_policy = agent_policy


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
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

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            betas=betas
        )

        self.policy_old = ActorCritic(
            state_dim,
            action_dim,
            n_latent_var
        ).to(device).share_memory()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        old_states = memory.states.detach()
        old_actions = memory.actions.detach()
        old_logprobs = memory.logprobs.detach()
        old_disReturn = memory.disReturn.detach()

        # if old_disReturn.std() == 0:
        #     old_disReturn = (old_disReturn - old_disReturn.mean()) / 1e-5
        # else:
        #     old_disReturn = (old_disReturn - old_disReturn.mean()) / \
        #         (old_disReturn.std())

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


class Agent(mp.Process):
    """creating a single agent, which contains the agent's gym environment
    and relevant information, such as its ID
    """

    def __init__(self, name, memory, pipe, env_name, max_episode, max_timestep,
                 update_timestep, gamma, seed=None, render=False):
        """initialization

        Args:
            memory (object): shared memory object
            pipe (object): connection used to talk to the main process
            name (str): a number that represent the ith agent. Also used
                        to determine the memory index for this agent to pool
            max_timestep (int): limit steps to this for each episode. Used
                                for environment that does not have step limit
            update_timestep (int): step to take in the env before update policy
        """
        mp.Process.__init__(self, name=name)

        # variables usef for multiprocessing
        self.memory = memory
        self.pipe = pipe

        # variables for training
        self.max_episode = max_episode
        self.max_timestep = max_timestep
        self.update_timestep = update_timestep
        self.gamma = gamma
        self.render = render
        self.env = gym.make(env_name)
        self.env.reset()
        self.env.seed(seed)

    def run(self):
        print("Agent {} started, Process ID {}".format(self.name, os.getpid()))
        actions = []
        rewards = []
        states = []
        logprobs = []
        is_terminal = []
        timestep = 0
        # lists to collect agent experience
        # variables for logging
        reward_msg = 0
        episodes_lapsed = 0
        # ep_reward_log = []

        for i_episodes in range(1, self.max_episode+2):
            state = self.env.reset()

            if i_episodes == self.max_episode+1:
                self.pipe.send("END")
                break

            for i in range(self.max_timestep):

                timestep += 1

                states.append(state)

                action, logprob = self.memory.agent_policy.act(state)
                state, reward, done, _ = self.env.step(action)

                actions.append(action)
                logprobs.append(logprob)
                rewards.append(reward)
                is_terminal.append(done)

                reward_msg += reward

                if timestep % self.update_timestep == 0:
                    stateT, actionT, logprobT, disReturn = \
                        self.experience_to_tensor(
                            states, actions, rewards, logprobs, is_terminal)

                    self.add_experience_to_pool(stateT, actionT,
                                                logprobT, disReturn)

                    episodes_lapsed = i_episodes - episodes_lapsed
                    avg_reward = reward_msg/episodes_lapsed
                    episodes_lapsed = i_episodes

                    self.pipe.send((self.name, i_episodes, avg_reward))
                    msg = self.pipe.recv()
                    if msg == "RENDER":
                        self.render = True
                    timestep = 0
                    reward_msg = 0
                    actions = []
                    rewards = []
                    states = []
                    logprobs = []
                    is_terminal = []

                if done:

                    break

                if self.render:
                    time.sleep(0.005)
                    self.env.render()

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

        start_idx = int(self.name)*self.update_timestep
        end_idx = start_idx + self.update_timestep
        self.memory.states[start_idx:end_idx] = stateTensor
        self.memory.actions[start_idx:end_idx] = actionTensor
        self.memory.logprobs[start_idx:end_idx] = logprobTensor
        self.memory.disReturn[start_idx:end_idx] = disReturnTensor


def main():

    ######################################
    # Training Environment configuration
    env_name = "LunarLander-v2"
    # env_name = "CartPole-v0"
    num_agents = 1
    max_timestep = 300        # per episode the agent is allowed to take
    update_timestep = 2000    # total number of steps to take before update
    max_episode = 50000
    seed = None               # seeding the environment
    render = False
    solved_reward = 230

    # gets the parameter about the environment
    sample_env = gym.make(env_name)
    state_dim = sample_env.observation_space.shape[0]
    action_dim = 4
    action_dim = sample_env.action_space.n
    print("State dim {} Action dim {}".format(state_dim, action_dim))
    del sample_env

    # PPO & Network Parameters
    n_latent_var = 64
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    # logging settings
    log_interval = 5
    ######################################

    ppo = PPO(state_dim, action_dim, n_latent_var,
              lr, betas, gamma, K_epochs, eps_clip)

    # TODO verify if i should pass in ppo.policy_old
    memory = Memory(num_agents, update_timestep, state_dim, ppo.policy_old)

    # starting agents and pipes
    agents = []
    pipes = []

    for agent_id in range(num_agents):
        p_start, p_end = mp.Pipe()
        agent = Agent(str(agent_id), memory, p_end, env_name, max_episode,
                      max_timestep, update_timestep, gamma, seed, render)
        agents.append(agent)
        pipes.append(p_start)

    for agent in agents:
        agent.start()

    # starting training loop
    update_iteration = 0
    current_reward = 0
    while True:

        agent_info = []
        pipe_to_remove = []
        for pipe in pipes:
            info = pipe.recv()
            if info == "END":
                pipe_to_remove.append(pipe)
            else:
                agent_info.append(info)
        pipes = [x for x in pipes if x not in pipe_to_remove]

        # this checks if all agents have finished
        if len(pipes) == 0:
            break
        else:
            ppo.update(memory)
            update_iteration += 1

        if update_iteration % log_interval == 0:
            agents_avg_reward = 0
            for info in agent_info:
                print("Agent {} Episode {}, Avg Reward/Episode {:.2f}"
                      .format(info[0], info[1], info[2]))
                agents_avg_reward += info[2]

                writer.add_scalar('Agent {} Reward/Episode'
                                  .format(info[0]), info[2], update_iteration)
            current_reward = round(agents_avg_reward/len(agent_info), 2)
            print("Main: Update Iteration: {}, Avg Reward Amongst Agents: {}\n"
                  .format(update_iteration, current_reward))
            writer.add_scalar(
                'Reward/Episode', current_reward, update_iteration)

        if solved_reward <= current_reward:
            print("==============TanH SOLVED==============")
            break

        # if update_iteration % 50 == 0:
        #     msg = "RENDER"
        # else:
        #     msg = update_iteration
        msg = update_iteration
        for pipe in pipes:
            pipe.send(msg)

    today = date.today()
    torch.save(ppo.policy.state_dict(),
               './v3_tanh_PPO_{}_{}_{}_{}.pth'.format(env_name, num_agents, current_reward, today))

    for agent in agents:
        agent.terminate()


if __name__ == "__main__":

    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("Training Completed, {:.2f} sec elapsed".format(end-start))
