# TODO: implement batching
# TODO: implement GAE
# TODO: implement value clipping (check openAI baseline)
# TODO: see if i need to do value averaging
# FIXME: subprocess hangs when terminate due to max steps

import os
import gym
import time
from print_util import *
from datetime import date
from collections import namedtuple
import csv
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
mp.set_start_method('spawn', True)
# writer = SummaryWriter()

# creating msgs for communication between subprocess and main process.
# for when agent reached logging episode
MsgRewardInfo = namedtuple('MsgRewardInfo', ['agent', 'episode', 'reward'])
# for when agent reached update timestep
MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])
# for when agent reached max episodes
MsgMaxReached = namedtuple('MsgMaxReached', ['agent', 'reached'])


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

    def act(self, state, evaluate):
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

        if evaluate:
            _, action = action_probs.max(0)
        else:
            action = dist.sample()

        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
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

        if old_disReturn.std() == 0:
            old_disReturn = (old_disReturn - old_disReturn.mean()) / 1e-5
        else:
            old_disReturn = (old_disReturn - old_disReturn.mean()) / \
                (old_disReturn.std())

        # old_disReturn = (old_disReturn - old_disReturn.mean()) / \
        #     (old_disReturn.std()+1e-5)

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
                self.MseLoss(state_values, old_disReturn) - 0.005*dist_entropy

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
                 update_timestep, log_interval, gamma, seed=None, render=False):
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
        self.proc_id = name
        self.memory = memory
        self.pipe = pipe

        # variables for training
        self.max_episode = max_episode
        self.max_timestep = max_timestep
        self.update_timestep = update_timestep
        self.log_interval = log_interval

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
        running_reward = 0

        for i_episodes in range(1, self.max_episode+2):
            state = self.env.reset()

            if i_episodes == self.max_episode+1:
                printInfo("Max episodes reached")
                msg = MsgMaxReached(self.proc_id, True)
                self.pipe.send(msg)
                break

            for i in range(self.max_timestep):

                timestep += 1

                states.append(state)

                action, logprob = self.memory.agent_policy.act(state, False)
                state, reward, done, _ = self.env.step(action)

                actions.append(action)
                logprobs.append(logprob)
                rewards.append(reward)
                is_terminal.append(done)

                running_reward += reward

                if timestep % self.update_timestep == 0:
                    stateT, actionT, logprobT, disReturn = \
                        self.experience_to_tensor(
                            states, actions, rewards, logprobs, is_terminal)

                    self.add_experience_to_pool(stateT, actionT,
                                                logprobT, disReturn)

                    msg = MsgUpdateRequest(int(self.proc_id), True)
                    self.pipe.send(msg)
                    msg = self.pipe.recv()
                    if msg == "RENDER":
                        self.render = True
                    timestep = 0
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

            if i_episodes % self.log_interval == 0:
                running_reward = running_reward/self.log_interval
                # printInfo("sending reward msg")
                msg = MsgRewardInfo(self.proc_id, i_episodes, running_reward)
                self.pipe.send(msg)
                running_reward = 0

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
    # env_name = "Reacher-v2"
    env_name = "LunarLander-v2"
    # env_name = "CartPole-v0"
    num_agents = 4
    max_timestep = 300        # per episode the agent is allowed to take
    update_timestep = 2000    # total number of steps to take before update
    max_episode = 50000
    seed = None               # seeding the environment
    render = False
    solved_reward = 230
    log_interval = 100
    save_log_to_csv = True

    # gets the parameter about the environment
    sample_env = gym.make(env_name)
    state_dim = sample_env.observation_space.shape[0]
    action_dim = 4
    # action_dim = sample_env.action_space.n
    print("#################################")
    print(env_name)
    print("Number of Agents: {}".format(num_agents))
    print("#################################\n")
    del sample_env

    # PPO & Network Parameters
    n_latent_var = 64
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    ######################################

    ppo = PPO(state_dim, action_dim, n_latent_var,
              lr, betas, gamma, K_epochs, eps_clip)

    # TODO verify if i should pass in ppo.policy_old
    memory = Memory(num_agents, update_timestep, state_dim, ppo.policy_old)

    # starting agents and pipes
    agents = []
    pipes = []

    # tracking subprocess request status
    update_request = [False]*num_agents
    agent_completed = [False]*num_agents

    # tracking training status
    update_iteration = 0
    log_iteration = 0
    average_eps_reward = 0
    reward_record = [[None]*num_agents]

    # initialize subproceses experience
    for agent_id in range(num_agents):
        p_start, p_end = mp.Pipe()
        agent = Agent(str(agent_id), memory, p_end, env_name, max_episode,
                      max_timestep, update_timestep, log_interval, gamma)
        agent.start()
        agents.append(agent)
        pipes.append(p_start)

    # starting training loop
    while True:
        for i, conn in enumerate(pipes):
            if conn.poll():
                msg = conn.recv()

                # parsing information recieved from subprocess

                # if agent reached maximum training episode limit
                if type(msg).__name__ == "MsgMaxReached":
                    agent_completed[i] = True
                # if agent is waiting for network update
                elif type(msg).__name__ == "MsgUpdateRequest":
                    update_request[i] = True
                    if False not in update_request:
                        ppo.update(memory)
                        update_iteration += 1
                        update_request = [False]*num_agents
                        msg = update_iteration
                        # send to signal subprocesses to continue
                        for pipe in pipes:
                            pipe.send(msg)
                # if agent is sending over reward stats
                elif type(msg).__name__ == "MsgRewardInfo":
                    idx = int(msg.episode/log_interval)
                    if len(reward_record) < idx:
                        reward_record.append([None]*num_agents)
                    reward_record[idx-1][i] = msg.reward

                    # if all agents has sent msg for this logging iteration
                    if (None not in reward_record[log_iteration]):
                        eps_reward = reward_record[log_iteration]
                        average_eps_reward = 0
                        for i in range(len(eps_reward)):
                            print("Agent {} Episode {}, Avg Reward/Episode {:.2f}"
                                  .format(i, (log_iteration+1)*log_interval,
                                          eps_reward[i]))
                            average_eps_reward += eps_reward[i]
                        print("Main: Update Iteration: {}, Avg Reward Amongst Agents: {:.2f}\n"
                              .format(update_iteration,
                                      average_eps_reward/num_agents))
                        log_iteration += 1

        if False not in agent_completed:
            print("=Training ended with Max Episodes=")
            break
        if solved_reward <= average_eps_reward/num_agents:
            print("==============SOLVED==============")
            break

    for agent in agents:
        agent.terminate()

    # saving training results
    today = date.today()
    file_name = './Parallel_PPO_{}_{}_{:.2f}_{}_{}' \
        .format(env_name, num_agents, average_eps_reward/num_agents,
                (log_iteration+1)*log_interval, today)

    # # saving trained model weights
    torch.save(ppo.policy.state_dict(), file_name+'.pth')

    # # saving reward log to csv
    if save_log_to_csv:
        heading = []
        for i in range(num_agents):
            heading.append("Agent {}".format(i))
        reward_record.insert(0, heading)

        with open(file_name+'.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for entry in reward_record:
                wr.writerow(entry)


if __name__ == "__main__":

    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("Training Completed, {:.2f} sec elapsed".format(end-start))
