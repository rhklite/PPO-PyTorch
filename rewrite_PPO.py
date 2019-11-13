import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(torch.__version__)
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

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

        # Monte Carlo estimate of state rewards:
        # Its common to use discounted rewards to give higher weight to near
        # rewards recieved near than rewards received further in the future
        rewards = []
        discounted_reward = 0

        # this zip is just to check if that reward is recieved at a
        # the reverse makes this iterator start from the end.
        # this reward is just the normal discounted reward
        for reward, is_terminal in zip(reversed(memory.rewards),
                                       reversed(memory.is_terminal)):
            if is_terminal:
                # by setting discounted reward = 0; this would serve as a
                # break between episodes. because then the expected return
                # at this point would only be ther terminal reward when
                # calculating the discounted reward.
                discounted_reward = 0
            # the reward sequence from this iterator recieved is...
            # R5, R4, R3, R2... so this formulation makes sense
            discounted_reward = reward + (self.gamma*discounted_reward)
            # insert at localtion 0, since the discounted reward sequence was
            # calculated from T first to n, so by inserting each discounted
            # reward calculated into position 0 the rewards list would look
            # like [n, n+1, n+2,... T ]
            rewards.insert(0, discounted_reward)
        # print(rewards)

        # the rewards variable is type(list) of the discounted expected return.

        # regarding this reward here; each discounted reward inserted would
        # correspond to a state. and each discounted reward in the rewards list
        # is calculated with discounting starting from that state.
        # basically, if a trajectory yields [s0, s1, s2,...,sn]
        # the rewards list would look like [R0, R1, ... Rn], with each element
        # of this list being the discounted expected return at that state.
        # recall that the expected return is the discounted summation of
        # the reward trajectory

        # Normalizing the reward. But why?
        # https://datascience.stackexchange.com/a/20127
        # this is done for PRACTICAL reasons, not theoritical
        # in general the algorithm behaves better as the backpropagation does
        # not lead your network weights to extreme values
        # This way we’re always encouraging and discouraging roughly half of
        # the performed actions.
        # Mathematically you can also interpret these tricks as a way of
        # controlling the variance of the policy gradient estimator
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean())/(rewards.std()+1e-5)

        # convert the memory's lists of tensors into a single tensor

        # memory.states is a list of tensors.
        # each element is a tensor of one state, which a size 8 list, because
        # the environment observation is a size 8 vector

        # old_states is stacking the list of tensors into a single tensor that
        # now has an additional column dimension
        # before stack [tensor[], tensor[]]
        # after stack tensor[ [], [] ]
        old_states = torch.stack(memory.states).to(device).detach()

        # same thing as actions
        old_actions = torch.stack(memory.actions).to(device).detach()

        # actions and states didn't actually need to be detached, because they
        # didn't come from the actor-critic network. but logprobs needs to be
        # because log is actually produced by the actor network. If you run
        # this in debug mode, you will notice that ...
        # memory.logprobs: require_grad = True
        # for memory.action and memory.state, its false
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # when doing this for parallel agents, APPEND all the experiences of
        # each agent to the list.
        # when calculating the expected return of each agent, i think I need to
        # calculate the discounted return first, then append.
        # because otherwise, other agents might contain a PARTIAL TRAJECTORY.
        # imagine a scenario where a agent ended with a partial trajectory,
        # when append reward to that agent before calculating the discounted
        # return, this partial trajectory will not have the correct return
        # because there is no is_terminal to indicate the end of an episode

        # TODO look into how LSTM might have an affect on this
        # TODO look into if LSTM in the current DRL sim2real stuff is actually
        # helping or not
        # paper: Deep Recurrent Q-Learning for Partially Observable MDPs
        # additionally, PPO paper talks about an advantage function that works
        # well for recurrent networks
        # the A3C paper also talks about asyn method and recurrent networks

        # print(memory)
        # print(memory.states)
        # print(old_states)
        # Optimize policy for k epochs:
        # this part is some what confusing to me. The same set of experiences
        # is used to update the network K times... The paper said this is the
        # way for PPO using fixed-length trajectory segment
        # TODO look into fixed-length trajectory segment
        # according to Daniel, this is updating method is also common
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta/ pi_theta_old):
            # using exponential returns the log back to non-log version
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding the surrogate loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip)*advantages

            # see paper for this loss formulation; this loss function
            # need to be used if the policy and value network shares
            # parameters, however, i think the author of this code just used
            # this, even though the two network are not sharing parameters
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    # Hyperparameters
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var,
              lr, betas, gamma, K_epochs, eps_clip)

    print("Learning Rate: "+str(lr)+"Betas: "+str(betas))

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    # max episodes is more of a stoping condition
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminal.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            running_reward += reward
            # running_reward.append(reward)
            if render:
                env.render()
            if done:
                break

        avg_length += t

        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(),
                       './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        # stop training if avg_reward > solved_reward
        # if sum(running_reward[-10:])/len(running_reward[-10:]) > \
        #         (log_interval*solved_reward):
        #     print("Solved!")
        #     torch.save(ppo.policy.state_dict(),
        #                './PPO_{}.pth'.format(env_name))
        #     break

        # writer.add_graph(ppo.policy.action_layer,
        #                  input_to_model=torch.from_numpy(state).float())

        # logging
        # if i_episode % log_interval == 0:
        #     avg_length = int(avg_length/log_interval)
        #     current_reward = sum(
        #         running_reward[-10:])/len(running_reward[-10:])

        #     print('Episode {} \t avg length: {} \t reward: {}'.format(
        #         i_episode, avg_length, current_reward))
        #     writer.add_scalar('Average Episode Length', avg_length, i_episode)
        #     writer.add_scalar('Average Episode Reward',
        #                       current_reward, i_episode)
        #     running_reward = []
        #     avg_length = 0


if __name__ == '__main__':
    main()
