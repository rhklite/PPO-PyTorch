import gym
from multiprocessing import Process, Pipe


class EnvWorker(Process):

    def __init__(self, env_name, pipe, name=None):
        Process.__init__(self, name=name)
        self.env = gym.make(env_name)
        self.env.reset()
        self.pipe = pipe
        self.name = name
        print("Environment initialized. ", self.name)

    def run(self):
        while True:
            action = self.pipe.recv()
            _, reward, done, _ = self.env.step(action)
            observation = self.env.render(mode="rgb_array")
            self.pipe.send((observation, reward, done))
            self.env.render()
            if done:
                print("Done with an epsidode for %s" % self.name)
                self.env.reset()


class ParallelEnvironment(object):

    def __init__(self, env_name, num_env):
        assert num_env > 0, "Number of environments must be postive."
        self.num_env = num_env
        self.workers = []
        self.pipes = []
        self.sample_env = gym.make(env_name)
        for env_idx in range(num_env):
            p_start, p_end = Pipe()
            env_worker = EnvWorker(env_name, p_end, name=str(env_idx))
            env_worker.start()
            self.workers.append(env_worker)
            self.pipes.append(p_start)

    def step(self, actions):
        observations, rewards, dones = [], [], []
        for idx in range(self.num_env):
            self.pipes[idx].send(actions[idx])
        for idx in range(self.num_env):
            observation, reward, done = self.pipes[idx].recv()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
        return observations, rewards, dones

    def get_action_space(self):
        return self.sample_env.action_space

    def __del__(self):
        """
        Terminate all spawned processes.
        """
        for worker in self.workers:
            worker.terminate()
            worker.join()


# Works fine if I use Atari environment
num_envs = 4
p_env = ParallelEnvironment("LunarLander-v2", num_envs)
action_space = p_env.get_action_space()
for i in range(1000000):
    actions = [action_space.sample() for _ in range(num_envs)]
    obs, rwds, dones = p_env.step(actions)
    print(obs[0].shape)
