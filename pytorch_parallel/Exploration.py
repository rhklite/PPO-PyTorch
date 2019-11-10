import os
import queue
import torch
import random
import cProfile
import numpy as np
import ConfigureEnv
import TensorConfig
import threading
import torch.multiprocessing as mp
from dqn_utils import ReplayBuffer
from torch.autograd import Variable
from dqn_utils import get_wrapper_by_name

#
# When to stop
def stopping_criterion(env):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps()

class EpsilonGreedy(object):

    def __init__(self, exploreSched, tensorCfg, replay, env, model, maxSteps):
        self.toTensorImg, self.toTensor, self.use_cuda = tensorCfg.getConfig()
        self.replay_buffer = replay
        self.env = env
        self.model = model
        self.lastObs = env.reset()
        self.nAct = env.action_space.n
        self.exploreSched = exploreSched
        self.nSteps = 0
        self.maxSteps = maxSteps
        # if self.use_cuda:
        #     model.cuda()

    def explore_nobuffer(self, t):
        self.nSteps += 1
        #
        # Epsilon greedy exploration.
        action = None
        if random.random() < self.exploreSched.value(self.nSteps):
            action = np.random.randint(0, self.nAct, dtype=np.int64)
        else:
            obs = self.toTensorImg(np.expand_dims(self.replay_buffer.encode_recent_observation(), axis=0))
            #
            # Forward through network.
            mode = self.model.training
            self.model.eval()
            _, action = self.model(Variable(obs, volatile=True)).max(1)
            self.model.train(mode)
            # _, action = targetQ_func(Variable(obs, volatile=True)).max(1)
            action = action.data.cpu().numpy().astype(np.int64)
        self.lastObs, reward, done, info = self.env.step(action)
        #
        # Step and save transition.
        if done:
            self.lastObs = self.env.reset()
        return (self.lastObs, reward, done, info, action)

    def explore(self, t, *kwargs):
        #
        # Store the latest frame in the replay buffer.
        storeIndex = self.replay_buffer.store_frame(self.lastObs)
        self.lastObs, reward, done, info, action = self.explore_nobuffer(t)
        self.replay_buffer.store_effect(storeIndex, action, reward, done)

    def can_sample(self, batchSize):
        return self.replay_buffer.can_sample(batchSize)

    def sample(self, batchSize):
        return self.replay_buffer.sample(batchSize)

    def epsilon(self):
        return self.exploreSched.value(self.nSteps)

    def shouldStop(self):
        return stopping_criterion(self.env) >= self.maxSteps

    def numSteps(self):
        return self.nSteps

    def stepSize(self):
        return 1

    def getRewards(self):
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        rews = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        ret = -float('nan')
        if len(rews) > 100:
            ret = np.mean(rews[-100:])
        return ret

    def getNumEps(self):
        return len(get_wrapper_by_name(self.env, "Monitor").get_episode_rewards())


class ExploreParallelCfg(object):
    numEnv = 5
    model = None
    exploreSched = None
    stackFrameLen = 4
    numFramesInBuffer = 1
    maxSteps = 4e7
    sampleLatest = False

class ExploreProcess(mp.Process):

    def __init__(self, coms, cfg, seed, procId, actionVec, barrier):
        super(ExploreProcess, self).__init__()
        self.com = coms
        self.model = cfg.model
        self.seed = seed
        self.procId = procId
        self.lastObs = None
        self.cfg = cfg
        self.env = ConfigureEnv.configureEnv(self.seed, 'dqn_' + str(procId))
        # self.replay_buffer = ReplayBuffer(self.cfg.stackFrameLen, self.cfg.stackFrameLen)
        frameSize = self.env.observation_space.shape
        self.retFrame = torch.ByteTensor(frameSize[0], frameSize[1], frameSize[2])
        self.reward = torch.FloatTensor(1)
        self.done = torch.ByteTensor(1)
        self.action = torch.ByteTensor(1)
        self.meanRewards = torch.FloatTensor(1)
        self.nEps = torch.LongTensor(1)
        self.retFrame.storage().share_memory_()
        self.reward.storage().share_memory_()
        self.done.storage().share_memory_()
        self.action.storage().share_memory_()
        self.meanRewards.storage().share_memory_()
        self.nEps.storage().share_memory_()
        #
        # Shared memory to transfer the action commands.
        self.actionVec = actionVec
        self.stor = self.actionVec.storage()
        self.barrier = barrier
        print('Initialized process ', procId)

    def run(self):
        # self.explore()
        cProfile.runctx('self.explore()', globals(), locals(), 'profile-%d.perf'%self.procId)

    def explore(self):
        print('Process: %d has PID: %d'%(self.procId, os.getpid()))
        #
        # For the first run, just setup a random action.
        self.lastObs = self.env.reset()
        obs, reward, done, info = self.env.step(0)
        self.retFrame.copy_(torch.from_numpy(obs))
        # self.com.send(( reward, done, 0, 0, 0))
        self.lastObs = obs
        self.retFrame.copy_(torch.from_numpy(obs))
        self.reward.copy_(torch.from_numpy(np.atleast_1d(reward)))
        self.done.copy_(torch.from_numpy(np.atleast_1d(done).astype(np.uint8)))
        self.action.copy_(torch.from_numpy(np.atleast_1d(0)))
        self.meanRewards.copy_(torch.from_numpy(np.atleast_1d(-float('nan'))))
        self.nEps.copy_(torch.from_numpy(np.atleast_1d(0)))
        #
        # Notify that remembory is ready.
        self.barrier.wait()
        # self.com.send(0)
        minEp = 100 // self.cfg.numEnv
        #
        # Loop and do work.
        while True:
            #
            # Wait for actions.
            step = self.com.recv()
            action = self.actionVec.clone().numpy().astype(np.int64)[self.procId]
            obs, reward, done, info = self.env.step(action)
            #
            # Step and save transition.
            if done:
                obs = self.env.reset()
            #
            # Store effects.
            lastRew = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
            mean_episode_reward = -float('nan')
            if (len(lastRew) > minEp):
                mean_episode_reward = np.mean(lastRew[-minEp:])
            # self.com.send(( reward, done, action, mean_episode_reward, len(lastRew)))
            self.lastObs = obs
            self.retFrame.copy_(torch.from_numpy(self.lastObs))
            self.reward.copy_(torch.from_numpy(np.atleast_1d(reward)))
            self.done.copy_(torch.from_numpy(np.atleast_1d(done).astype(np.uint8)))
            self.action.copy_(torch.from_numpy(np.atleast_1d(action)))
            self.meanRewards.copy_(torch.from_numpy(np.atleast_1d(mean_episode_reward)))
            self.nEps.copy_(torch.from_numpy(np.atleast_1d(len(lastRew))))
            #
            # Notify that remembory is ready.
            self.barrier.wait()
            # self.com.send(0)

class ParallelExplorer(object):

    def __init__(self, cfg):
        super(ParallelExplorer, self).__init__()
        #
        # This must be set in the main.
        # mp.set_start_method('forkserver')
        self.processes = []
        self.comms = []
        self.followup = []
        self.replayBuffers = []
        self.curThread = 0
        self.nThreads = cfg.numEnv
        self.meanRewards = [-float('nan')] * self.nThreads
        self.numEps = [0] * self.nThreads
        self.nInBuffers = 0
        self.totSteps = 0
        self.maxBuffers = cfg.numFramesPerBuffer
        self.exploreSched = cfg.exploreSched
        self.model = cfg.model
        self.actionVec = torch.LongTensor(self.nThreads).zero_()
        self.actionVec.storage().share_memory_()
        self.threads = np.atleast_1d(np.arange(self.nThreads, dtype=np.int64))
        self.toTensorImg, self.toTensor, self.use_cuda = TensorConfig.getTensorConfiguration()
        self.cfg = cfg
        self.barrier = mp.Barrier(self.nThreads + 1)
        #
        # How to sample.
        self.sampleFn = self._sampleRandom
        if cfg.sampleLatest:
            self.sampleFn = self._sampleLatest
        #
        # Sample from all threads.
        for idx in range(self.nThreads):
            print('Exploration: Actually set the seed properly.')
            sendP, subpipe = mp.Pipe()
            explorer = ExploreProcess(subpipe, cfg, idx, idx, self.actionVec, self.barrier)
            explorer.daemon = True
            explorer.start()
            self.processes.append(explorer)
            self.comms.append(sendP)
            self.replayBuffers.append(ReplayBuffer(cfg.numFramesPerBuffer, cfg.stackFrameLen))
            self.followup.append(idx)
        self.nAct = self.processes[0].env.action_space.n
        self.imshape = self.processes[0].env.observation_space.shape
        print('Parent PID: %d'%os.getpid())

    def __del__(self):
        for proc in self.processes:
            proc.terminate()
            proc.join()

    def recv(self):
        #
        # Gather the responses from each.
        self.barrier.wait()
        for pipeIdx in self.followup:
            # ret = self.comms[pipeIdx].recv()
            # reward, done, action, meanReward, nEp = ret
            reward = self.processes[pipeIdx].reward.clone().numpy()
            done = self.processes[pipeIdx].done.clone().numpy().astype(np.bool)
            action = self.processes[pipeIdx].action.clone().numpy()
            meanReward = self.processes[pipeIdx].meanRewards.clone().numpy()
            nEp = self.processes[pipeIdx].nEps.clone().numpy()
            self.meanRewards[pipeIdx] = meanReward
            self.numEps[pipeIdx] = nEp
            storeIndex = self.replayBuffers[pipeIdx].store_frame(self.processes[pipeIdx].retFrame.clone().numpy())
            self.replayBuffers[pipeIdx].store_effect(storeIndex, action, reward, done)
        #
        # We have finished following up.
        self.followup = []

    def send(self, nStep):
        #
        # Keep track of the effective number of steps.
        curStep = self.totSteps
        self.totSteps += nStep
        #
        # Select each of the threads to use.
        thSelect = torch.from_numpy(np.random.choice(self.threads, self.nThreads, replace=False))
        exploration = np.atleast_1d(torch.from_numpy(np.random.uniform(size=self.nThreads)))
        randomActions = torch.from_numpy(np.random.randint(0, self.nAct, size=self.nThreads, dtype=np.int64))
        self.actionVec.copy_(randomActions)
        runNetIdx = np.atleast_1d(np.atleast_1d(self.threads[thSelect])[exploration > self.exploreSched.value(curStep)])
        obsList = []
        #
        # Ensure that we actually even want to do anything.
        if runNetIdx.shape[0] > 0:
            #
            # Build the batch of images.
            preAllocated = np.empty((runNetIdx.shape[0], self.imshape[0], self.imshape[1], self.cfg.stackFrameLen), dtype=np.uint8)
            for allocatedIdx, netIdx in enumerate(runNetIdx):
                preAllocated[allocatedIdx, ...] = self.replayBuffers[netIdx].encode_recent_observation()
                # obsList.append(self.replayBuffers[idx].encode_recent_observation())
            #
            # Forward through the network.
            obsStack = Variable(self.toTensorImg(preAllocated), volatile=True)
            mode = self.model.training
            self.model.eval()
            _, actions = self.model(obsStack).max(1)
            self.model.train(mode)
            self.actionVec.scatter_(0, torch.from_numpy(runNetIdx), actions.data.cpu())
        #
        # Notify all workers.
        for idx in thSelect:
            self.comms[idx].send(curStep)
            self.followup.append(idx)

    def explore(self, nStep):
        #
        # Can only do at most nThreads steps at once.
        # assert nStep <= self.nThreads
        self.recv()
        self.send(nStep)

    def close(self):
        for proc in self.processes:
            proc.terminate()
            proc.join()

    def can_sample(self, batchSize):
        #
        # Ensure that all can sample.
        ret = True
        for buf in self.replayBuffers:
            ret = ret and buf.can_sample(batchSize // self.nThreads)
        return ret

    def _sampleRandom(self, threadIdx, n):
        return self.replayBuffers[threadIdx].sample(threadBatch)

    def _sampleLatest(self, threadIdx, n):
        return self.replayBuffers[threadIdx].sample_latest()

    def sample(self, batchSize):
        bufferSamples = batchSize // self.nThreads
        extra = batchSize - self.nThreads * bufferSamples
        extraBuff = np.zeros(self.nThreads, dtype=np.int8)
        addList = np.random.choice(self.nThreads, replace=False)
        extraBuff[addList] = 1
        samplelist = []
        for threadIdx in range(self.nThreads):
            threadBatch = bufferSamples + extraBuff[threadIdx]
            samplelist.append(self.sampleFn(threadIdx, threadBatch))
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = zip(*samplelist)
        obs_batch = np.concatenate(obs_batch)
        act_batch = np.concatenate(act_batch)
        rew_batch = np.concatenate(rew_batch)
        next_obs_batch = np.concatenate(next_obs_batch)
        done_mask = np.concatenate(done_mask)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def epsilon(self):
        return self.exploreSched.value(self.totSteps)

    def shouldStop(self):
        return stopping_criterion(self.processes[0].env) >= self.cfg.maxSteps

    def numSteps(self):
        return self.totSteps

    def stepSize(self):
        return self.nThreads

    def getRewards(self):
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        # TODO: make sure this is correct when there are no rewards yet.
        return np.mean(np.array(self.meanRewards))

    def getNumEps(self):
        return np.sum(np.array(self.numEps))


