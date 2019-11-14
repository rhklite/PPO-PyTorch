"""
File Description: testing pytorch and pipes() and process inheritance way of mp

the access test s
Project: University of Toronto - ASBLab, Rough Terrain Navigation
Author: Richard Hu
Date: Nov-12-2019
"""

import torch
import torch.multiprocessing as mp
import os

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
mp.set_start_method("spawn", True)

# you don't need lock if you work in a different part of the memory.
# you do need it if the subprocesses is trying to access the same memory block


class Memory:
    def __init__(self, num_proc, n_iter):
        self.tensor = torch.zeros(num_proc*2).to(device).share_memory_()
        self.status = torch.tensor([0]).to(device).share_memory_()
        self.n_iter = torch.tensor([0]).to(device).share_memory_()
        self.iter_max = n_iter*num_proc
        self.num_proc = num_proc


class Worker(mp.Process):
    def __init__(self, memory, proc, pipe, name=None):
        mp.Process.__init__(self, name=name)
        self.memory = memory
        self.access_test = 'not changed'
        self.pipe = pipe
        self.proc = proc

    def run(self):
        recieved = self.pipe.recv()
        position = self.memory.status.item()
        mylist = [self.proc, self.proc]
        data = torch.tensor(mylist).float().to(device)
        self.memory.tensor[position:position+2] = data
        self.memory.status.add_(2)
        self.memory.n_iter.add_(1)
        # lock.release()
        print("{} proc, process ID {}, Tensor Out: {}\n"
              .format(self.proc, os.getpid(), self.memory.tensor))
        print("test string: {}".format(self.access_test))
        print()


class ParallelWorker:
    def __init__(self, memory, num_proc):
        self.memory = memory
        self.workers = []
        self.pipes = []

        for worker_id in range(num_proc):
            print("Worker ID: {} initialized".format(worker_id))
            p_start, p_end = mp.Pipe()
            worker = Worker(memory, worker_id, p_end, name=str(worker_id))
            self.workers.append(worker)
            self.pipes.append(p_start)

    def run_parallel(self, n_iter):
        for worker in self.workers:
            print("worker {} started".format(worker.name))
            worker.start()

        for n in range(n_iter):
            self.memory.status[0] = 0
            for pipe in self.pipes:
                pipe.send("Hello subprocess\n")

        # for worker in self.workers:
        #     worker.join()

    def __del__(self):
        for worker in self.workers:
            worker.join()
            worker.terminate()


if __name__ == "__main__":
    num_proc = 4
    n_iter = 4
    memory = Memory(num_proc, n_iter)
    run = ParallelWorker(memory, num_proc)
    run.run_parallel(n_iter)

    print("main process {}, is shared {}".format(
        memory.tensor, memory.tensor.is_shared()))
