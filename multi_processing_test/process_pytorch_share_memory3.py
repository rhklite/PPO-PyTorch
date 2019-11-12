"""
File Description: testing pytorch shared_memory_() way of pooling information.
done by preallocating shared memory
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
    def __init__(self, size):
        self.tensor = torch.zeros(size*2).to(device).share_memory_()
        self.status = torch.tensor(0).to(device).share_memory_()


class test:
    def __init__(self, memory):
        self.memory = memory

    def add_tensors(self, proc, lock):
        # tensor.add_(1)
        # lock.acquire()
        # print("{} proc, Tensor In: {}, shared: {}"
        #       .format(proc, tensor.tensor, tensor.tensor.is_shared()))

        position = self.memory.status.item()
        print(position)
        mylist = []
        mylist.append(proc)
        mylist.append(proc)
        data = torch.tensor(mylist).float().to(device)
        self.memory.tensor[position:position+2] = data
        self.memory.status.add_(2)
        # lock.release()
        print("{} proc, process ID {}, \nTensor Out: {}, shared: {}"
              .format(proc, os.getpid(), self.memory.tensor, self.memory.tensor.is_shared()))

    def run_parallel(self, num_proc):

        processes = []
        lock = mp.Lock()
        p = mp.Pool()

        for i in range(1, num_proc+1):
            p = mp.Process(target=self.add_tensors,
                           args=(i, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    num_proc = 10
    tensor = Memory(num_proc)
    run = test(tensor)
    run.run_parallel(num_proc)

    print("main process {}, is shared {}".format(
        tensor.tensor, tensor.tensor.is_shared()))
