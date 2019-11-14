"""
File Description: testing pytorch shared_memory_() way of pooling information.
done by preallocating shared memory

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
    def __init__(self, size):
        self.tensor = torch.zeros(size*2).to(device).share_memory_()
        self.status = torch.tensor([0]).to(device).share_memory_()


class test:
    def __init__(self, memory):
        self.memory = memory
        self.access_test = 'not changed'

    def add_tensors(self, proc, lock):

        # tensor.add_(1)
        # lock.acquire()
        # print("{} proc, Tensor In: {}, shared: {}"
        #       .format(proc, tensor.tensor, tensor.tensor.is_shared()))

        position = self.memory.status.item()
        mylist = [proc, proc]
        data = torch.tensor(mylist).float().to(device)
        self.memory.tensor[position:position+2] = data
        # lock.acquire()
        self.memory.status.add_(2)
        # lock.release()
        print("{} proc, status {}, \nTensor Out: {}, shared: {}"
              .format(proc, self.memory.status.item(), self.memory.tensor, self.memory.tensor.is_shared()))
        print("test string: {}".format(self.access_test))

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
            p.terminate()

    def change_param(self, update):
        self.access_test = update


if __name__ == "__main__":
    num_proc = 10
    tensor = Memory(num_proc)
    run = test(tensor)
    update = "I HAVE BEEN CHANGED"
    for i in range(1):
        print("iter: {}".format(i))
        run.run_parallel(num_proc)
        tensor.status[0] = 0
        run.access_test = "I Have been changed"

    print("After processes completed: {}".format(run.access_test))

    print("main process {}, is shared {}".format(
        tensor.tensor, tensor.tensor.is_shared()))
