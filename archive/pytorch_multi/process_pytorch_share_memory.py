import torch
import torch.multiprocessing as mp
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
mp.set_start_method("spawn", True)

# you don't need lock if you work in a different part of the memory.
# you do need it if the subprocesses is trying to access the same memory block


def add_tensors(tensor, proc, lock):
    # tensor.add_(1)
    # lock.acquire()
    print("{} proc, Tensor In: {}, shared: {}"
          .format(proc, tensor, tensor.is_shared()))

    for _ in range(1000000):
        tensor[proc] = proc*10+1
    # lock.release()
    print("{} proc, process ID {}, Tensor Out: {}, shared: {}"
          .format(proc, os.getpid(), tensor, tensor.is_shared()))


if __name__ == "__main__":
    num_proc = 10
    tensor = torch.zeros(num_proc).to(device)
    tensor.storage().share_memory_()
    print()
    processes = []
    lock = mp.Lock()
    for i in range(num_proc):
        p = mp.Process(target=add_tensors, args=(tensor, i, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("main process {}, is shared {}".format(
        tensor, tensor.is_shared()))
