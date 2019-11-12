import torch
import torch.multiprocessing as mp
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mp.set_start_method("spawn", True)


def add_tensors(tensor, i):
    # tensor.add_(1)
    tensor[i] = i*10
    print("process ID {}, Tensor: {}, shared: {}".format(
        os.getpid(), tensor, tensor.is_shared()))


if __name__ == "__main__":
    tensor = torch.zeros(5).to(device)
    tensor.storage().share_memory_()
    print()
    processes = []
    for i in range(5):
        p = mp.Process(target=add_tensors, args=(tensor, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("main process {}, is shared {}".format(
        tensor, tensor.is_shared()))
