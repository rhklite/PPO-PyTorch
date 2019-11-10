import multiprocessing
import os
import numpy as np


def square(n):
    print("Worker process ID for {0}: {1}".format(n, os.getpid()))
    return (n*n)


if __name__ == "__main__":

    mylist = np.arange(150)
    # creating a pool, spawning 2 processes, and assign at most 3 task per
    # process
    p = multiprocessing.Pool(processes=2, maxtasksperchild=3)

    result = p.map(square, mylist)
    print(result)
