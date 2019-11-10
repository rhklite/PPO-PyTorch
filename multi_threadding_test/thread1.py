"""
File Description: understanding difference between multiprocessing
versus multithreading

a thread is a sequence of instructions within a program that can be
executed independently of other code

almost like a subset of a process

threads are like child processes that share the parent process resources
but execute independently


multiprocess allocates separate memory and resources for each process/program.
multithreading, thread belonging to the same process shares the same memory
and resources as that of the process

- Example
    - multiprocessing is like working on MS Office while VLC player is running
    - multithreading is like in MS Word, while you are writing, the
    auto-correction is performed simultaneously

https://www.geekboots.com/story/multiprocessing-vs-multithreading
https://stackoverflow.com/questions/3044580/multiprocessing-vs-threading-python

Project: self learning python
Author: Richard Hu
Date: Nov-07-2019
"""

# Python program to illustrate the concept
# of threading
# importing the threading module
import threading


def print_cube(num):
    """
    function to print cube of given num
    """
    for i in range(100):
        print("Cube: {}".format(num * num * num))


def print_square(num):
    """
    function to print square of given num
    """
    for i in range(10):
        print("Square: {}".format(num * num))


if __name__ == "__main__":
    # creating thread
    t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=print_cube, args=(10,))

    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()

    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()

    # both threads completely executed
    print("Done!")
