"""
File Description: process synchronization is defined as a mechanism which
ensures that two or more concurrent processes do not simultaneously execute
some particular program segment known as critical section

Having this happen will mean: the critical section could behave unpredictbly

In this piece of code, we are demonstrating the effect of lock; which deals
with this situation

https://www.geeksforgeeks.org/synchronization-pooling-processes-python/

Project: self learning python
Author: Richard Hu
Date: Nov-07-2019
"""

import multiprocessing

# function to withdraw from account


def withdraw(balance, lock):
    for _ in range(5):
        lock.acquire()
        balance.value = balance.value - 10
        print("Balance after withdraw: {}".format(balance.value))
        lock.release()


def deposit(balance, lock):
    for _ in range(5):
        lock.acquire()
        balance.value = balance.value + 20
        print("Balance after deposit: {}".format(balance.value))
        lock.release()


def perform_transaction(balance):

    # creating a lock object
    lock = multiprocessing.Lock()

    # creating new processes

    p1 = multiprocessing.Process(target=withdraw, args=(balance, lock))
    p2 = multiprocessing.Process(target=deposit, args=(balance, lock))

    p1.start()

    p2.start()
    p1.join()
    p2.join()

    print("Final Balance = {}".format(balance.value))


if __name__ == "__main__":

    # initial balance (in shared memory)
    balance = multiprocessing.Value('i', 100)
    print("Initial Balance: {}".format(balance.value))

    for _ in range(10):
        perform_transaction(balance)
