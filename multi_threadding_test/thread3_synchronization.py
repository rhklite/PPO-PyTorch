# demonstrating race condition; two threads accessing the
# same memory at the same time would result in unpredictable behaviours.
# this expected final value of x should be 200,000

import threading
import time

# global variable x
x = 0


def increment():
    global x
    x += 1


def thread_task():

    for _ in range(100000):
        increment()


def main_task():
    global x

    x = 0

    t1 = threading.Thread(target=thread_task)
    t2 = threading.Thread(target=thread_task)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    for i in range(10):
        start = time.time()
        main_task()
        end = time.time()
        print("Iteration {0}: x={1}, time = {2}".format(i, x, end-start))
