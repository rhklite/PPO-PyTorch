import concurrent.futures
import time
import os
from multiprocessing import Pool

start = time.perf_counter()


class sleeper(object):
    def __init__(self):
        pass

    def do_something(self, sec):
        print('Sleeping {} second...'.format(sec))
        time.sleep(sec)

        return'Done Sleeping {} seconds. Worker process ID {}'\
              .format(sec, os.getpid())

    def sleep_more(self):
        # pool = Pool()
        # pool.map(self.do_something, (1, 2, 3, 4, 5))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.do_something, (1, 2, 3, 4, 5))


def main():
    worker = sleeper()
    worker.sleep_more()


if __name__ == "__main__":
    main()

# workers = []

# for i in range(5):
#     p = sleeper(i)
#     workers.append(p.do_something())


# with concurrent.futures.ProcessPoolExecutor() as executor:
#     # results = [executor.submit(do_something, sec) for sec in secs]
#     results = executor.map(workers)

finish = time.perf_counter()

print('Finished in {} second(s)'.format(round(finish-start, 2)))
