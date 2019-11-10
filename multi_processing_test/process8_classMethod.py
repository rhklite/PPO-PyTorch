
from multiprocessing import Pool
import time
import concurrent.futures


class C:
    def f(self):
        # print('hello %s,' % name)
        print('hello Dave')
        time.sleep(5)
        print('nice to meet you.')

    def run(self):
        # pool = Pool()
        # pool.map(self.f, ('frank', 'justin', 'osi', 'thomas'))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self.f) for _ in range(5)]


if __name__ == '__main__':
    c = C()
    c.run()
