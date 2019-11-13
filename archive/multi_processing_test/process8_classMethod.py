
from multiprocessing import Pool
import time
import concurrent.futures


class C:
    def __init__(self):
        self.sharedlist = []


    def f(self, dummy):
        # print('hello %s,' % name)
        print('hello Dave')
        time.sleep(0)
        print('nice to meet you.')
        self.sharedlist.append(1)

        print(self.sharedlist)
        return self.sharedlist

    def run(self):
        pool = Pool()
        results = pool.map(self.f, ('frank', 'justin', 'osi', 'thomas'))

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     results = [executor.submit(self.f) for _ in range(5)]


        print(results)
        for result in results:
            print(result)
            

if __name__ == '__main__':
    c = C()
    c.run()
    print(c.sharedlist)