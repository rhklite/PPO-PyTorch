import multiprocessing


def square_list(mylist, q):

    for num in mylist:
        q.put(num * num)


def print_queue(q):

    print("Queue elements:")
    while not q.empty():
        print(q.get())
    print("Queue is now empty!")


if __name__ == "__main__":
    mylist = [1, 2, 3, 4]
    q = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=square_list, args=(mylist, q))
    p2 = multiprocessing.Process(target=print_queue, args=(q,))

    p1.start()
    p1.join()

    p2.start()
    p2.join()
