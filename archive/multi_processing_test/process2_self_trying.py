import multiprocessing


def square_list(mylist, result, square_sum):

    for idx, num in enumerate(mylist):
        result[idx] = num * num
        print("idx: {}".format(idx))

    square_sum.value = sum(result)

    print("Result(inprocess p1): {}".format(result[:]))

    print("Sum of squares(in process p1): {}".format(square_sum.value))


if __name__ == "__main__":

    mylist = [1, 2, 3, 4]

    # array of type int, with len = 4
    result = multiprocessing.Array('i', 4)
    square_sum = multiprocessing.Value('i')

    p1 = multiprocessing.Process(
        target=square_list, args=(mylist, result, square_sum)
    )

    p1.start()
    p1.join()

    print("Result(in main program): {}".format(result[:]))

    print("Sum of squares(in main program):{}".format(square_sum.value))
