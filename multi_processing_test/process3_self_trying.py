import multiprocessing


def print_records(records):

    for record in records:
        print("Name: {0}\nScore: {1}\n".format(record[0], record[1]))


def insert_record(record, records):

    records.append(record)
    print("New record added!\n")


if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        my_list = [('Sam', 10), ('Adam', 9), ('Kevin', 9)]
        records = manager.list(my_list)

        new_record = ('Richard', 24)

        p1 = multiprocessing.Process(
            target=insert_record, args=(new_record, records))
        p2 = multiprocessing.Process(target=print_records, args=(records,))

        p1.start()
        p1.join()

        p2.start()
        p2.join()