"""
You may use the following codes to implement your sorting algorithm
    - https://www.geeksforgeeks.org/sorting-algorithms-in-python/
    - https://www.tutorialspoint.com/python_data_structure/python_sorting_algorithms.htm
    - https://stackabuse.com/sorting-algorithms-in-python/
Be sure to comment your code.
"""
import time
import csv, sys, ast, time, json, math
csv.field_size_limit(1000000000)


def main():
    with open("data.csv") as data:
        dt_lst = data.read().split()
    with open("queries.json", 'r') as f:
        search_queries = json.load(f)
    result = open("output.csv", "w")
    write = csv.writer(result)
    write.writerow(["Name", "Time", "Output"])
    dt_lst.pop(0)
    prepare_dist = []
    for i in dt_lst:
        key, value = i.split(",")
        prepare_dist.append([key, value])

    # linear search
    final_lst = []
    time_begin = time.time()
    temp0 = [i for i in prepare_dist]
    for j in search_queries:
        find = True
        for i in temp0:
            if i[1] == j and find:
                final_lst.append(i)
                find = False
    time_end = time.time()
    write.writerow(["Linear Search", time_end - time_begin, final_lst])

    # bubble sort
    temp1 = [i for i in prepare_dist]
    diff_time = test_bubble(temp1)
    write.writerow(["Bubble Sort", str(diff_time), temp1])

    # Other sort
    temp2 = [k for k in prepare_dist]
    differ = test_insertion(temp2)
    write.writerow(["Other Sort", differ, temp2])

    # Other search
    fin_lst = []

    b_time = time.time()
    for j in search_queries:
        find = True
        while find:
            fin_lst.append(temp2[binary_search(temp2, j, len(temp2)-1)])
            find = False
    c_time = time.time()
    write.writerow(["Other Search", c_time - b_time, fin_lst])


def bubble_sort(lst):
    while True:
        swap = False
        for i in range(0, len(lst)-1):
            if lst[i][1] > lst[i+1][1]:
                lst[i], lst[i+1] = lst[i+1], lst[i]
                swap = True
        if swap is False:
            break


def test_bubble(lst):
    time1 = time.time()
    bubble_sort(lst)
    time2 = time.time()
    return time2 - time1


def insertion_sort(lst):
    for i in range(1, len(lst)):
        save = lst[i][1]
        j = i
        while j > 0 and lst[j - 1][1] > save:
            lst[j], lst[j-1] = lst[j - 1], lst[j]
            j -= 1
        lst[j][1] = save


def test_insertion(lst):
    time_b = time.time()
    insertion_sort(lst)
    time_c = time.time()
    return time_c - time_b


def binary_search(lst, val, high, low=0):
    if val > lst[high][1]:
        return False
    mid = (high + low)//2
    if high == low + 1:
        if lst[high][1] == val:
            return high
        if lst[low][1] == val:
            return low
    if lst[mid][1] == val:
        return mid
    else:
        if lst[mid][1] > val:
            return binary_search(lst, val, mid, low)
        elif lst[mid][1] < val:
            return binary_search(lst, val, high, mid)


class TestSortAndSearch():
    def setUp(self):
        self.data = {}
        with open('data.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data[row['Date']] = row['Price']

        with open("queries.json",'r') as f:
            self.queries = json.load(f)

        main()
        self.output = {}
        with open('output.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.output[row['Name']] = {'Time': float(row['Time']),
                                            'Output': ast.literal_eval(row['Output'])}


    def test_case1(self):
        """03 Sort and Search: Checking linear search"""
        print(self.output['Linear Search']['Time']>1.)
        print(self.output['Linear Search']['Time']<15.)
        for i,(date,price) in enumerate(self.output['Linear Search']['Output']):
            print(self.data[date] == price)
            print(self.queries[i] == price)
        print(len(self.output['Linear Search']['Output']) == len(self.queries))
        print("Second test case (sort and search): Completed")
tester = TestSortAndSearch()
tester.setUp()
tester.test_case1()








