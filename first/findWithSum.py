temp_lst = []


def findSum(arr,sum):
    for i in arr:
        if i == sum:
            temp_lst.append(i)
            return temp_lst
        else:
            if i is not None:
                if i < sum:
                    temp_lst.append(i)
                    arr[arr.index(i)] = None
                    if findSum(arr, sum-i):
                        return temp_lst
                    else:
                        temp_lst.pop()
    return False

print(findSum([1,2,3,4], 10))


