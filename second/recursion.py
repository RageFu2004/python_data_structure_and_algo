final_lst = []
temp_lst = []
max_time = [0]
allow_in = [True]
times = [0]
# final list for the set of successful results that add up to the n
# temp list for storing the part of each way, prepared for counting the total number of ways
# max_time for recording the origin where we first start traversing
# allow_in for check whether it is the first time we run the function
# times for the times that we return to the origin


def howManyGroups(n, m):
    remain_position = n
    # special case: if m == 0, it should be considered separately
    if m == 0:
        # if remain is also 0, it is only one way
        if remain_position == 0:
            return 1
        # if remain is else than 0, it is 0 way
        else:
            return 0
    # if m > remain_position, it is the same result as they are the same
    elif m > remain_position:
        m = remain_position
    # if it is the first time running
    if allow_in[0] is True:
        # if at first n == 0, just 1 way
        if remain_position == 0:
            return 1
        # if first time running, not recursion, store the root
        max_time[0] = remain_position
        # make it to False so that during recursion, it will not be disturbed
        allow_in[0] = False
    # smallest case
    if remain_position == 0:
        # store the temporary list into final list
        lst = [i for i in temp_lst]
        final_lst.append(lst)
        # at first, sort all the temp_lst so that we can see whether there is repetition
        for item in final_lst:
            item.sort()
        # delete all the repetition
        for i in range(len(final_lst)):
            for j in range(i+1, len(final_lst)):
                if final_lst[i] == final_lst[j]:
                    final_lst.pop(j)
        return
    else:
        # regular recursion
        for i in range(1, m+1):
            # if from 1 to m, it is smaller than the remaining n, make remain -= i
            if i <= remain_position:
                # making the size of input smaller until it is smallest case
                remain_position -= i
                # store ways into temporary list
                temp_lst.append(i)
                # recursion to find the rest part of the one way, and the count
                howManyGroups(remain_position, m)
                # after one way is done, go back to the former remain_position
                remain_position += i
                # also remove the recorded node from the temp_lst
                temp_lst.remove(i)
                # end case: if we return to the root m times(m cannot exceed n as set before), we return the result
                if remain_position == max_time[0]:
                    # count the times and store
                    times[0] += 1
                    # end case
                    if times[0] == m:
                        # return the length of the non-repetitive final_lst
                        k = len(final_lst)
                        # turn all the list defined outside the function back
                        final_lst.clear()
                        allow_in[0] = True
                        times[0] = 0
                        max_time[0] = 0
                        temp_lst.clear()
                        # return the result
                        return k
        # if we cannot find an i that satisfies smaller than remain position, just return to the last node
        else:
            return
















