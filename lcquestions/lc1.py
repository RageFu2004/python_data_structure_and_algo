def wordbreak(s, wordDict):
    dp = {}

    def word(s, wordDict):
        if s in dp:
            return dp[s]
        if s == "":
            print('ok')
            return True
        for i in wordDict:
            if s.startswith(i):
                dp[s] = word(s[len(i):], wordDict)
                if dp[s]:
                    return True
        return False

    word(s, wordDict)
    return dp[s]

def robhouse(nums):
    color_lst = [0]*len(nums)
    dp = [0]*len(nums)
    dp[0] = nums[0]
    color_lst[0] = 1
    for i in range(len(dp)-1):
        if i == len(dp) - 2:
            if color_lst[0] == 1:
                dp[1 + i] = dp[i]
                continue

        if color_lst[i] == 0:
            dp[1+i] = dp[i] + nums[1+i]
            color_lst[1+i] = 1

        if color_lst[i] == 1:
            dp[1+i] = dp[i]
    result1 = max(dp)

    dp[0] = 0
    color_lst[0] = 0
    for i in range(len(dp) - 1):
        if i == len(dp)-2:
            if color_lst[0] == 1:
                dp[1 + i] = dp[i]
                continue

        if color_lst[i] == 0:
            dp[1 + i] = dp[i] + nums[1 + i]
            color_lst[1 + i] = 1

        if color_lst[i] == 1:
            dp[1 + i] = dp[i]

    result2 = max(dp)

    return max(result1, result2)

def coinpile(lst, k):
    dp = []
    num = len(lst[0])+1
    # len(lst[0]) = #of items in a pile
    for i in range(num):
        dp.append([None]*num)

    item_b = len(lst[0])
    item_a = len(lst[0])

    def mem_search(item_a, item_b, k):

        if dp[item_a][item_b]:
            print(766)
            return dp[item_a][item_b]

        if k == 0:
            dp[item_a][item_b] = 0
            return 0

        if item_b == 0 and item_a == 0:
            dp[item_a][item_b] = 0
            return 0

        dp[item_a][item_b] = max(mem_search(item_a-1, item_b, k-1)+lst[0][item_a-1],
                                 mem_search(item_a, item_b-1, k-1)+lst[1][item_b-1])
        return dp[item_a][item_b]

    mem_search(item_a, item_b, k)
    return dp

#print(coinpile([[3,100,1], [9, 8, 7]], 2))

def rob(nums) :
    dp1 = [0] * (len(nums) + 2)
    dp2 = [0] * (len(nums) + 2)

    for i in range(3, len(dp1)):
        dp1[i] = max(dp1[i - 1], dp1[i - 2] + nums[i - 2])

    for j in range(2, len(dp2) - 1):
        dp2[j] = max(dp2[j - 1], dp2[j - 2] + nums[j - 2])

    return max(dp1[-1], dp2[len(dp2) - 2])

def robhouseiii(root):
    for i in root:
        if i == 'null':
            i = 0
    dp = [0]*(len(root)+1)
    dp[0] = root[0]
    for i in range(1, len(dp)-1):
        daddynode = (i+1)//2 - 1
        daddy = dp[daddynode]
        daddy_daddy = dp[(daddynode+1)//2 - 1] + root[i]
        if daddy > daddy_daddy:
            dp[i] = daddy
        else:
            dp[i] = daddy_daddy
            dp[daddynode+1//2 - 1] = daddy_daddy
        print(dp)
    return dp[-2], dp

def jump(nums) :
    dp = [0] * len(nums)
    for i in range(1, len(dp)):
        index = len(dp)-1-i
        min_val = 100
        n = 1
        while n <= nums[index] and index+n < len(dp):
            if dp[index+n] < min_val:
                min_val = dp[index+n]
            n += 1
        dp[index] = min_val + 1

    return dp[0]
#print(jump([2,3,0,1,4]))

def canJump( nums):
    dp = [0] * len(nums)
    dp[-1] = 1
    for i in range(1, len(dp)):
        index = len(dp) - 1 - i
        dp[index] = 0
        if 1 in dp[index: index + nums[index]+1]:
            dp[index] = 1

    if dp[0] == 1:
        return True,dp
    else:
        return False,dp

def canReach( arr, start):
    dp = [2]*len(arr)
    for i, q in enumerate(arr):
        print(i ,q)
        if q == 0:
            dp[i] = 1

    def reach(arr, start):
        #print(start)
        #print(dp)
        if start <0 or start>len(arr)-1:
            return 0
        if dp[start]:
            if dp[start] == 'find':
                print('its o')
                return 0
        if dp[start] and dp[start]!=2:
            return dp[start]
        dp[start] = 'find'
        j8 = reach(arr, start + arr[start]) + reach(arr, start - arr[start])
        #print('res', j8)
        dp[start] = j8
        return dp[start]

    return reach(arr, start),dp

#print(canReach([4,2,3,0,3,1,2],5))

def minJumps(arr):
    met = {}
    dp = [0] * len(arr)
    met[arr[-1]] = 0
    for i in range(1, len(arr)):
        index = len(arr)-1-i
        if arr[index] not in met:
            dp[index] = dp[index+1] + 1
            met[arr[index]] = dp[index+1] +1
        else:
            if arr[index] in met:
                print(met, arr[index], met[arr[index]])
                dp[index] = min(met[arr[index]], dp[index+1])+1
                met[arr[index]] = min(dp[index], met[arr[index]])
                for j in range(index+1, len(arr)-1):
                    dp[j] = min(dp[j-1] + 1, dp[j])
                    if dp[j] < met[arr[j]]:
                        met[arr[j]] = dp[j]
        print(dp)
    return dp[0],dp

#print(minJumps([100,-23,-23,404,100,23,23,23,3,404]))

import math
#lst = [1, 4, 5, 9, 10, 12, 15, 18, 19, 20]
#c = 1
def rod(arr, cons):
    dp = [0]*len(arr)
    for i in range(len(dp)):
        max = 0
        for j in range(0, i+1):
            if j == 0:
                price = arr[i]
            else:
                price = dp[i-j] + arr[j-1] - cons
            if price > max:
                max = price

        dp[i] = max
    return dp

#print('q1', rod(lst, c))

#p = [10, 5, 10, 5]
def matrix_chain(p):
    dp = []
    for i in range(len(p)-1):
        dp.append([0]*(len(p)-1))
    print(dp)
    for i in range(0, len(dp)):
        for j in range(0, len(dp)):
            if i == j:
                dp[i][j] = 0

            else:
                min = math.inf
                for k in range(i, j):
                    price = dp[i][k] + dp[k+1][j] + p[i]*p[k+1]*p[j+1]
                    print(price)
                    if price < min:
                        min = price
                print(i, j, min)
                dp[i][j] = min
    return dp
#print(matrix_chain(p))

def matrix_chain_order(p):
    n = len(p) - 1
    m = []
    s = []
    for i in range(n):
        m.append([0]*n)
        s.append([0]*n)
    for d in range(2, n + 1):
        for i in range(n - d + 1):
            j = i + d - 1
            m[i][j] = math.inf
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k+1
    return m, s

'''p = [5, 6, 3, 7, 5, 3]
m, s = matrix_chain_order(p)
print("m-table:")
for row in m:
    print(row)
print("s-table:")
for row in s:
    print(row)
'''

def lcs_length(X, Y):
    n = len(X)
    m = len(Y)
    c = [[0] * (m+1) for i in range(n+1)]
    b = [[""] * (m+1) for i in range(n+1)]
    for i in range(1, n+1):
        c[i][0] = 0
    for j in range(m+1):
        c[0][j] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if X[i-1] == Y[j-1]:
                c[i][j] = c[i-1][j-1] + 1
                b[i][j] = "up-left"
            elif c[i-1][j] >= c[i][j-1]:
                c[i][j] = c[i-1][j]
                b[i][j] = "up"
            else:
                c[i][j] = c[i][j-1]
                b[i][j] = "left"
    return c, b
'''
X = ["B", "C", "A", "A", "B", "A"]
Y = ["A", "B", "A", "C", "B"]
c, b = lcs_length(X, Y)
print("c-table:")
print(row)
for row in b:
    print(row)
'''
def mergesort(lst, start, end):
    if end - start == 1:
        if lst[start] > lst[end]:
            return [lst[end], lst[start]]
        else:
            return [lst[start], lst[end]]
    ret = []
    a = mergesort(lst, start, (start+end)//2)
    b = mergesort(lst, (start+end)//2+1, end)
    while a or b:
        if not a:
            for i in range(len(b)):
                ret.append(b.pop(0))
            break
        if not b:
            for i in range(len(a)):
                ret.append(a.pop(0))
            break
        if a[0] > b[0]:
            ret.append(b.pop(0))
        else:
            ret.append(a.pop(0))
    return ret

def insert_sort_recur(lst):
    n = len(lst)
    if n == 2:
        return [min(lst), max(lst)]
    ret = insert_sort_recur(lst[0:n-1])
    if lst[n-1] > ret[-1]:
        ret.append(lst[n-1])
        return ret
    else:
        ret.append(lst[n-1])
        i = n-2
        temp = ret[n-1]
        while ret[i] > temp and i >= 0:
            ret[i+1] = ret[i]
            i -= 1
        ret[i+1] = temp
        return ret

#print(insert_sort_recur([2,3,1,5,7,9,10,4]))

def dnc_compare(lst, low, high):
    print(lst)
    if high - low == 1:
        return min(lst), max(lst)
    mid = (low+high)//2
    print('mid', mid)
    min1, max1 = dnc_compare(lst[low: mid+1], low, mid)
    min2, max2 = dnc_compare(lst[mid+1: high+1], mid+1, high)
    return min([min1, min2]), max([max1, max2])

#print(dnc_compare([1,2,3,4],0, 3))

def find_i(lst, low, high):
    if low > high:
        return False
    mid = (low+high)//2
    if lst[mid] == mid:
        return mid
    elif lst[mid] > mid:
        return find_i(lst, low, mid-1)
    else:
        return find_i(lst, mid+1, high)
#print(find_i([-5,0,2,5], 0, 3))

def wordbreak2(s, wordDict):
    word_store = {}
    def mem_search(s, wordDict):
        if s in word_store:
            return word_store[s]
        if s == "":
            return ['']
        mem_list = []
        for i in wordDict:
            if s.startswith(i):
                n = mem_search(s[len(i):], wordDict)
                if n:
                    for j in n:
                        if j == '':
                            mem_list.append(i)
                        else:
                            mem_list.append(i+' '+j)

        if mem_list:
            word_store[s] = mem_list
            return mem_list
        else:
            word_store[s] = []
            return []
    mem_search(s, wordDict)
    return word_store[s]
#print(wordbreak2("catsandog", ["cats","dog","sand","and","cat"]))

def topk(nums, k):
    def heapify(heap):
        index = 0
        while index <= len(heap)//2:
            if heap[index] > heap[index*2 + 1]:
                heap[index],heap[index*2+1] = heap[index*2+1], heap[index]
                index = index*2 + 1
            if heap[index] > heap[index*2 + 2]:
                heap[index], heap[index * 2 + 2] = heap[index * 2 + 2], heap[index]
                index = index*2+2

    minheap = [None]*k
    count = {}
    for i in nums:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    for j in count:
        if not minheap[0]:
            minheap[0] = j
            heapify(minheap)
        else:
            if count[j] > count[minheap[0]]:
                minheap[0] = j
                heapify(minheap)

    return minheap

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        def mergesort(lists):
            if len(lists) == 1:
                return lists
            mid = len(lists)//2
            left = mergesort(lists[0:mid])
            right = mergesort(lists[mid+1:])
            left_head = left[0]
            right_head = right[0]

            ret_list_head = ListNode(None, None)
            p = ret_list_head
            while left_head or right_head:
                if not left_head:
                    if p.val:
                        p.next = ListNode(val=right_head.val)
                        p = p.next
                    if not p.val:
                        p.val = right_head.val
                    right_head = right_head.next
                    continue
                if not right_head:
                    if p.val:
                        p.next = ListNode(val=left_head.val)
                        p = p.next
                    if not p.val:
                        p.val = left_head.val
                    left_head = left_head.next
                    continue


                if left_head.val <= right_head.val:
                    if p.val:
                        p.next = ListNode(val=left_head.val)
                        p = p.next
                    if not p.val:
                        p.val = left_head.val
                    left_head = left_head.next
                    continue
                else:
                    if p.val:
                        p.next = ListNode(val=right_head.val)
                        p = p.next
                    if not p.val:
                        p.val = right_head.val
                    right_head = right_head.next
                    continue
            return [ret_list_head]
        return mergesort(lists)

