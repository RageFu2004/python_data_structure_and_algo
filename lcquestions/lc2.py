import heapq
import math
from collections import deque
def findKthLargest(nums, k):
    def heapify(lst):
        i = 0
        while i*2+1 <= len(lst)-1:
            j = i * 2 + 1
            temp = j
            if j+1 <= len(lst) -1 and lst[j+1]<lst[j]:
                if lst[i] > lst[j+1]:
                    lst[i], lst[j+1] = lst[j+1], lst[i]
                    temp = j+1
                    i = temp
                    continue
            else:
                if lst[i] > lst[j]:
                    lst[i], lst[j] = lst[j], lst[i]
                    temp = j
            i = temp

    minheap = [-math.inf] * k
    for i in nums:
        if i > minheap[0]:
            minheap[0] = i
            heapify(minheap)
            print(minheap)
    return minheap[0]

#print(findKthLargest([3,2,1,5,6,4],2))
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
def addTwoNumbers(l1, l2) :
    takeout = 0
    l1_head = l1
    l2_head = l2
    ret_list = ListNode(Null)
    p = ret_list
    while l1_head or l2_head or takeout != 0:
        if not l1_head and not l2_head:
            p.next = ListNode(val=takeout)
            break
        if not l1_head:
            sum_ = l2_head.val + takeout
            takeout = sum_ // 10
            if p.val is not None:
                p.next = ListNode(val=sum_ % 10)
                p = p.next
            else:
                p.val = sum_ % 10
            l2_head = l2_head.next
            continue
        if not l2_head:
            sum_ = l1_head.val + takeout
            takeout = sum_ // 10
            if p.val is not None:
                p.next = ListNode(val=sum_ % 10)
                p = p.next
            else:
                p.val = sum_ % 10
            l1_head = l1_head.next
            continue

        sum_ = l1_head.val + l2_head.val + takeout
        takeout = sum_ // 10
        if p.val is not None:
            newnode = ListNode(sum_ % 10)
            p.next = newnode
            p = p.next
        else:
            p.val = sum_ % 10
        l1_head = l1_head.next
        l2_head = l2_head.next
    return ret_list

class Solution:
    def search(self, nums, target) :
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            # Check if left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Otherwise, right half is sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1

def solution(blockCount, writes, threshold):
    memory = [0]*blockCount
    ret_list = ['n']*blockCount
    dp = [0]*blockCount
    for pair in writes:
        for i in range(pair[0], pair[1]+1):
            memory[i] += 1
            if memory[i] == threshold:
                ret_list[i] = 'y'
    for j in range(1, len(ret_list)):
        if ret_list[j] == 'y':
            return

def sumofdepth(lst):
    dp_sum = [0]*len(lst)
    for i in range(1, len(dp_sum)):
        dp_sum[i] = dp_sum[(i-1)//2]+1
    return sum(dp_sum), dp_sum

#print(sumofdepth([1,2,3,4,5,6,7,8,9]))

def nqueens(n):
    record = [-1]*n
    ret_list = []
    def recur(i, right, record):
        if i == len(record):
            ret_list.append(right[::])
            return ret_list
        for j in range(len(record)):
            check = True
            if i > 0:
                for item in range(0, i):
                    if record[item] == j:
                        check = False
                    if (record[item] - j)**2 == (i - item)**2:
                        check = False
                if check:
                    record[i] = j
                    right.append('.'*j+'Q'+'.'*(len(record)-j-1))
                    recur(i+1, right, record)
                    right.pop()
            else:
                record[i] = j
                right.append('.'*j+'Q'+'.'*(len(record)-j-1))
                recur(i + 1, right, record)
                right.pop()
    recur(0, [], record)
    return ret_list
#print(nqueens(4))

def uniquePathsWithObstacles(obstacleGrid):
        dp = []
        for i in range(len(obstacleGrid)):
            dp.append([0]*len(obstacleGrid[0]))
        for row in range(len(dp)):
            for col in range(len(dp[0])):
                if obstacleGrid[row][col] == 1:
                    dp[row][col] = 0
                    continue
                if row == 0:
                    dp[row][col] = 1
                    continue
                if col == 0:
                    dp[row][col] = 1
                    continue

                dp[row][col] = dp[row-1][col]+dp[row][col-1]
        return dp[-1][-1], dp

def longestPalindrome(s):
    dp = []
    max_store = [' ',0]
    for i in range(len(s)):
        dp.append([0]*len(s))
    for adder in range(len(dp)):
        for row in range(len(dp)):
            if adder+row == len(dp):
                break
            if adder+row == row:
                dp[row][row+adder] = 1
            else:
                if s[row] == s[row+adder]:
                    if row + 1 == row + adder:
                        dp[row][row + adder] = 2
                    else:

                        if dp[row+1][row+adder-1] == 0:
                            dp[row][row+adder] = 0
                        else:

                            dp[row][row+adder] = dp[row+1][row+adder-1]+2
                else:
                    dp[row][row+adder] = 0
            if dp[row][row+adder] > max_store[1]:
                max_store = [s[row:row+adder+1],dp[row][row+adder]]
        continue
    return max_store[0]
#rint(longestPalindrome(''))

def minDistance(word1, word2):

        dp = []
        for i in range(len(word1)+1):
            dp.append([0]*(len(word2)+1))
        for row in range(0, len(dp)):
            for col in range(0, len(dp[0])):
                if row == 0 or col == 0:
                    if row == 0:
                        dp[row][col] = col
                        continue
                    if col == 0:
                        dp[row][col] = row
                        continue
                if word1[row-1] == word2[col-1]:
                    dp[row][col] = dp[row-1][col-1]
                else:
                    dp[row][col] = min(dp[row-1][col], dp[row][col-1], dp[row-1][col-1])+1
        return dp[-1][-1]
#print(minDistance('intention', 'execution'))

def maxProfit(prices):
    diff_lst = []
    for i in range(1, len(prices)):
        diff_lst.append(prices[i] - prices[i - 1])
    diff_lst.append(0)
    dp = [diff_lst]
    for _ in range(2):
        dp.append([0] * len(diff_lst))

    for row in range(1, len(dp)):
        for col in range(1, len(dp[0])):
            dp[row][col] = max(dp[row - 1][col - 1], dp[row][col - 1]+diff_lst[col-1])

    return dp
#print(maxProfit([3,2,6,5,0,3]))

def maximalSquare(matrix) -> int:
    dp = []
    maxi = 0
    for _ in range(len(matrix)):
        dp.append([0] * len(matrix[0]))
    for i in range(len(dp)):
        dp[i][0] = int(matrix[i][0])
        for j in range(len(dp[0])):
            dp[0][j] = int(matrix[0][j])
            if matrix[i][j] == '0':
                continue
            if i != 0 and j != 0:
                if dp[i - 1][j] and dp[i][j - 1] and dp[i - 1][j - 1]:
                    if dp[i - 1][j] == dp[i][j - 1] and dp[i][j - 1] == dp[i - 1][j - 1]:
                        dp[i][j] = dp[i - 1][j] + 1

                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                else:
                    dp[i][j] = int(matrix[i][j])
            if dp[i][j] > maxi:
                maxi = dp[i][j]

    return maxi * maxi


def lengthOfLIS(nums):
    dp = [1]*len(nums)
    max_ = 1
    for i in range(1, len(dp)):
        maxi = 0
        for j in range(0, i):
            if nums[j]<nums[i] and dp[j]>maxi:
                maxi = dp[j]
        dp[i] = maxi+1
        if dp[i]>max_:
            max_ = dp[i]
    return dp

#print(lengthOfLIS([0,1,0,3,2,3]))

def maxEnvelopes(envelopes):
    envelopes.append([10001,10001])
    store = [0]*len(envelopes)
    def dfs(lst, i):
        if store[i]:
            return store[i]
        maxi = 1
        for j in range(0, len(lst)):
            if lst[j][0] < lst[i][0] and lst[j][1] < lst[i][1]:
                num = dfs(lst, j)+1
                if num > maxi:
                    maxi = num
        store[i] = maxi
        return store[i]
    dfs(envelopes,len(envelopes)-1)
    return store[-1]-1

def maxEnvelopes1(envelopes):
        envelopes.sort()
        dp = [1]*len(envelopes)
        for i in range(1, len(dp)):
            maxi = 0
            for j in range(0, i):
                if envelopes[j][1] < envelopes[i][1] and dp[j]>maxi and dp[j]!=1:
                    maxi = dp[j]
            dp[i] = maxi+1
        return dp, envelopes

def numTeams(rating) :
    def get(rating):
        dp = []
        for _ in range(3):
            dp.append([0] * len(rating))
        for row in range(0, 3):
            for col in range(row, len(dp[0])):
                dp[0][col] = 1
                if row != 0:
                    for k in range(0, col):
                        if rating[col] > rating[k]:
                            dp[row][col] += dp[row - 1][k]
        print(dp)
        return sum(dp[2])

    return get(rating) + get(rating[-1:-len(rating)-1:-1])
#print(numTeams([2,5,3,4,1]))

def findMaxForm(strs, m, n):
    dp = {}

    def A(curm, curn, ind):
        if (curm, curn, ind) in dp:
            return dp[(curm, curn, ind)]
        if curm < 0 or curn < 0:
            return -1
        if ind == len(strs):
            return 0
        ans1 = A(curm, curn, ind+1)
        ans2 = 1+A(curm-(strs[ind].count('0')), curn-(strs[ind].count('1')),ind+1)
        dp[(curm, curn, ind)] = max(ans1, ans2)
        return max(ans1, ans2)
    return A(m, n, 0)

def countGoodStrings(low, high, zero, one):
    dp = [0] * (high+1)
    dp[0] = 1
    for i in range(1, len(dp)):
        print('this',i)
        if i >= zero or i >= one:
            if i>= zero:
                dp[i] += dp[i-zero]
            if i >= one:
                dp[i] += dp[i-one]

        else:
            dp[i] = 0
    return sum(dp[low:high+1]),dp
#print(countGoodStrings(2,3,1,2))

def peopleAwareOfSecret(n, delay, forget):
    know = [1]*n
    new = [0]*n
    new[0]=1
    for i in range(delay, len(new)):
        if i -forget+1 < 0:
            new[i] = sum(new[0:i-delay+1])
        else:
            new[i]=sum(new[i-forget+1:i-delay+1])
    for j in range(1, len(know)):
        if j- forget <0:
            know[j] = know[j-1]+new[j]
        else:
            know[j] = know[j-1]+new[j]-new[j-forget]
    return know[-1],know, new

#print(peopleAwareOfSecret(4,1,3))

def countHousePlacements(n: int):
    dp = [1] * n
    if n > 2:
        for i in range(2, len(dp)):
            dp[i] = dp[i - 2] + dp[i - 1]
    return ((sum(dp) + 1) ** 2) % (10 ** 9 + 7)

def constrainedSubsetSum(nums, k):
    dp = [0]*len(nums)
    dp[0] = max(nums[0], 0)
    for i in range(1,len(dp)):
        if i - k < 0:
            dp[i] = max(max(dp[0:i])+nums[i], 0)
        else:
            dp[i] = max(max(dp[i-k:i])+nums[i], 0)
    if dp[-1]:
        return dp[-1],dp
    else:
        return max(nums)
#print(constrainedSubsetSum([-5266,4019,7336,-3681,-5767],2))

def maximumScore(nums, k):
    dp = []
    max_ = 0
    for i in range(k+2):
        dp.append([0]*(len(nums)-k+1))
    for row in range(len(dp)):
        for col in range(len(dp[0])):
            dp[row][0] = nums[k+1-row]
            dp[0][col] = nums[k+col-1]
    for i in range(1,len(dp)):
        for j in range(1, len(dp[0])):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1])
            if dp[i][j]*(j+i-1) > max_:
                max_ = dp[i][j]*(j+i-1)
    return max_, dp
#print(maximumScore([1,4,3,7,4,5],3))

'''def minimumSum(nums):
    min = math.inf
    for i in range(0, len(nums)):
        for j in range(i, len(nums)):
            for k in range(j, len(nums)):
                if nums[i] < nums[j] and nums[k] < nums[j]:
                    if nums[i] + nums[j] + nums[k] < min:
                        min = nums[i] + nums[j] + nums[k]
    return min
#print(minimumSum([5,4,8,7,10,2]))'''

# sliding window1: non-repetitive substring
def slide_1(string):
    left = 0
    right = 0
    max_ = 0
    while right < len(string)-1:
        if string[right+1] in string[left:right+1]:
            if right - left+1 > max_:
                max_ = right - left+1
            left = right+1
        right += 1
    return max(right-left, max_)
#print(slide_1('abcabcbb'))

def equalSubstring(s, t, maxCost):
    cost = {}
    string = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(string)):
        cost[string[i]] = i+1

    left, right = 0, 0
    max = 0
    while right < len(s):
        sum_ = 0
        for j in range(left, right+1):
            sum_ += cost[t[j]] - cost[s[j]]
        while sum_ > maxCost:
            sum_ -= cost[t[left]] - cost[s[left]]
            left += 1
            if left > right:
                break
        if right-left+1 > max:
            max = right-left+1
        right += 1
    return max
#print(equalSubstring('abcd', 'acde', 0))

def maxVowels(s, k):
    q = deque()
    max_ = 0
    for i in range(len(s)):
        while q and q[0] <= i-k:
            q.popleft()
        if s[i] in 'aeiou':
            q.append(i)
        if i >= k-1:
            if len(q) > max_:
                max_ = len(q)
    return max_

#print(maxVowels('leetcode', 3))

def largestValues(root):
    nums = []
    window = deque()
    ret = []
    def bfs(root):
        stack = [root]
        while stack:
            rt = stack.pop(0)
            if rt.left:
                stack.append(rt.left)
            if rt.right:
                stack.append(rt.right)
            if rt.val:
                nums.append(rt.val)
            else:
                nums.append(0)

    bfs(root)
    print(nums)
    for i in range(len(nums)):
        if window and int(math.log(i+1, 2)) != int(math.log(window[0]+1,2)):
            ret.append(nums[window.popleft()])
        if not window:
            window.append(i)
        else:
            if nums[i] > nums[window[0]]:
                window[0] = i
    ret.append(nums[window.popleft()])
    return ret

def numWays(steps, arrLen):
    dp = []
    for i in range(steps + 1):
        dp.append([0] * min(steps + 1, arrLen))
    dp[0][0] = 1
    for row in range(len(dp)):
        for col in range(min(len(dp[0]), len(dp) - row)):
            if row >= 1:
                dp[row][col] += dp[row - 1][col]
            if row >= 1 and col >= 1:
                dp[row][col] += dp[row - 1][col - 1]
            if row >= 1 and col < len(dp[0]) - 1:
                dp[row][col] += dp[row - 1][col + 1]

    return dp[-1][0] % 1000000007
#print(numWays(3,3))

def countSubarrays(nums, minK, maxK):

    def dfs(window, minK, maxK):
        if not window:
            return 0
        if max(window) != maxK or min(window) != minK:
            return 0
        res = window.popleft()
        result = dfs(window, minK, maxK)+1
        window.insert(0, res)
        return result
    right = 0
    window = deque()
    ret = 0
    while right < len(nums):
        window.append(nums[right])

        result = dfs(window, minK, maxK)

        ret += result


        while (min(window) < minK or max(window) > maxK):
            window.popleft()
            if len(window) == 0:
                break
        right += 1

    return ret

# print(countSubarrays([1,1,1,1],1,1))

def numTilings(n):
    dp = {}
    dp[(0, 0)] = 1
    dp[(1, 0)] = 0
    dp[(0, 1)] = 0
    def dfs(n, m, two=False):
        if (n,m) in dp:
            return dp[(n, m)]
        if (m,n) in dp and two:
            return 0
        dp[(n, m)] = 0

        if n >= 2 and m >= 1:
            dp[(n, m)] += dfs(n - 2, m - 1)
        if n >= 1 and m >= 2:
            dp[(n, m)] += dfs(n - 1, m - 2)

        if n >= 2 and m >= 0:
            dp[(n, m)] += dfs(n - 2, m, True)
        if n >= 0 and m >= 2:
            dp[(n, m)] += dfs(n, m - 2, True)

        if n == m:
            if n >= 1 and m >= 1:
                dp[(n, m)] += dfs(n - 1, m - 1, True)

        return dp[(n, m)]

    return dfs(n, n), dp

def minimumSum(nums):
    max_ = math.inf
    left_min = [0] * len(nums)
    right_min = [0] * len(nums)
    left_min[0] = nums[0]
    right_min[-1] = nums[-1]
    for i in range(1, len(nums)):
        left_min[i] = min(nums[i - 1], left_min[i - 1])
        right_min[len(nums) - 1 - i] = min(nums[len(nums) - i], right_min[len(nums) - i])
    for j in range(1, len(nums)):
        if nums[j] > left_min[j] and nums[j] > right_min[j]:
            if nums[j] + left_min[j] + right_min[j] < max_:
                max_ = nums[j] + left_min[j] + right_min[j]
    return max_, left_min, right_min

def kthGrammar(n, k):
    dict_ = {1: [1, 0], 0: [0, 1]}

    def recur(k):
        #print(k)
        print(k)
        if k == 1 or k == 0:
            return 0
        index = recur(math.floor(k+1//2))
        return dict_[index][k + 1 % 2]

    return recur(k)

#print(kthGrammar(2,2))

def numFactoredBinaryTrees(arr):
    dp = {}
    arr.sort()
    sum_ = 1
    dp[arr[0]] = 1
    for i in range(1, len(arr)):
        num = 1
        left=0
        right = i-1
        while left <= right:
            if arr[left]*arr[right] == arr[i]:
                if left == right:
                    num += dp[arr[left]]**2
                else:
                    num += 2 * dp[arr[left]]* dp[arr[right]]
                left+=1
                right-=1
            else:
                if arr[left]*arr[right] < arr[i]:
                    left += 1
                else:
                    right -= 1
        dp[arr[i]] = num
        sum_ += num
    return sum_,dp

#print(numFactoredBinaryTrees([18,

#print(int(math.sqrt(5)))

def numSquares(n):
    max_ = int(math.sqrt(n))
    dp = [0] * (n + 1)
    for i in range(1, len(dp)):
        min_ = math.inf
        for j in range(1, max_ + 1):
            if j ** 2 <= i:
                if dp[i - j ** 2] + 1 < min_:
                    min_ = dp[i - j ** 2] + 1
        dp[i] = min_
    return dp[-1], dp

#print(numSquares(12))

