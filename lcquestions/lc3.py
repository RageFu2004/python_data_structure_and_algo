from collections import deque
import math
def threeSum(nums):
    nums.sort()
    left = 0
    right = len(nums) - 1
    ret = []
    while left < right:

        if left >= 1 and right <= len(nums) - 2:
            if nums[left - 1] == nums[left] and nums[right + 1] == nums[right]:
                left -= 1
                right += 1
                continue

        if nums[left] + nums[right] == 0:
            for i in range(left + 1, right):
                if nums[i] == 0:
                    ret.append([nums[left], nums[right], 0])
            right -= 1
            left += 1
            continue
        if nums[left] + nums[right] < 0:
            if nums[left] + 2 * nums[right] < 0:
                left += 1
            else:
                for i in range(left + 1, right):
                    if nums[i] + nums[left] + nums[right] == 0:
                        ret.append([nums[left], nums[right], nums[i]])
                        break
                left += 1
            continue
        if nums[left] + nums[right] > 0:
            if nums[right] + 2 * nums[left] > 0:
                right -= 1
            else:
                for i in range(left + 1, right):
                    if nums[i] + nums[left] + nums[right] == 0:
                        ret.append([nums[left], nums[right], nums[i]])
                        break
                right -= 1
    return ret
#print(threeSum([0,0,0,0]))

def minIncrementOperations(nums, k):
    window = deque()
    count = 0
    pointer = 0
    while pointer < len(nums):
        print(window)
        if not window:
            window.append(pointer)
        else:
            while window and (nums[window[0]] < nums[pointer] or pointer - window[0] > 2):
                window.popleft()
            window.append(pointer)

            if pointer >= 2:
                if nums[window[0]] < k:
                    count += k - nums[window[0]]
                    nums[window[0]] = k
        pointer += 1
    return count
#print(minIncrementOperations([2,3,0,0,2],4))

def validPartition(self, nums) -> bool:
    dp = [0] * (len(nums) + 1)
    dp[0] = 1
    for i in range(2, len(dp)):
        if nums[i - 1] == nums[i - 2]:
            dp[i] += dp[i - 2]
        if i >= 3:
            if nums[i - 1] == nums[i - 2] and nums[i - 2] == nums[i - 3]:
                dp[i] += dp[i - 3]
            if nums[i - 1] == nums[i - 2] + 1 and nums[i - 2] == nums[i - 3] + 1:
                dp[i] += dp[i - 3]
    if dp[-1]:
        return True
    else:
        return False

def findMode(self, root):
    max_ = []
    memo = {}
    q = [root]
    while q:
        if max_:
            num = memo[max_[0]]
        tempnode = q.pop(0)
        if tempnode.val not in memo:
            memo[tempnode.val] = 1
        else:
            memo[tempnode.val] += 1
        if not max_:
            max_.append(tempnode.val)
        else:
            if memo[tempnode.val] >= num:
                if memo[tempnode.val] > num:
                    max_ = [tempnode.val]
                else:
                    if tempnode.val not in max_:
                        max_.append(tempnode.val)

        if tempnode.left:
            q.append(tempnode.left)
        if tempnode.right:
            q.append(tempnode.right)
    return max_

def minIncrementOperations2(self, nums, k):

    n = len(nums)

    dp = [0] * n
    dp[0] = max(0, k - nums[0])
    dp[1] = max(0, k - nums[1])
    dp[2] = max(0, k - nums[2])

    for i in range(3, n):
        dp[i] = min(dp[i - 3: i]) + max(0, k - nums[i])

    return min(dp[n - 3:])

def minSum(self, nums1, nums2) -> int:
    if 0 not in nums1 and 0 not in nums2:
        if sum(nums1) == sum(nums2):
            return sum(nums1)
        else:
            return -1
    ones = [0, 0]
    for i in nums1:
        if i == 0:
            ones[0] += 1
    for j in nums2:
        if j == 0:
            ones[1] += 1
    if 0 not in ones:
        return max(sum(nums1) + ones[0], sum(nums2) + ones[1])
    else:
        if ones[0] == 0:
            if sum(nums1) - sum(nums2) >= ones[1]:
                return sum(nums1)
            return -1
        if ones[1] == 0:
            if sum(nums2) - sum(nums1) >= ones[0]:
                return sum(nums2)
            return -1

def threeSum(self, nums):

    n = len(nums)
    res = []
    if (not nums or n < 3):
        return []
    nums.sort()
    res = []
    for i in range(n):
        if (nums[i] > 0):
            return res
        if (i > 0 and nums[i] == nums[i - 1]):
            continue
        L = i + 1
        R = n - 1
        while (L < R):
            if (nums[i] + nums[L] + nums[R] == 0):
                res.append([nums[i], nums[L], nums[R]])
                while (L < R and nums[L] == nums[L + 1]):
                    L = L + 1
                while (L < R and nums[R] == nums[R - 1]):
                    R = R - 1
                L = L + 1
                R = R - 1
            elif (nums[i] + nums[L] + nums[R] > 0):
                R = R - 1
            else:
                L = L + 1
    return res

def lengthOfLongestSubstring(self, s):
    if not s:
        return 0
    left, right = 0, 0
    max_ = 0
    while right < len(s) - 1:
        if s[right + 1] not in s[left:right + 1]:
            right += 1
        else:
            max_ = max(max_, right - left + 1)

            while s[right + 1] in s[left:right + 1]:
                left += 1
            right += 1
    max_ = max(max_, right - left + 1)
    return max_

def countVowelPermutation(self, n):

    def dfs(n):
        if n == 1:
            return 1, 1, 1, 1, 1
        a, e, i, o, u = dfs(n - 1)
        return e + i + u, a + i, e + o, i, i + o

    a, e, i, o, u = dfs(n)
    return (a + e + i + o + u) % (1000000007)
#print(math.ceil(4.5))

def minimizeArrayValue(nums):
    dp = []
    for lenj in range(len(nums)):
        dp.append([0] * len(nums))
    print(dp)
    for adder in range(len(dp)):
        for row in range(len(dp)):
            if row + adder > len(dp) - 1:
                continue
            else:
                if adder == 0:

                    dp[row][row] = nums[row]
                    print(dp)
                else:
                    min_ = math.inf

                    for k in range(row, row + adder):
                        if dp[k + 1][row + adder] > dp[row][k]:
                            res = math.ceil((dp[k + 1][row + adder] + dp[row][k]) / 2)
                        else:
                            res = dp[row][k]
                        if res < min_:
                            min_ = res


                    dp[row][row + adder] = min_


    return dp[0][-1],dp
#print(minimizeArrayValue([3, 7, 1, 6, 8, 9]))

#print(11//2)

def minInsertions(s):
    dp = []
    for _ in range(len(s)):
        dp.append([0] * len(s))
    for row in range(len(dp)):
        for col in range(len(dp)):
            if len(dp) - row - 1 < col:
                if s[len(dp) - row - 1] == s[col]:
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    dp[row][col] = min(dp[row - 1][col], dp[row][col - 1]) + 1
    return dp[-1][-1]

def getWinner(arr, k):
    count = 0
    window = []
    while count != k:
        print(window)
        temp = arr.pop(0)
        if not window:
            window.append(temp)
        else:
            if temp > window[0]:
                arr.append(window.pop())
                window.append(temp)
                count = 1
            else:
                arr.append(temp)
                count += 1
    return window[0]

#print(getWinner([1,9,8,2,3,7,6,4,5], 7))

def findChampion(n, edges):
    clean = [i for i in range(n)]

    def dfs(num):
        print(clean, num)
        count = 0
        for i in clean:
            if i != None:
                count +=1
        if count == 1:
            return 1
        sum_ = 0
        for j in edges:
            if j[0] == num:
                clean[j[1]] = None
                sum_ += dfs(j[1])
        return sum_

    for i in clean:
        if i!=None:
            if dfs(i):
                return i
    return -1

#rint(findChampion(3,[[0,1],[1,2]]))

def maximumScoreAfterOperations(edges, values):
    def find_dickhead(root):
        dick = False
        for i in edges:
            if i[1] == root:
                dick = True
                return find_dickhead(i[0])
        if not dick:
            return root

    def dfs(root):
        exist = False
        compare = values[root]
        sum_ = 0
        for i in edges:
            if i[0] == root:
                exist = True
                son_max, son = dfs(i[1])
                compare += son
                sum_ += son_max
        if not exist:
            return values[root], 0
        return sum_ + values[root], max(sum_, compare)

    head = find_dickhead(edges[0][0])
    print(head)
    fuck, ans = dfs(head)
    return fuck, ans

#print(maximumScoreAfterOperations([[7,0],[3,1],[6,2],[4,3],[4,5],[4,6],[4,7]],[2,16,23,17,22,21,8,6]))

def findNumberOfLIS(nums):
    dp = []
    for _ in range(len(nums)):
        dp.append([1, 1])
    for i in range(1, len(dp)):
        max_ = 0
        for j in range(0, i):
            if nums[j] < nums[i]:
                if dp[j][0] >= max_:
                    if dp[j][0] > max_:
                        max_ = dp[j][0]
                        dp[i][1] = dp[j][1]
                    else:
                        dp[i][1] += dp[j][1]

        dp[i][0] = max_ + 1

    ret = 0
    fmax = 0
    for item in dp:
        if item[0] >= fmax:
            if item[0] > fmax:
                fmax = item[0]
                ret = item[1]
            else:
                ret += item[1]
    return ret, dp
#print(findNumberOfLIS([2,2,2,2,2]))

def longestPalindromeSubseq(s):
    dp = []
    for _ in range(len(s)):
        dp.append([0] * len(s))

    for row in range(len(dp)):
        for col in range(len(dp)):
            if len(s) - 1 -row == col:
                dp[row][col] =1
            if len(s) - 1 - row < col:
                if s[len(s) - 1 - row] == s[col]:
                    dp[row][col] = dp[row - 1][col - 1] + 2
                else:
                    dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
    return dp
#print(longestPalindromeSubseq('bbbab'))

def eliminateMaximum(dist, speed):

    count = 0
    while True:
        print(dist)
        all_inf = True
        leetcode_j8_small_666_jb66777 = False
        for s in dist:
            if s <= 0:
                leetcode_j8_small_666_jb66777 = True
            if s != math.inf:
                all_inf = False
        if all_inf or leetcode_j8_small_666_jb66777:
            break

        for j in range(len(dist)):
            dist[j] -= speed[j]
        min_ = 0
        for num in range(len(dist)):
            if dist[num] < dist[min_]:
                min_ = num
        dist[min_] = math.inf
        count += 1
    return count
#print(eliminateMaximum([46,33,44,42,46,36,7,36,31,47,38,42,43,48,48,25,28,44,49,47,29,32,30,6,42,9,39,48,22,26,50,34,40,22,10,45,7,43,24,18,40,44,17,39,36],[1,2,1,3,1,1,1,1,1,1,1,1,1,1,7,1,1,3,2,2,2,1,2,1,1,1,1,1,1,1,1,6,1,1,1,8,1,1,1,3,6,1,3,1,1]))

def isReachableAtTime(sx, sy, fx, fy, t):
    memo = {}
    memo[(0, 0, 0)] = 1

    def dfs(i, x, y):
        print(i,x,y)
        if i == 0 and (x != 0 or y != 0):
            return 0
        if i < 0:
            return 0
        if (i, x, y) in memo:
            return memo[(i, x, y)]
        sum_ = 0
        for k in [-1, 1, 0]:
            for j in [-1, 1, 0]:
                if k == 0 and j == 0:
                    continue
                else:
                    sum_ += dfs(i - 1, x + k, y + j)

        memo[(i, x, y)] = sum_
        return sum_

    if dfs(t, abs(fx - sx), abs(fy - sy)):
        return True,memo
    return False,memo

#print(isReachableAtTime(2,4,7,7,6))

def maxArea(height):
    left = 0
    right = len(height) - 1
    max_ = 0
    while left < right:
        print(left, right)
        max_ = max(max_, min(height[left], height[right]) * (right - left))
        if height[left] == height[right]:
            left += 1
            right -= 1
        if height[left] > height[right]:
            while height[right - 1] < height[right]:
                right -= 1
            right -= 1
        if height[left] < height[right]:
            while height[left + 1] < height[left]:
                left += 1
            left += 1
    return max_
#print(maxArea([1,3,2,5,25,24,5]))

def twoSum(numbers, target):
    left = 0
    right = len(numbers) - 1
    while left < right:
        sum_ = numbers[left] + numbers[right]
        if sum_ == target:
            return [left + 1, right + 1]
        if sum_ > target:
            right -= 1
        else:
            left += 1


class Graph:

    def __init__(self, n, edges):
        self.edges = edges

    def addEdge(self, edge):
        self.edges.append(edge)

    def shortestPath(self, node1, node2):
        memo = {}
        memo[node1] = 0
        visited = [node2]
        def dfs(root):
            print(root,memo)
            if root in memo:
                return memo[root]
            min_ = math.inf
            for i in self.edges:
                if i[1] == root:
                    if i[0] not in visited:
                        visited.append(i[0])
                        min_ = min(min_, dfs(i[0]) + i[2])
            memo[root] = min_
            return min_

        dfs(node2)
        if memo[node2] != math.inf:
            return memo[node2]
        else:
            return -1

# Your Graph object will be instantiated and called as such:
# obj = Graph(n, edges)
# obj.addEdge(edge)
# param_2 = obj.shortestPath(node1,node2)

#graph1 = Graph(13,[[7,2,131570],[9,4,622890],[9,1,812365],[1,3,399349],[10,2,407736],[6,7,880509],[1,4,289656],[8,0,802664],[6,4,826732],[10,3,567982],[5,6,434340],[4,7,833968],[12,1,578047],[8,5,739814],[10,9,648073],[1,6,679167],[3,6,933017],[0,10,399226],[1,11,915959],[0,12,393037],[11,5,811057],[6,2,100832],[5,1,731872],[3,8,741455],[2,9,835397],[7,0,516610],[11,8,680504],[3,11,455056],[1,0,252721]])
#graph1.shortestPath(9,3)

def numBusesToDestination(routes, source, target):
    memo = {}
    memo[target] = 0
    visited = []

    def dfs(src):
        if src in memo:
            return memo[src]
        min_ = math.inf
        final_min = math.inf
        in_ = False
        for i in routes:
            if src in i:
                if i not in visited:
                    #print(src, i)
                    visited.append(i)
                    in_ = True

                    for j in i:
                        if j != src:
                            # print(src, j)
                            print(j, visited, 'now', i)
                            min_ = min(min_, dfs(j) + 1)
                    visited.remove(i)
                    print('after remove', visited)
            final_min = min(final_min, min_)

        memo[src] = final_min
        return final_min

    ret = dfs(source)
    if ret != math.inf:
        return ret, memo
    else:
        return -1
#print(numBusesToDestination([[0,1,6,16,22,23],[14,15,24,32],[4,10,12,20,24,28,33],[1,10,11,19,27,33],[11,23,25,28],[15,20,21,23,29],[29]],4,21))
#print(numBusesToDestination([[1,9,12,20,23,24,35,38],[10,21,24,31,32,34,37,38,43],[10,19,28,37],[8],[14,19],[11,17,23,31,41,43,44],[21,26,29,33],[5,11,33,41],[4,5,8,9,24,44]],37, 28))

def maxFrequency(nums, k):
    print(len(nums))
    nums.sort()
    dp = [0] * len(nums)
    count = 1
    for i in range(1, len(dp)):
        dp[i] = dp[i - 1] + i * (nums[i] - nums[i - 1])
        if dp[i] <= k:
            count += 1
        else:
            break

    return count
#print(maxFrequency([9930,9923,9983,9997,9934,9952,9945,9914,9985,9982,9970,9932,9985,9902,9975,9990,9922,9990,9994,9937,9996,9964,9943,9963,9911,9925,9935,9945,9933,9916,9930,9938,10000,9916,9911,9959,9957,9907,9913,9916,9993,9930,9975,9924,9988,9923,9910,9925,9977,9981,9927,9930,9927,9925,9923,9904,9928,9928,9986,9903,9985,9954,9938,9911,9952,9974,9926,9920,9972,9983,9973,9917,9995,9973,9977,9947,9936,9975,9954,9932,9964,9972,9935,9946,9966], 3056))

def reductionOperations(nums):
    dict_ = {}
    for i in nums:
        if i in dict_:
            dict_[i] += 1
        else:
            dict_[i] = 1
    n = len(nums)
    count = 0
    num1 = list(set(nums))
    print(nums)
    num1.sort()
    print(num1)
    for i in num1:
        n -= dict_[i]
        count += n
    return count
#print(reductionOperations([1,1,2,2,3]))

def numberOfWays(corridor):
    all_ = True
    pt = 0
    window = []
    count = 1
    time = False
    cont = 1
    while pt < len(corridor):
        if len(window) == 2:
            time = True
            window = []
        if corridor[pt] == 'S':
            all_ = False
            if time:
                count *= cont
                cont = 1
                time = False
            window.append(1)
        if corridor[pt] == 'P':
            if time:
                cont += 1
        pt += 1

    if len(window) == 1 or all_:
        return 0
    else:
        return count % (10 ** 9 + 7)

def knightDialer(n):
    count = [1] * 10
    for i in range(n - 1):
        new_count = [0] * 10
        new_count[0] = count[4] + count[6]
        new_count[1] = count[6] + count[8]
        new_count[2] = count[7] + count[9]
        new_count[3] = count[4] + count[8]
        new_count[4] = count[0] + count[3] + count[9]
        new_count[5] = 0
        new_count[6] = count[0] + count[1] + count[7]
        new_count[7] = count[2] + count[6]
        new_count[8] = count[1] + count[3]
        new_count[9] = count[2] + count[4]
        count = new_count
    return sum(count) % (10 ** 9 + 7)

def getSumAbsoluteDifferences(nums):
    dp1 = [0] * len(nums)
    dp2 = [0] * len(nums)
    for i in range(1, len(dp1)):
        dp1[i] = dp1[i - 1] + abs(nums[i] - nums[i - 1]) * i
        dp2[len(dp1) - 1 - i] = dp2[len(dp1) - i] + abs(nums[len(dp1) - 1 - i] - nums[len(dp1) - i]) * i
    for j in range(len(dp1)):
        dp2[j] += dp1[j]
    return dp2

def hammingWeight(n):
    count = 0
    for i in str(n):
        if i == '1':
            count += 1
    return count
#print(hammingWeight(00000000000000000000000000001011))