"""
本文件中存放leetcode经典题目
"""
import heapq

from datastructure import *


def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    """
    2 两数相加。给定两个非空链表，它们都以逆序的方式存储一个非负整数。以同样的方式返回两数之和\n
    关键词：链表 进位
    """
    result = ListNode()
    head = result

    c = 0
    while l1 is not None and l2 is not None:
        result.next = ListNode(0, None)
        result = result.next

        s = l1.val + l2.val + c
        result.val = s % 10
        c = s // 10

        l1 = l1.next
        l2 = l2.next

    l = l1 if l2 is None else l2
    while l is not None:
        result.next = ListNode()
        result = result.next
        s = l.val + c
        result.val = s % 10
        c = s // 10
        l = l.next

    if c:
        result.next = ListNode(c)

    return head.next


def lengthOfLongestSubstring(s: str) -> int:
    """
    3 求最长无重复子串

    关键词：划窗 集合
    """
    if len(s) < 2:
        return len(s)

    se = {s[0]}
    max_L = 1
    i, j = 0, 1
    while j < len(s):
        if s[j] not in se:
            se.add(s[j])
            max_L = max(len(se), max_L)
        else:
            while s[i] != s[j]:
                se.remove(s[i])
                i += 1
            i += 1
        j += 1
    return max_L


def reverse(x: int) -> int:
    """
    7 反转一个整数，且避免溢出
    """
    result = 0
    sig = 1 if x >= 0 else -1
    x = abs(x)
    while x != 0:
        if result > (2**31-1 - x%10) // 10:
            return 0

        result = result * 10 + x % 10
        x = x // 10
    return sig * result


# 8 atoi
def myAtoi(s: str) -> int:
    s = s.strip()

    result = 0
    sib = 1
    for i in range(len(s)):
        ch = s[i]
        if i == 0 and ch in ['+', '-']:
            sib = 1 if ch == '+' else -1
        elif ch.isdigit():
            result = result * 10 + int(ch)
        else:
            break
    result = sib * result
    return min(max(result, -2**31), 2**31-1)


def longestCommonPrefix(strs: list) -> str:
    if not strs:
        return ""

    prefix = strs.pop(0)
    while len(strs) > 0:
        s = strs.pop(0)
        prefix_ = []
        for i in range(min(len(prefix), len(s))):
            if prefix[i] == s[i]:
                prefix_.append(prefix[i])
            else:
                break
        prefix = "".join(prefix_)
    return prefix


def threeSum(nums: list):
    """"
    15 三数之和，给定整数数组nums，给出所有和为0的不重复的三元组

    思路：排序数组，依次选定一个值，将其作为target，双指针求二元组

    注意：去除重复项
    """
    result = []
    nums.sort()

    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        target = -1 * nums[i]
        k = len(nums) - 1
        for j in range(i+1, len(nums)-1):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue

            while k > j and nums[j] + nums[k] > target:
                k = k - 1

            if k == j:
                break

            if nums[j] + nums[k] == target:
                result.append([nums[i], nums[j], nums[k]])

    return result


def threeSumClosest(nums: list, target: int) -> int:
    """
    16. 最接近的三数之和。找到和与target最接近的三元组，返回它们的和\n
    思路：排序+双指针
    """
    result = None
    min_diff = None

    nums = sorted(nums)
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        j, k = i+1, len(nums)-1
        while j < k:
            s = nums[i] + nums[j] + nums[k]
            if s == target:
                return target
            diff = abs(s-target)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                result = s
            if s > target:
                k -= 1
            else:
                j += 1

    return result



def fourSum(nums: list, target: int) -> list:
    """
    18. 四数之和，给定整数数组nums，给出所有和为target的不重复的四元组\n
    时间复杂度是O(3)
    """
    result = []

    nums = sorted(nums)
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1, len(nums)):
            if j > i+1 and nums[j] == nums[j-1]:
                continue
            l = len(nums) - 1
            t = target - nums[i] - nums[j]
            for k in range(j+1, len(nums)):
                if k > j+1 and nums[k] == nums[k-1]:
                    continue
                while l > k and  nums[k] + nums[l] > t:
                    l -= 1

                if l == k:
                    break

                if nums[k] + nums[l] == t:
                    result.append([nums[i],nums[j],nums[k],nums[l]])
    return result


def mergeKLists(lists: list) -> ListNode:
    """
    23. 合并K个升序链表
    :param lists: 包含ListNode的数组
    :return: 合并后的升序链表
    关键词：优先队列
    """
    lists = [l for l in lists if l is not None]

    if not lists:
        return None

    heap = []
    for idx, l in enumerate(lists):
        heapq.heappush(heap, (l.val, idx))

    head = ListNode()
    p = head
    while heap:
        val, idx = heapq.heappop(heap)
        l = lists[idx]
        p.next = l
        p = p.next

        if l.next is not None:
            lists[idx] = l.next
            heapq.heappush(heap, (l.next.val, idx))

    return head.next


def reverseKGroup(head: ListNode, k: int) -> ListNode:
    """
    25. K 个一组翻转链表
    """

    def reverseList(head, tail):
        # return head, tail
        prev = None
        p = head
        end = tail.next
        while p != tail:
            nex = p.next
            p.next = prev
            prev = p
            p = nex
        tail.next = prev
        head.next = end
        return tail, head

    hair = ListNode()
    hair.next = head

    prev, tail = hair, hair
    while head:
        for i in range(k):
            tail = tail.next
            if tail is None:
                return hair.next
        head, tail = reverseList(head, tail)
        prev.next = head
        prev = tail
        head = prev.next
    return hair.next

# 26
def removeDuplicates(nums):
    # 给的是有序数组
    i = 0
    for e in nums:
        if i < 1 or e != nums[i - 1]:
            nums[i] = e
            i += 1
    return i


# 28 用滚动哈希实现strStr
def strStr(haystack: str, needle: str) -> int:
    if len(needle) > len(haystack):
        return -1

    def ch2int(ch):
        return ord(ch) - 97

    a = 26
    modulus = 2**32
    hashHaystack, hashNeedle = 0, 0
    for i in range(len(needle)):
        hashNeedle = (hashNeedle * a + ch2int(needle[i])) % modulus
        hashHaystack = (hashHaystack * a + ch2int(haystack[i])) % modulus
    if hashNeedle == hashHaystack:
        return 0

    aL = 1
    for i in range(len(needle)):
        aL = (aL * a) % modulus

    for i in range(1, len(haystack) - len(needle) + 1):
        hashHaystack = ((hashHaystack * a - ch2int(haystack[i-1])*aL) + ch2int(haystack[i+len(needle)-1])) % modulus
        if hashHaystack == hashNeedle:
            return i
    return -1


def divide(dividend: int, divisor: int) -> int:
    """
    29 两数相除 返回被除数dividend除以除数divisor得到的商。
    """
    # the only situation that overflow occurs
    if dividend == -2 ** 31 and divisor == -1:
        return 2 ** 31 - 1

    signal = (dividend > 0 and divisor > 0) or \
             (dividend < 0 and divisor < 0)

    dividend, divisor = abs(dividend), abs(divisor)
    i = 0
    k = 1
    cur_divisor = divisor

    while dividend >= divisor:
        if dividend >= cur_divisor:
            dividend -= cur_divisor
            i += k

            k += k
            cur_divisor += cur_divisor

        else:
            k = 1
            cur_divisor = divisor
    i = i if signal else -i
    return i


def maxSubArray(nums: list) -> int:
    """
    53 最大子序和。给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    """
    if len(nums) == 1:
        return nums[0]

    result = nums[0]
    current_sum = 0
    for n in nums:
        current_sum = max(current_sum, 0) + n
        result = max(result, current_sum)
    return result


def spiralOrder(matrix: list) -> list:
    """
    54 螺旋矩阵。给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
    关键词：递归
    """
    if len(matrix) == 0:
        return []
    if len(matrix) == 1:
        return matrix[0]
    if len(matrix[0]) == 1:
        return [j for i in matrix for j in i]
    result = []
    result += matrix.pop(0)
    for i in range(len(matrix)):
        result.append(matrix[i].pop(-1))
    result += matrix.pop(-1)[::-1]
    for i in range(len(matrix)-1, 0, -1):
        result.append(matrix[i].pop(0))
    return result + spiralOrder(matrix)


def lengthOfLastWord(s: str) -> int:
    """
    58 最后一个单词的长度。给你一个字符串 s，由若干单词组成，单词之间用空格隔开。返回字符串中最后一个单词的长度。
    如果不存在最后一个单词，请返回 0

    思考：如果不用split应该怎么做
    """
    l = s.split()
    if len(l) == 0:
        return 0
    else:
        return len(l[-1])


def climbStairs(n):
    """
    70 爬楼梯。假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

    关键词： 动态规划
    """
    dp = [0, 1, 2]
    if n < 3:
        return dp[n]

    for i in range(3, n + 1):
        dp.append(dp[i - 1] + dp[i - 2])
    return dp[-1]


def subsets(nums):
    """
    78 给定整数数组sums，其中所有元素互不相同，返回所有子集

    关键词：互不相同
    """
    result = [[]]
    for i in nums:
        result = [r + [i] for r in result] + result
    return result


def removeDuplicates2(nums: list) -> int:
    """
    80 删除有序数组中的重复项，每个元素最多出现两次

    关键词：有序数组 最多两次
    """
    i = 0
    for e in nums:
        if i < 2 or e != nums[i - 2]:
            nums[i] = e
            i += 1
    return i


def subsetsWithDup(nums):
    """
    90 给定整数数组nums，其中元素可能重复，返回所有子集

    关键词：去重 参考78题
    """
    result = [()]
    for i in nums:
        result = [r + (i,) for r in result] + result

    result = [tuple(sorted(r)) for r in result]
    result = set(result)
    return list(result)


def maxProfit(prices: list) -> int:
    """
    121 买卖股票的最佳时机。\n
    关键词：动态规划 只能买卖一次
    """
    if len(prices) == 1:
        return 0
    minprice = prices[0]
    result = 0
    for p in prices[1:]:
        minprice = min(minprice, p)
        result = max(result, p-minprice)
    return result


def maxProfit2(prices: list) -> int:
    """
    122 买卖股票的最佳时机 II
    关键词：动态规划 能买卖任意次 卖出后才能再买
    """
    # 动态规划的做法
    # if len(prices) == 1:
    #     return 0
    # dp = [[0]*2] * len(prices)
    # # 初始状态
    # dp[0][0] = 0
    # dp[0][1] = -1 * prices[0]
    # for i in range(1, len(prices)):
    #     dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
    #     dp[i][1] = max(dp[i-1][0]-prices[i], dp[i-1][1])
    # return dp[-1][0]

    # 离谱的做法，不放过每次能挣钱的机会
    result = 0
    for i in range(1, len(prices)):
        result += max(prices[i]-prices[i-1], 0)
    return result


def findMin(nums):
    """
    153 寻找无重复升序旋转数组中的最小值\n
    关键词：二分查找 无重复
    """
    low = 0
    high = len(nums) - 1
    while high - low > 1:
        pivot = (high+low)//2
        if nums[pivot] > nums[high]:
            low = pivot + 1
        elif nums[pivot] < nums[high]:
            high = pivot
    return min(nums[low], nums[high])


def findMin2(nums):
    """
    154 寻找有重复升序旋转数组中的最小值\n
    关键词：有重复
    """
    low = 0
    high = len(nums) - 1
    while high - low > 1:
        pivot = (high+low)//2
        if nums[pivot] > nums[high]:
            low = pivot + 1
        elif nums[pivot] < nums[high]:
            high = pivot
        else:
            high -= 1   # 当nums[pivot]==nums[high]时，无法判断pivot在左边还是右边，因此干脆让high减一
    return min(nums[low], nums[high])


def largestNumber(nums: list) -> str:
    """
    179 最大数 给定一组非负整数nums重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
    :param nums: 非负整数数组
    :return: 重新排列后的最大整数
    """
    class myStr():
        def __init__(self, s):
            self.s = str(s)
        def __lt__(self, other):
            a, b = self.s, other.s
            if a == b:
                return True
            return a+b < b+a

    nums = sorted([myStr(n) for n in nums], reverse=True)
    return str(int("".join([n.s for n in nums])))


def isUgly(n: int) -> bool:
    """
    263 丑数 丑数就是只包含质因数 2、3 和/或 5 的正整数。
    """
    while n // 2 == n / 2:
        n /= 2
    while n // 3 == n / 3:
        n /= 3
    while n // 5 == n / 5:
        n /= 5
    return n == 1


# 474
def findMaxForm(strs, m, n) -> int:
    # dp = [[0]*(n+1)]*(m+1) #! 这种写法是错误的，改一个值会影响其他值
    dp = [[0]*(n+1) for _ in range(m+1)]
    for s in strs:
        m_ = s.count('0')
        n_ = s.count('1')

        # 为何这里要从大往小？
        for i in range(m, m_-1, -1):
            for j in range(n, n_-1, -1):
                # 这里每次修改i,j时，用到的i-m_,j-n_总比i,j要小，即为还未修改的上一个状态的值
                dp[i][j] = max(dp[i][j], 1+dp[i-m_][j-n_])
                print(i, j)
                print(i-m_, j-n_)
    return dp[m][n]




# 1006
def clumsy(N):
    product_div = []
    for i in range(N, 0, -4):
        if i >= 3:
            product_div.append(i * (i - 1) // (i - 2))
        elif i == 2:
            product_div.append(i * (i - 1))
        else:
            product_div.append(i)

    result = sum(range(N - 3, 0, -4))
    result = result + product_div[0] - sum(product_div[1:]) if len(product_div) else result
    return result
