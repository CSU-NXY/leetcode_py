"""
本文件中存放leetcode经典题目
"""

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


def findMin(nums):
    """
    153 寻找旋转数组中的最小值\n
    关键词：二分查找
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
