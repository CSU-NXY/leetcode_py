"""
本文件中存放leetcode经典题目
"""
import heapq
from typing import List

from datastructure import *


def twoSum(nums: List[int], target: int) -> List[int]:
    """
    1. 两数之和
    """
    dict = {}
    for idx, e in enumerate(nums):
        if target - e in dict:
            return [idx, dict[target-e]]
        else:
            dict[e] = idx
    return []


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


def findMedianSortedArrays(nums1: list, nums2: list) -> float:
    """
    4. 寻找两个正序数组的中位数，要求O(log(m+n))
    思路：将问题转化为寻找两个已排序数组中第k小的数，每次剔除k/2个数
    """
    def findKthSmallest(nums1: list, nums2: list, k: int) -> float:
        m, n = len(nums1), len(nums2)
        if m == 0:
            return nums2[k-1]
        if n == 0:
            return nums1[k-1]
        if k == 1:
            return min(nums1[0], nums2[0])
        i = min(m, k//2) - 1
        j = min(n, k//2) - 1
        if nums1[i] > nums2[j]:
            return findKthSmallest(nums1, nums2[j+1:], k - j - 1)
        else:
            return findKthSmallest(nums1[i+1:], nums2, k - i - 1)

    m, n = len(nums1), len(nums2)
    # m+n为奇数时left等于right
    left = (m + n + 1) // 2
    right = (m + n + 2) // 2
    return (findKthSmallest(nums1, nums2, left) + findKthSmallest(nums1, nums2, right)) / 2


def longestPalindrome(s: str) -> str:
    """
    5. 最长回文子串
    """
    def findPalindrome(s: str, idx1: int, idx2: int):
        result_str = s[idx1]
        while idx1 >= 0 and idx2 < len(s):
            if s[idx1] == s[idx2]:
                result_str = s[idx1:idx2 + 1]

                idx1 -= 1
                idx2 += 1
            else:
                break
        return result_str

    result = ""
    for i in range(len(s)):
        result1 = findPalindrome(s, i, i)
        result2 = findPalindrome(s, i, i+1)
        result = result1 if len(result1) > len(result) else result
        result = result2 if len(result2) > len(result) else result
    return result


def convert(s: str, numRows: int) -> str:
    """
    6. Z 字形变换
    先向下后向上，记录每行的字符
    """
    # 如果行数为１，则直接返回即可
    if numRows == 1:
        return s

    row = 0
    direct = 1

    from collections import defaultdict
    row_strs = defaultdict(list)

    for i in range(len(s)):
        row_strs[row].append(s[i])
        if row == numRows - 1:
            direct = -1
        if row == 0:
            direct = 1
        row += direct

    result = ""
    for _, line in row_strs.items():
        result += "".join(line)
    return result


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


def isMatch(s: str, p: str) -> bool:
    """
    10. 正则表达式匹配
    动态规划，注意dp矩阵的下标与字符串下标的差异
    """
    def match(i: int, j: int) -> bool:
        """
        匹配s中第i个字符和p中第j个字符
        """
        if i == 0:
            return False
        return p[j-1] == '.' or s[i-1] == p[j-1]

    def isStar(j: int) -> bool:
        return p[j-1] == '*'

    dp = [[False] * (len(p)+1) for _ in range(len(s)+1)]
    dp[0][0] = True

    # 在dp矩阵中，<i,j>表示匹配到s中第i个字符，p中第j个字符
    for i in range(len(s)+1):
        for j in range(1, len(p)+1):
            if isStar(j):
                dp[i][j] = dp[i][j-2]
                if match(i, j-1):
                    dp[i][j] |= dp[i-1][j]
            else:
                dp[i][j] |= match(i, j) and dp[i-1][j-1]
    return dp[len(s)][len(p)]


def maxArea(height: list) -> int:
    """
    11. 盛最多水的容器
    """
    area = -1
    l, r = 0, len(height)-1
    while l < r:
        area = max(area, (r-l)*min(height[l], height[r]))
        # 每次移动更矮的那边，因为那边已经达到了人生最巅峰了
        if height[l] <= height[r]:
            l += 1
        else:
            r -= 1
    return area


def intToRoman(num: int) -> str:
    """
    12. 数字转罗马字符
    """
    def bitToRoman(n: int, level: int) -> str:
        ones = ['I', 'X', 'C', 'M']
        fives = ['V', 'L', 'D']

        if n <= 3:
            return ones[level] * n
        elif n == 4:
            return ones[level] + fives[level]
        elif 5 <= n <= 8:
            return fives[level] + ones[level] * (n-5)
        elif n == 9:
            return ones[level] + ones[level+1]

    result = ""
    str_num = str(num)
    for i in range(len(str_num)):
        result = result + bitToRoman(int(str_num[i]), len(str_num)-i-1)
    return result


def romanToInt(s: str) -> int:
    """
    13.　罗马字符转数字
    """
    special = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
    normal = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    result = 0
    last_ch = '#'
    for ch in s:
        if last_ch == '#':
            last_ch = ch
        else:
            if last_ch + ch in special:
                result += special[last_ch+ch]
                last_ch = '#'
            else:
                result += normal[last_ch]
                last_ch = ch
    if last_ch != '#':
        result += normal[last_ch]
    return result


def longestCommonPrefix(strs: List[str]) -> str:
    """
    14. 最长公共前缀
    """
    if not strs:
        return ""
    length0, count = len(strs[0]), len(strs)
    for i in range(length0):
        if any(i == len(strs[j]) or strs[j][i] != strs[0][i] for j in range(1, count)):
            return strs[0][:i]
    return strs[0]


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


def letterCombinations(digits: str) -> list:
    """
    17. 电话号码的字母组合
    """
    if len(digits) == 0:
        return []

    dic = {"2":"abc", "3":"def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8":"tuv", "9":"wxyz"}
    result = [""]
    for d in digits:
        letters = dic[d]
        result = [s + l for s in result for l in letters]
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


def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    """
    19. 删除链表的倒数第 N 个结点
    双指针，第一个指针先走n步，然后两个指针一起向前走。当第一个指针到达末尾时，第二个指针刚好在要删除的节点那里
    """
    prev = ListNode(next=head)
    p1, p2 = prev, prev
    for _ in range(n):
        p1 = p1.next

    while p1.next:
        p1 = p1.next
        p2 = p2.next
    p2.next = p2.next.next
    return prev.next


def isValid(s: str) -> bool:
    """
    20. 有效的括号
    """
    def match(ch1: str, ch2: str) -> bool:
        case1 = (ch1 == '(' and ch2 == ')')
        case2 = (ch1 == '[' and ch2 == ']')
        case3 = (ch1 == '{' and ch2 == '}')
        return any([case1, case2, case3])

    stack = []
    for ch in s:
        if ch in ['(', '[', '{']:
            stack.append(ch)
        elif stack and match(stack[-1], ch):
            stack.pop(-1)
        else:
            return False
    return not stack


def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    21. 合并两个有序链表
    用递归的思想逐个合并节点
    """
    if not l1 or not l2:
        return l1 if not l2 else l2
    if l1.val < l2.val:
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists(l1, l2.next)
        return l2


def generateParenthesis(n: int) -> list:
    """
    22. 括号生成 \n
    思路：递归逐一添加左括号或右括号，剩余的左括号数应当始终小于等于右括号数
    """
    def helper(s, left, right, result):
        if left == right == 0:
            result.append(s)
            return
        if left > right or right < 0 or left < 0:
            return
        helper(s+'(', left-1, right, result)
        helper(s + ')', left, right-1, result)

    result = []
    helper("", n, n, result)
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


def removeElement(nums: list, val: int) -> int:
    """
    27. 移除元素。Inplace移除元素，返回移除后的长度
    """
    i = 0
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += 1
    return i


def removeElement(nums: list, val: int) -> int:
    """
    27. 移除元素。Inplace移除数值等于val的元素，返回移除后的长度.
    """
    l, r = 0, len(nums)
    while l < r:
        if nums[l] == val:
            nums[l] = nums[r-1]
            r -= 1
        else:
            l += 1
    return l


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


def nextPermutation(nums: list) -> None:
    """
    31. 下一个排列
    """
    if len(nums) == 1:
        return

    flag = True
    for i in range(len(nums)-1, 0, -1):
        if nums[i] > nums[i-1]:
            nums[i], nums[i-1] = nums[i-1], nums[i]
            flag = False
            break

    if flag:
        nums.sort()


def combinationSum(candidates: list, target: int) -> list:
    """
    39. 组合总和
    思路：升序，去重，从小到大选择数字
    """
    candidates = sorted(candidates)

    result = []

    def helper(s, use, remain):
        for i in range(s, len(candidates)):
            c = candidates[i]
            if c == remain:
                result.append(use+[c])
            elif c < remain:
                helper(i, use+[c], remain-c)
            else:
                return
    helper(0, [], target)
    return result


def multiply(num1: str, num2: str) -> str:
    """
    43. 字符串相乘
    """
    if num1 == '0' or num2 == '0':
        return '0'

    m, n = len(num1), len(num2)
    tmp = [0] * (m + n)
    for i in range(m):
        for j in range(n):
            tmp[m+n-1-i-j] += int(num1[m-1-i]) * int(num2[n-1-j])

    t = 0
    for i in range(len(tmp)-1, -1, -1):
        tmp[i] += t
        t = tmp[i] // 10
        tmp[i] = tmp[i] % 10

    i = 0
    while tmp[i] == 0:
        i += 1

    result = ""
    while i < len(tmp):
        result += str(tmp[i])
        i += 1
    return result


def permute(nums: list) -> list:
    """
    46. 全排列 返回数组nums的全排列
    思路：可以用itertools.permutations函数；参考答案用的是回溯，但是我没看懂
    """
    if len(nums) == 0:
        return []
    if len(nums) == 1:
        return [nums]

    result = []
    for i in range(len(nums)):
        # 选定一个数作为开头，将问题转化为求剩余n-1个数的全排列
        result += [[nums[i]] + p for p in permute(nums[:i]+nums[i+1:])]
    return result


def rotate(matrix: list) -> None:
    """
    48. 旋转图像。Inplace顺时针旋转90°，O(1)空间复杂度
    思路：从外往内一圈一圈旋转
    """
    n = len(matrix)
    for i in range(n//2):
        for j in range(i, n-1-i):
            tmp = matrix[i][j]
            x, y = n-1-i, n-1-j
            matrix[i][j] = matrix[y][i]
            matrix[y][i] = matrix[x][y]
            matrix[x][y] = matrix[j][x]
            matrix[j][x] = tmp


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


def myPow(x: float, n: int) -> float:
    """
    50. Pow(x, n)
    """
    sign = n >= 0

    n = abs(n)
    i = 0
    result = 1
    tmp, k = x, 1
    while i < n:
        if i+k <= n:
            result *= tmp
            i += k

            tmp *= tmp
            k *= 2
        else:
            tmp, k = x, 1
    return result if sign else 1/result


def solveNQueens(n: int) -> list:
    """
    51. N 皇后
    思路：回溯，可以考虑用哈希表或位运算加速isValid函数
    """
    def isValid(board, row, col):
        for i in range(n):
            if board[row][i] == 'Q' or board[i][col] == 'Q':
                return False
            if row+i < n and col+i < n and board[row + i][col + i] == 'Q':
                return False
            if row-i >= 0 and col-i >= 0 and board[row - i][col - i] == 'Q':
                return False
            if row+i < n and col-i >= 0 and board[row + i][col - i] == 'Q':
                return False
            if row-i >= 0 and col+i < n and board[row - i][col + i] == 'Q':
                return False
        return True

    def placeOrRemoveQueen(board, row, col, place=True):
        list_row = list(board[row])
        list_row[col] = 'Q' if place else '.'
        board[row] = ''.join(list_row)

    result = []
    def backtrace(board, k, result, last_row):
        if k == 0:
            result.append(list(board))
            return
        row = last_row+1
        for col in range(n):
            if isValid(board, row, col):
                placeOrRemoveQueen(board, row, col, True)
                backtrace(board, k-1, result, row)
                placeOrRemoveQueen(board, row, col, False)
        return
    backtrace(["."*n for _ in range(n)], n, result, -1)
    return result

def totalNQueens(n: int) -> int:
    """
    52. N皇后 II
    """
    def backtrace(row, columnConfilict=0, dConfilict=0, idConfilict=0):
        if row == n:
            return 1
        cnt = 0
        available = ((1 << n) - 1) & ~(columnConfilict | dConfilict | idConfilict)
        while available:
            position = (-available) & available
            available = available & (available-1)
            cnt += backtrace(row + 1, columnConfilict | position, (dConfilict | position) >> 1, (idConfilict | position) << 1)
        return cnt
    return backtrace(0)


def maxSubArray(nums: List[int]) -> int:
    """
    53. 最大子序和
    """
    result, s = nums[0], nums[0]
    for j in range(1, len(nums)):
        s = max(s+nums[j], nums[j])
        result = max(result, s)
        s = max(s, 0)
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


def canJump(nums: list) -> bool:
    """
    55. 跳跃游戏

    给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

    数组中的每个元素代表你在该位置可以跳跃的最大长度。

    判断你是否能够到达最后一个下标。
    """
    maxJump = -1
    for idx in range(len(nums)):
        maxJump = max(maxJump, idx+nums[idx])
        if maxJump == idx:
            break
    return maxJump >= len(nums)-1


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


def plusOne(digits: list) -> list:
    """
    66. 加一
    """
    c = 0
    for i in range(len(digits)-1, -1, -1):
        c = (digits[i] + 1) // 10
        digits[i] = (digits[i] + 1) % 10

        if c == 0:
            break
    if c == 1:
        digits.insert(0, 1)
    return digits


def addBinary(a: str, b: str) -> str:
    """
    67. 二进制求和
    """
    return '{0:b}'.format(int(a,2) + int(b,2))


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


def combine(n: int, k: int) -> list:
    """
    77. 组合 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
    """
    def subcombine(k, arr):
        if k == 0 or len(arr) < k:
            return [[]]
        result = []
        for i in range(len(arr)-k+1):
            result += [[arr[i]] + subresult for subresult in subcombine(k-1, arr[i+1:])]
        return result
    return subcombine(k, list(range(1, n+1)))


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


def isScramble(s1: str, s2: str) -> bool:
    """
    87. 扰乱字符串
    关键词：记忆化递归
    """
    if s1 == s2:
        return True

    def eq(s1: str, s2: str):
        a, b = {}, {}
        for ch in s1:
            a[ch] = a.get(ch, 0) + 1
        for ch in s2:
            b[ch] = b.get(ch, 0) + 1
        return a == b

    def helper(s1: str, s2: str, memory):
        result = memory.get(s1+'#'+s2, -1)
        if result == 1:
            return True
        if result == 0:
            return False
        if len(s1) != len(s2):
            memory[s1+'#'+s2] = 0
            return False
        if s1 == s2:
            memory[s1+'#'+s2] = 1
            return True

        for i in range(len(s1)-1):
            if eq(s1[:i+1], s2[:i+1]) and eq(s1[i+1:], s2[i+1:]) and\
                    helper(s1[:i+1], s2[:i+1], memory) and helper(s1[i+1:], s2[i+1:], memory):
                memory[s1+'#'+s2] = 1
                return True
            if eq(s1[:i+1], s2[-i-1:]) and eq(s1[i+1:], s2[:len(s1)-i-1]) and\
                    helper(s1[:i+1], s2[-i-1:], memory) and helper(s1[i+1:], s2[:len(s1)-i-1], memory):
                memory[s1+'#'+s2] = 1
                return True
        memory[s1+'#'+s2] = 0
        return False
    memory = {}
    return helper(s1, s2, memory)


def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    88. 合并两个有序数组
    """
    i, j = m-1, n-1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[i+j+1] = nums1[i]
            i -= 1
        else:
            nums1[i+j+1] = nums2[j]
            j -= 1
    if i == -1:
        nums1[:j+1] = nums2[:j+1]
    else:
        nums1[:i+1] = nums1[:i+1]


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


def rob(nums: list) -> int:
    """
    198. 打家劫舍
    关键词：动态规划
    """
    if len(nums) == 1:
        return nums[0]

    nosteal = nums[0]
    steal = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        # 这回合不能偷/能偷
        nosteal, steal = steal, max(nosteal+nums[i], steal)
    return steal


def numIslands(grid: List[List[str]]) -> int:
    """
    200. 岛屿数量
    """
    def bfs(i, j, grid, visited):
        if i > 0 and grid[i-1][j] == '1' and not visited[i-1][j]:
            visited[i-1][j] = True
            bfs(i-1, j, grid, visited)
        if j > 0 and grid[i][j-1] == '1' and not visited[i][j-1]:
            visited[i][j-1] = True
            bfs(i, j-1, grid, visited)
        if i < len(grid)-1 and grid[i+1][j] == '1' and not visited[i+1][j]:
            visited[i+1][j] = True
            bfs(i+1, j, grid, visited)
        if j < len(grid[0])-1 and grid[i][j+1] == '1' and not visited[i][j+1]:
            visited[i][j+1] = True
            bfs(i, j+1, grid, visited)

    count = 0
    visited = [[False] * len(_) for _ in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1' and not visited[i][j]:
                visited[i][j] = True
                count += 1
                bfs(i, j, grid, visited)
    return count


def rangeBitwiseAnd(left: int, right: int) -> int:
    """
    201. 数字范围按位与
    """
    if left == right:
        return left

    left = bin(left)[2:]
    right = bin(right)[2:]

    if len(left) != len(right):
        return 0

    result = '0b'
    for i in range(len(left)):
        if left[i] == right[i]:
            result += left[i]
        else:
            break
    left = left[len(result)-2:]
    right = right[len(result)-2:]

    result += '0' * len(left)
    return int(result, 2)


def removeElements(head: ListNode, val: int) -> ListNode:
    """
    203. 移除链表元素
    """
    pprev = ListNode()
    pprev.next = head

    prev, p = pprev, head
    while p:
        if p.val == val:
            p = p.next
            prev.next = p
            continue
        p = p.next
        prev = prev.next
    return pprev.next


def countPrimes(n: int) -> int:
    """
    204. 计数质数
    """
    primes = []
    isPrime = [True] * n
    for i in range(2, n):
        if isPrime[i]:
            primes.append(i)
        for p in primes:
            if i * p >= n:
                break
            isPrime[i*p] = False
            if p != 1 and i % p == 0:
                break
    return len(primes)


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    207. 课程表
    """
    indegree, outdegree = {}, {}
    for p in prerequisites:
        indegree[p[1]] = indegree.get(p[1], 0) + 1
        outdegree.setdefault(p[0], []).append(p[1])

    zeroindegrees = list(set(range(numCourses)) - set(indegree.keys()))
    while zeroindegrees:
        z = zeroindegrees.pop(-1)
        for out in outdegree.get(z, []):
            indegree[out] -= 1
            if indegree[out] == 0:
                zeroindegrees.append(out)
                indegree.pop(out)
    return not indegree



def minSubArrayLen(target: int, nums: List[int]) -> int:
    """
    209. 长度最小的子数组
    """
    i, total = 0, 0
    min_length = len(nums) + 1
    for j in range(len(nums)):
        total += nums[j]
        while i <= j and total >= target:
            min_length = min(min_length, j-i+1)
            total -= nums[i]
            i += 1
        if min_length == 1:
            break
    return min_length if min_length != len(nums) + 1 else 0


def rob2(nums: list) -> int:
    """
    213. 打家劫舍 II
    思路：拆成[1:]和[:-1]两个序列，返回rob的最大值就行
    """
    if len(nums) == 1:
        return nums[0]

    return max(rob(nums[1:]), rob(nums[:-1]))


def findKthLargest(nums: List[int], k: int) -> int:
    """
    215. 数组中的第K个最大元素
    """
    kHeap = nums[:k].copy()
    heapq.heapify(kHeap)
    for i in nums[k:]:
        heapq.heappushpop(kHeap, i)
    return kHeap[0]


def containsDuplicate(nums: List[int]) -> bool:
    """
    217. 存在重复元素
    """
    s = set()
    for e in nums:
        if e in s:
            return True
        else:
            s.add(e)
    return False


def findKthLargest2(nums: List[int], k: int) -> int:
    """
    自己实现堆
    """
    def push(nums: List[int], i: int):
        # 不断提升
        while i and nums[i] > nums[(i-1)//2]:
            nums[i], nums[(i-1)//2] = nums[(i-1)//2], nums[i]
            i = (i-1) // 2

    def pop(nums: List[int]) -> int:
        result = nums[0]

        idx = 0
        nums[idx] = nums.pop(-1)
        left, right = idx*2+1, idx*2+2
        largest = idx
        # 不断下拉
        while True:
            if left < len(nums) and nums[left] > nums[idx]:
                largest = left
            if right < len(nums) and nums[largest] < nums[right]:
                largest = right
            if largest == idx:
                break

            nums[idx], nums[largest] = nums[largest], nums[idx]
            idx = largest
            left, right = idx*2+1, idx*2+2
        return result

    def heapify(nums: List[int]):
        for i in range(1, len(nums)):
            push(nums, i)

    heapify(nums)
    for _ in range(k-1):
        pop(nums)
    return nums[0]


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


def nthUglyNumber(n: int) -> int:
    """
    264. 丑数 II
    思路：三个指针分别对应2、3和5
    """
    dp = [0] * (n+1)
    dp[1] = 1
    p2, p3, p5 = 1, 1, 1
    for i in range(2, n+1):
        tmp2, tmp3, tmp5 = dp[p2] * 2, dp[p3] * 3, dp[p5] * 5
        dp[i] = min(tmp2, tmp3, tmp5)
        # 注意这里不能用if else。比如当dp[i]等于6时，p2和p3要同时加一
        if dp[i] == tmp2:
            p2 += 1
        if dp[i] == tmp3:
            p3 += 1
        if dp[i] == tmp5:
            p5 += 1
    return dp[n]


def lengthOfLIS(nums: List[int]) -> int:
    """
    300. 最长递增子序列
    """
    # 方法一：动态规划
    # dp = [1] * len(nums)
    # for i in range(1, len(nums)):
    #     tmp = 0
    #     for j in range(i):
    #         if nums[j] < nums[i]:
    #             tmp = max(tmp, dp[j])
    #     dp[i] = tmp + 1
    # return max(dp)

    # 方法二：贪心+二分
    # 构建一个数组d，保证d[i]是，末尾元素最小的，长度为i的最长上升子序列，的末尾元素
    # d的长度就是就是最长子序列的长度
    def binary_search(arr, target):
        l, r = 0, len(arr)
        while l < r:
            mid = (l+r) >> 1
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                l = mid+1
            else:
                r = mid
        return l

    d = []
    for e in nums:
        if not d or e > d[-1]:
            d.append(e)
        else:
            d[binary_search(d, e)] = e
    return len(d)



def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    350. 两个数组的交集 II
    """
    nums1, nums2 = sorted(nums1), sorted(nums2)
    result = []
    i, j = 0, 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            result.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return result


def validUtf8(data: List[int]) -> bool:
    """
    393. UTF-8 编码验证
    """
    # data = [format(v, '08b')[-8:] for v in data]
    #
    # def checkBytes(v: str) -> int:
    #     if v[0] == '0':
    #         return 1
    #     for i in range(8):
    #         if v[i] == '0':
    #             break
    #     return i if 1 < i <= 4 else -1
    #
    # def checkN(values: List[str]) -> bool:
    #     for v in values[1:]:
    #         if v[0] != '1' or v[1] != '0':
    #             return False
    #     return True
    #
    # while data:
    #     n = checkBytes(data[0])
    #     if n == -1 or n > len(data) or not checkN(data[:n]):
    #         return False
    #     data = data[n:]
    # return True

    n = 0
    for i in range(len(data)):
        if n > 0:
            if data[i] >> 6 != 0b10:
                return False
            n -= 1
        elif data[i] >> 7 == 0:
            n = 0
        elif data[i] >> 5 == 0b110:
            n = 1
        elif data[i] >> 4 == 0b1110:
            n = 2
        elif data[i] >> 3 == 0b11110:
            n = 3
        else:
            return False
    return n == 0


def findMaxForm(strs, m, n) -> int:
    """
    474. 一和零
    关键词：01背包 动态规划
    """
    # dp = [[0]*(n+1)]*(m+1) #! 这种写法是错误的，改一个值会影响其他值
    dp = [[0]*(n+1) for _ in range(m+1)]
    for s in strs:
        m_ = s.count('0')
        n_ = s.count('1')

        for i in range(m, m_-1, -1):
            for j in range(n, n_-1, -1):
                # 这里每次修改i,j时，用到的i-m_,j-n_总比i,j要小，即为还未修改的上一个状态的值
                dp[i][j] = max(dp[i][j], 1+dp[i-m_][j-n_])
    return dp[m][n]


def topKFrequent(words: List[str], k: int) -> List[str]:
    """
    692. 前K个高频单词
    """
    class Entry:
        def __init__(self, word, count):
            self.word = word
            self.count = count

        def __lt__(self, other):
            result1 = self.count < other.count
            result2 = self.count == other.count and self.word > other.word
            return result1 or result2

    entries = {}
    for w in words:
        entries[w] = entries.get(w, 0) + 1

    heap = []
    dummy = []
    for idx, (word, count) in enumerate(entries.items()):
        if idx < k:
            heapq.heappush(heap, Entry(word, count))
        else:
            heapq.heappushpop(heap, Entry(word, count))
    result = []
    while heap:
        result.append(heapq.heappop(heap).word)
    return result[::-1]


def minDiffInBST(root: TreeNode) -> int:
    """
    783. 二叉搜索树节点最小距离 \n
    关键词：中序遍历， BFS
    """
    def dfs(root, prev, result):
        if root is None:
            return
        dfs(root.left, prev, result)
        # 题目中约定val非负
        # print(root.val)
        if prev[0] == -1:
            prev[0] = root.val
        else:
            result[0] = min(result[0], root.val-prev[0])
            prev[0] = root.val
        dfs(root.right, prev, result)

    prev, result = [-1], [10**6]
    dfs(root, prev, result)
    return result[0]


def isCousins(root: TreeNode, x: int, y: int) -> bool:
    """
    993. 二叉树的堂兄弟节点
    """
    parents, depths = [], []

    def traverse(node, parent=None, depth=0):
        if node is None:
            return
        if node.val in [x, y]:
            parents.append(parent)
            depths.append(depth)

        if len(parents) == 2:
            return
        traverse(node.left, node, depth+1)
        traverse(node.right, node, depth+1)

    traverse(root)
    return parents[0] != parents[1] and depths[0] == depths[1]

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


def countParis(deliciousness: List[int]) -> int:
    """
    1711. 大餐计数
    """
    targets = set([2**i for i in range(22)])

    from collections import defaultdict
    course_dict = defaultdict(int)
    for course in deliciousness:
        course_dict[course] += 1

    result = 0
    courses = list(course_dict.keys())
    for course in courses:
        for target in targets:
            c = target - course
            if c in course_dict:
                if c == course:
                    result += course_dict[course] * (course_dict[course]-1) // 2
                else:
                    result += course_dict[course] * course_dict[c]
        course_dict.pop(course)
    return result % (10**9+7)


def kthLargestValue(matrix: List[List[int]], k: int) -> int:
    """
    1738. 找出第 K 大的异或坐标值
    """
    valueMatrix = [[] for _ in matrix]
    values = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i == 0 and j == 0:
                valueMatrix[i].append(matrix[i][j])
            elif i == 0:
                valueMatrix[i].append(matrix[i][j] ^ valueMatrix[i][j-1])
            elif j == 0:
                valueMatrix[i].append(matrix[i][j] ^ valueMatrix[i-1][j])
            else:
                valueMatrix[i].append(matrix[i][j] ^ valueMatrix[i-1][j] ^ valueMatrix[i][j-1] ^ valueMatrix[i-1][j-1])
            heapq.heappush(values, valueMatrix[i][-1])
    asdf = heapq.nlargest(k, values)
    return heapq.nlargest(k, values)[-1]