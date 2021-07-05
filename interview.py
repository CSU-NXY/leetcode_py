"""
本文件中存放leetcode《面试题》题目
"""
import random
from typing import List


def isUnique(astr: str) -> bool:
    """
    01.01 确定一个字符串 s 的所有字符是否全都不同。
    """
    count = [0] * 128
    for ch in astr:
        asci = ord(ch)
        count[asci] += 1
        if count[asci] > 1:
            return False
    return True


def CheckPermutation(s1: str, s2: str) -> bool:
    """
    01.02 判定s1经过重新排列后能否变成s2
    """
    nums = [0] * 128
    for ch in s1:
        nums[ord(ch)] += 1
    for ch in s2:
        nums[ord(ch)] -= 1
    for n in nums:
        if n != 0:
            return False
    return True


def replaceSpaces(S: str, length: int) -> str:
    """
    01.03 编写一种方法，将字符串中的空格全部替换为%20。假定该字符串尾部有足够的空间存放新增字符，并且知道字符串的“真实”长度。\n
    这题似乎要我们做inplace的解法，但我觉得没啥意思就没写
    """
    result = ''
    for i in range(length):
        ch = S[i]
        if ch == ' ':
            result += '%20'
        else:
            result += ch
    return result


def maxAliveYear(birth: List[int], death: List[int]) -> int:
    """
    16.10. 生存人数
    """
    N = len(birth)
    year_count = [0] * 2002

    for i in range(N):
        year_count[birth[i]] += 1
        year_count[death[i]+1] -= 1

    result = -1
    count, max_count = 0, 0
    for idx in range(1900, 2001):
        count += year_count[idx]
        if count > max_count:
            max_count = count
            result = idx
    return result


def divingBoard(shorter: int, longer: int, k: int) -> List[int]:
    """
    16.11. 跳水板
    """
    if k == 0:
        return []
    result = set()
    for i in range(k+1):
        result.add(shorter*i + longer*(k-i))
    return sorted(list(result))


def masterMind(solution: str, guess: str) -> List[int]:
    """
    16.15. 珠玑妙算
    """
    answer = [0, 0]
    answer[0] = sum(solution[i] == guess[i] for i in range(4))

    count = [0, 0, 0, 0]
    count[0] = min(solution.count('R'), guess.count('R'))
    count[1] = min(solution.count('Y'), guess.count('Y'))
    count[3] = min(solution.count('G'), guess.count('G'))
    count[2] = min(solution.count('B'), guess.count('B'))

    answer[1] = sum(count) - answer[0]
    return answer


def calculate(s: str) -> int:
    """
    16.26. 计算器
    """
    expressions = []

    token = ''
    for ch in s:
        if ch == ' ':
            continue
        if ch in ['+', '-', '*', '/']:
            expressions.append(token)
            expressions.append(ch)
            token = ''
        else:
            token += ch
    expressions.append(token)

    stack = []
    for e in expressions:
        stack.append(e)
        if len(stack) > 1 and stack[-2] in ['*', '/']:
            a2 = stack.pop(-1)
            op = stack.pop(-1)
            a1 = stack.pop(-1)
            stack.append(str(int(a1) * int(a2) if op == '*' else int(a1) // int(a2)))
    stack = stack[::-1]
    while len(stack) > 1:
        a1 = stack.pop(-1)
        op = stack.pop(-1)
        a2 = stack.pop(-1)
        stack.append(str(int(a1) + int(a2) if op == '+' else int(a1) - int(a2)))
    return int(stack[0])


def pondSizes(land: List[List[int]]) -> List[int]:
    """
    16.19. 水域大小
    """
    results = []

    def dfs(i, j, land, visited):
        if land[i][j] > 0 or visited[i][j]:
            return 0
        cnt = 1
        visited[i][j] = True
        for k in [i - 1, i, i + 1]:
            for f in [j - 1, j, j + 1]:
                if k < 0 or k >= len(land) or f < 0 or f >= len(land[0]):
                    continue
                if not visited[k][f] and land[k][f] == 0:
                    cnt += dfs(k, f, land, visited)
        return cnt

    visited = [[False] * len(_) for _ in land]
    for i in range(len(land)):
        for j in range(len(land[0])):
            area = dfs(i, j, land, visited)
            if area:
                results.append(area)
    return sorted(results)


def majorityElement(nums: list) -> int:
    """
    17.10 数组中占比超过一半的元素称之为主要元素。给定一个整数数组，找到它的主要元素。若没有，返回-1。

    关键词：摩尔投票法，比拼消耗。空间复杂度O(1)
    """
    cnt, num = 1, nums[0]
    for i in range(1, len(nums)):
        if nums[i] != num:
            cnt -= 1
            if cnt == 0:
                num = nums[i]
                cnt = 1
        else:
            cnt += 1
    # 最后需要校验一下是否为majorityElement
    return num if nums.count(num) > len(nums) / 2 else -1


def findClosest(words: List[str], word1: str, word2: str) -> int:
    """
    17.11 单词距离
    """
    last_word1, last_word2 = None, None
    distance = len(words) + 1
    for idx, w in enumerate(words):
        if w == word1:
            distance = min(distance, idx - last_word2) if last_word2 else distance
            last_word1 = idx
        elif w == word2:
            distance = min(distance, idx - last_word1) if last_word1 else distance
            last_word2 = idx
    return distance


def massage(nums: List[int]) -> int:
    """
     17.16. 按摩师
    """
    N = len(nums)
    if N == 0:
        return 0
    dp0, dp1 = 0, nums[0]
    for i in range(1, N):
        dpi = max(dp1, dp0+nums[i])
        dp0, dp1 = dp1, dpi
    return dp1