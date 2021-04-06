"""
本文件中存放leetcode《面试题》题目
"""
import random


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


