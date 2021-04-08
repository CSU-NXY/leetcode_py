"""
本文件中存放leetcode《剑指Offer》题目
"""
from datastructure import *


def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    """
    剑指Offer 52 输入两个链表，找到它们的第一个公共子节点，没有的话返回null \n
    关键词：双指针 路径长度相同 同时为None
    """
    p1, p2 = headA, headB

    # 妙啊，如果没有相遇的话大家都是None
    while p1 != p2:
        p1 = p1.next if p1 else headB
        p2 = p2.next if p2 else headA

    return p1


def search(nums, target):
    """
    剑指Offer 53 统计一个数字在排序数组中出现的次数
    """
    cnt = 0
    for e in nums:
        if e == target:
            cnt += 1
    return cnt