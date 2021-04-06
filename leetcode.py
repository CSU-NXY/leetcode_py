from datastructure import *


def climbStairs(n):
    dp = [0, 1, 2]
    if n < 3:
        return dp[n]

    for i in range(3, n+1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[-1]


def subsets(nums):
    result = [[]]
    for i in nums:
        result = [r + [i] for r in result] + result
    return result


def subsetsWithDup(nums):
    result = [()]
    for i in nums:
        result = [r + (i,) for r in result] + result

    result = [tuple(sorted(r)) for r in result]
    result = set(result)
    return list(result)


def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
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


# 26
def removeDuplicates(nums):
    if len(nums) == 1:
        return 1
    i, j = 0, 1
    while j < len(nums):
        if nums[i] == nums[j]:
            j += 1
        else:
            i += 1
            nums[i] = nums[j]
            j += 1
    return i+1


# 1006
def clumsy(N):
    product_div = []
    for i in range(N, 0, -4):
        if i >= 3:
            product_div.append(i * (i-1) // (i-2))
        elif i == 2:
            product_div.append(i*(i-1))
        else:
            product_div.append(i)

    result = sum(range(N-3, 0, -4))
    result = result + product_div[0] - sum(product_div[1:]) if len(product_div) else result
    return result




