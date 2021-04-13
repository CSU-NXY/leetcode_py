# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        s = [self.val]
        p = self.next
        while p is not None:
            s.append(p.val)
            p = p.next
        return ", ".join([str(element) for element in s])

    def append(self, val):
        self.next = ListNode(val)
        return self.next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
