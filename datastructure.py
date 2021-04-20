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

    def setValues(self, vals: list):
        if len(vals) == 0:
            return

        self.val = vals[0]
        p = self
        for v in vals[1:]:
            p.next = ListNode(v)
            p = p.next

    def deleteNode(self, node):
        """
        从链表中删除一个节点，题目保证该节点不会是末尾节点
        """
        node.val = node.next.val
        node.next = node.next.next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        tmp = self.children
        for ch in word:
            if ch not in tmp:
                tmp[ch] = {}
            tmp = tmp[ch]
        tmp['#'] = "#"

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        tmp = self.children
        for ch in word:
            if ch not in tmp:
                return False
            tmp = tmp[ch]
        return '#' in tmp

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        tmp = self.children
        for ch in prefix:
            if ch not in tmp:
                return False
            tmp = tmp[ch]
        return True
