# kmp algo
def main():
    strP = input('matchmaking string:')
    strM = input('match string:')
    return kmp_algo(strP, strM)
def kmp_algo(matched, matcher):
    next_array = make_next(matcher)
    # print(next_array)
    i = 0
    j = 0
    while i <= len(matched)-1:
        if j == len(matcher)-1:
            return True
        # print('matching', matched[i], matcher[j])
        if matched[i] == matcher[j]:
            i += 1
            j += 1
        else:
            if next_array[j] == 0:
                i += 1
                j = 0
            else:
                j = next_array[j] - 1
    return False
def make_next(strM):
    next_array = [0, 1] + [0] * (len(strM) - 2)
    start = 0
    next_one = 1
    count = 0
    while next_one < len(strM)-1:
        if strM[start] == strM[next_one]:
            count += 1
            next_array[next_one+1] = count+1
            start += 1
            next_one += 1
        else:
            count = 0
            start = 0
            if strM[start] == strM[next_one]:
                next_array[next_one + 1] = 2
                count += 1
                start += 1
            else:
                next_array[next_one + 1] = 1
            next_one += 1
    return next_array

# huffman tree
class HuffTreeNode:
    def __init__(self, val):
        self.val = val
        self.right = None
        self.left = None
class HuffTree:
    def __init__(self):
        self.root = None

    def show(self):
        p = self.root
        ret = []

        def traverse(node):

            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
            ret.append(node.val)
            return

        traverse(p)
        return ret
def huff(lst):
    for i in range(len(lst)):
        num = lst[i]
        lst[i] = HuffTreeNode(num)
    tree = HuffTree()
    while len(lst) > 1:
        node1 = lst.pop(0)
        node2 = lst.pop(0)
        new_node = HuffTreeNode(node1.val+node2.val)
        new_node.left = node1
        new_node.right = node2
        lst.insert(0, new_node)
    root = lst.pop()
    tree.root = root
    return tree


