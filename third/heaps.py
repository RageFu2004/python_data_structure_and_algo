from utils import *


def heapify(arr, isMax):
    result_tree = BinaryTree()
    for i in arr:
        result_tree.addNode(i)
    print("temporary check：", result_tree.tolist())
    head = result_tree.root
    if isMax is True:
        while True:
            changed = [False]
            check_change(head, changed)
            if changed[0] is False:
                break

    if isMax is False:
        while True:
            changed = [False]
            check_change2(head, changed)
            if changed[0] is False:
                break
    return result_tree


def check_change(start, lst):
    if max_play2(start):
        lst[0] = True
    if start.less:
        check_change(start.less, lst)
    if start.more:
        check_change(start.more, lst)
    return


def check_change2(start, lst):
    if min_play2(start):
        lst[0] = True
    if start.less:
        check_change2(start.less, lst)
    if start.more:
        check_change2(start.more, lst)
    return


def min_play2(node):
    if node.less and node.more:
        if node.less.data < node.data:
            node.less.data, node.data = node.data, node.less.data
            min_play2(node.less)
            return True
        if node.more.data < node.data:
            node.more.data, node.data = node.data, node.more.data
            min_play2(node.more)
            return True
    else:
        return


def max_play2(node):
    if node.less and node.more:
        if node.less.data > node.data:
            node.less.data, node.data = node.data, node.less.data
            max_play2(node.less)
            return True
        if node.more.data > node.data:
            node.more.data, node.data = node.data, node.more.data
            max_play2(node.more)
            return True
    else:
        return


if __name__ == "__main__":
    """04 Heaps: Heapify 1 (min)"""
    tree = heapify([5, 1, 2, 3, 5], False)
    print(tree.tolist(), [1, 3, 2, 5, 5])

    """04 Heaps: Heapify 1 (max)"""
    tree = heapify([5, 1, 2, 3, 5], True)
    print(tree.tolist(), [5, 5, 2, 3, 1])

    """04 Heaps: Heapify 2 (min)"""
    tree = heapify([-1, 2, 5, 20, 1.2], False)
    print(tree.tolist(), [-1, 1.2, 5, 20, 2])

    """04 Heaps: Heapify 2 (max)"""
    tree = heapify([-1, 2, 5, 20, 1.2], True)
    print(tree.tolist(), [20, 2, 5, -1, 1.2])