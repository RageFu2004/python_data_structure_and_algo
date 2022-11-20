from utils import BinaryTree, TreeNode


class BST(BinaryTree):
    def __init__(self, arr, sort=False):
        super().__init__()
        self.arr = arr
        self.sort = sort
        self.return_lst = []
        if self.arr:
            if self.sort == True:
                self.arr.sort()
                mid = len(arr)//2
                self.addNode(self.arr[mid])
                self.arr[mid] = None
                for n in range(1, mid//2+1):
                    self.addNode(arr[mid-n*2])
                    self.addNode(arr[mid+n*2])
                    arr[mid-n*2] = None
                    arr[mid+n*2] = None
                for item in self.arr:
                    if item != None:
                        self.addNode(item)
            else:
                for i in self.arr:
                    self.addNode(i)

    def addNode(self, data):
        if self.root is None:
            self.root = TreeNode(data)
        else:
            head = self.root
            while True:
                if data < head.data:
                    if head.less:
                        head = head.less
                    else:
                        head.less = TreeNode(data)
                        break
                else:
                    if head.more:
                        head = head.more
                    else:
                        head.more = TreeNode(data)
                        break

    def removeNode(self, data):
        left = False
        right = False
        head = self.root
        if head.data == data:
            headz = head
        else:
            while True:
                if data < head.data:
                    if head.less:
                        if head.less.data == data:
                            headz = head.less
                            left = True
                            break
                        head = head.less
                else:
                    if head.more:
                        if head.more.data == data:
                            headz = head.more
                            right = True
                            break
                        head = head.more
        if headz.less or headz.more:
            if headz.less and headz.more:
                pointer = headz.more
                while pointer.less:
                    if pointer.less.less is None:
                        break
                    pointer = pointer.less
                temp = pointer.less.data
                headz.data = temp
                if pointer.less.more:
                    pointer.less = pointer.less.more
                else:
                    pointer.less = None
            else:
                if left:
                    if headz.less:
                        head.less = headz.less
                    if headz.more:
                        head.less = headz.more
                if right:
                    if headz.less:
                        head.more = headz.less
                    if headz.more:
                        head.more = headz.more
        else:
            if left:
                head.less = None
            if right:
                head.more = None

    def search(self, data):
        return data in self.tolist()
    
    def tolist(self):
        head = self.root
        self.traverse(head)
        return self.return_lst
            
    def height(self,data):
        #self.counter = [0]
        self.counter_f = [0]
        self.final_lst = []
        head = self.root
        while True:
            if data > head.data:
                head = head.more
            else:
                if data == head.data:
                    break
                else:
                    head = head.less
        #self.count(head)
        self.count_f(head)
        return max(self.final_lst)+1

    def balancefactor(self,data):
        self.counter_f = [0]
        self.final_lst = []
        head = self.root
        while True:
            if data > head.data:
                head = head.more
            else:
                if data == head.data:
                    break
                else:
                    head = head.less
        self.count_f(head)
        if len(self.final_lst) == 1:
            return self.final_lst[0]
        else:
            return max(self.final_lst[0], self.final_lst[1]) - min(self.final_lst[0], self.final_lst[1])

    def traverse(self, head):
        if head.less or head.more:
            if head.less:
                self.traverse(head.less)

            self.return_lst.append(head.data)

            if head.more:
                self.traverse(head.more)
            return
        else:
            self.return_lst.append(head.data)
            return

    def count(self, src):
        print(src)
        if src.less or src.more:
            if src.less:
                self.count(src.less)
            if src.more:
                self.count(src.more)
            self.counter[0] += 1
        else:
            return

    def count_f(self, src):
        if src.less or src.more:
            if src.less:
                self.counter_f[0] += 1
                self.count_f(src.less)
                self.counter_f[0] -= 1
            if src.more:
                self.counter_f[0] += 1
                self.count_f(src.more)
                self.counter_f[0] -= 1
        else:
            self.final_lst.append(self.counter_f[0])
            return



if __name__ == "__main__":
    tree = BST([4,2,6,1,7,9,10], sort=True)
    #print(tree.root, tree.root.less.more, tree.root.more.more)
    for n,h in zip([4,2,6,1,7,9,10],[1, 2, 3, 1, 1, 2, 1]):
        print("height", tree.height(n),h)