class Collections():
    def __init__(self):
        self.data = None

    def __getitem__(self, index):
        """
        Read the data at index of this collection.
        """
        pass

    def __setitem__(self, index, value):
        """
        Write the data at index of this collection to value.
        """
        pass

    def __delitem__(self, index):
        """
        Delete data at index of this collection.
        """
        pass

    def __len__(self):
        """
        Return the number of elements in this collection.
        """
        pass

    def get_size(self):
        """
        Return the total number of slots you have in this collection.
        """
        pass

    def __repr__(self):
        """
        Used to print out an instance of collection in readable form.
        """
        pass

    def __iter__(self):
        """
        Enable you to use this collection with iterable functions
        """
        pass

class StaticArray(Collections):
    def __init__(self, size):
        super().__init__()
        self.data = [None] * size
    def __getitem__(self, index):
        return self.data[index]
    def __setitem__(self, index, value):
        self.data[index] = value
    def __delitem__(self, index):
        self.data[index] = None
    def __len__(self):
        count = 0
        for v in self:
            if v is not None:
                count += 1
        return count
    def get_size(self):
        return len(self.data)
    def __repr__(self):
        return '['+','.join([str(i) for i in self])+']'
    def __iter__(self):
        for v in self.data:
            yield v
    def copy(self):
        copy = StaticArray(self.get_size())
        for i, value in enumerate(self):
            copy[i] = value
        return copy

class DynamicArray(StaticArray):
    def __init__(self, size):
        super().__init__(size)
    def reallocate(self, size):
        pass
    def resize(self, size):
        self.data = [None]*size

class TreeNode():
    def __init__(self,data):
        self.data = data
        self.less = None
        self.more = None
    def __repr__(self):
        return str(self.data)

class BinaryTree():
    def __init__(self):
        self.root = None
    def addNode(self,data):
        #Adds data in level-order (Not BST property)
        if self.root==None:
            self.root = TreeNode(data)
        else:
            node = self.root
            queue = [node]
            while queue:
                cur = queue.pop(0)
                for i, node in enumerate([cur.less,cur.more]):
                    if node==None:
                        if i==0:
                            cur.less = TreeNode(data)
                        else:
                            cur.more = TreeNode(data)
                        return
                    else:
                        queue.append(node)
    def removeNode(self,data):
        # Uses tolist to remove then adds back the data (Not BST property)
        ret = self.tolist()
        ret.remove(data)
        self.root = None
        if len(ret)>0:
            for i in ret:
                self.addNode(i)
    def search(self,data):
        # Check by converting to tolist (Not using BST property)
        return data in self.tolist()
    def tolist(self):
        # Load data into list using level order traversal
        ret = []
        if self.root!=None:
            queue = [self.root]
            while queue:
                cur = queue.pop(0)
                ret.append(cur.data)
                for node in [cur.less,cur.more]:
                    if node!=None:
                        queue.append(node)
        return ret
    def __repr__(self):
        return str(self.tolist())
