from utils import StaticArray

class DynamicArray(StaticArray):
    def __init__(self, size, isSet = False):
        super(DynamicArray,self).__init__(size)
        self.isSet = isSet
        
    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        # resize when exceeds 0.8*size
        if self.__len__() >= 0.8 * self.size:
            self.reallocate(2 * self.size)
        # if set, won't allow repetitive elements
        if self.isSet == True and len(self.argwhere(value)) > 0 :
            return
        # if the index is None, just add value to the last element
        if self[index] == None:
            for ind, val in enumerate(self):
                if val == None:
                    super().__setitem__(ind, value)
                    break
        else:
            # shift every element one slot back
            for i in range(self.get_size()-1 , index-1, -1):
                super().__setitem__(i, self[i-1])
            super().__setitem__(index, value)

    def __delitem__(self, index):
        # resize when smaller than 0.2 * size
        if self.__len__() <= self.size * 0.2:
            self.reallocate(self.size//2)
        # if None, no need to delete
        if self[index] == None:
            return
        # if in the last slot, just set it None
        elif index == self.size - 1:
            del self[index]
        else:
            # shift every slot behind the index one slot forward, recovering the slot
            for i in range(index, self.get_size() - 1):
                super().__setitem__(i, self[i + 1])
            super().__setitem__(self.get_size() - 1, None)

    def append(self, value):
        if self.__len__() >= 0.8 * self.size:
            self.reallocate(2 * self.size)
        # if set, no repetitive
        if self.isSet == True and len(self.argwhere(value)) > 0:
            return
        # after shifting, insert new value
        self[self.size - 1] = value

    def extend(self, arr):
        for i in arr:
            self.append(i)

    def remove(self, value):
        # resize
        if self.__len__() <= self.size * 0.2:
            self.reallocate(self.size//2)
        # traverse and check the value
        for i, v in enumerate(self):
            if v == value:
                del self[i]
                break

    def argwhere(self, value):
        # counter
        count = 0
        # check how many repetitions
        for i, v in enumerate(self):
            if v == value:
                count += 1
        ret = StaticArray(count)
        for i, v in enumerate(self):
            if v == value:
                ret.append(i)
        return ret

    def __len__(self):
        # traverse and count
        ret = 0
        for v in self.data:
            if v is not None:
                ret += 1
        return ret

    def get_size(self):
        return self.size

    def __eq__(self, arr):
        # check type first
        if type(arr) == DynamicArray:
            # traverse and check each value
            for v1,v2 in zip(self, arr):
                if v1!=v2:
                    return False
            return True
        return False

    def __repr__(self):
        return '['+','.join([str(i) for i in self.data])+']'

    def __iter__(self):
        # traverse and yield
        for i in self.data:
            yield i

    def reallocate(self, size):
        # first take a copy of existing data
        copy = self.copy()
        # create a bigger space
        self.resize(size)
        # insert all the copy
        self.extend(copy)

    def resize(self, size):

        # create a larger collection
        self.data = [None]*size
        # record the size and give it back to self.size
        self.size = size



