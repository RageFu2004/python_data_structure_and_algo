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

    def append(self, value):
        """
        Add value to the next open slot in this collection.
        """
        pass

    def extend(self, arr):
        """
        Appends values in arr to this collection.
        """
        pass

    def remove(self, value):
        """
        Remove the first instance of value in this collection.
        """
        pass

    def argwhere(self, value):
        """
        Find and return the indicies of all instances of value in this collection.
        """
        pass

    def __len__(self):
        """
        Return the number of elements in this collection.
        """
        pass

    def size(self):
        """
        Return the total number of slots you have in this collection.
        """
        pass

    def __eq__(self, arr):
        """
        Compare collections with each other.
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
        self.size = size
        self.data = [None] * self.size

    def __getitem__(self, index):
        return self.data[index]
        
    def __setitem__(self, index, value):
        # Override value at index
        self.data[index] = value

    def __delitem__(self, index):
        self.data[index] = None

    def append(self, value):
        #Place value in the next open slot.
        for i, v in enumerate(self):
            if v == None:
                self[i] = value
                break

    def extend(self, arr):
        for v in arr:
            self.append(v)

    def remove(self, value):
        for i,v in enumerate(self):
            if v == value:
                del self[i]
                break

    def argwhere(self, value):
        count = 0
        for i,v in enumerate(self):
            if v == value:
                count += 1
        ret = StaticArray(count)
        for i,v in enumerate(self):
            if v == value:
                ret.append(i)
        return ret

    def __len__(self):
        ret = 0
        for v in self.data:
            if v is not None:
                ret += 1
        return ret

    def get_size(self):
        return self.size

    def __repr__(self):
        return '['+','.join([str(i) for i in self.data])+']'

    def __eq__(self, arr):
        if type(arr) == StaticArray:
            for v1,v2 in zip(self, arr):
                if v1!=v2:
                    return False
            return True
        return False

    def __iter__(self):
        for v in self.data:
            yield v

    def copy(self):
        ret = StaticArray(self.get_size())
        for i,v in enumerate(self):
            ret[i] = v
        return ret




class Node(object):
    def __init__(self, data, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

    def __eq__(self,node):
        if type(node)==Node:
            return self.data == node.data
        return False

    def __repr__(self):
        return f'Node {self.data}'

