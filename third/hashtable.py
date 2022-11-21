from utils import DynamicArray
import random


class HashTable(DynamicArray): 
    def __init__(self, size, probe=0):
        super().__init__(size)
        self.probe = probe
        self.size = size
        self.collision_lst = [] #keys
    
    def hashCode(self, key):
        return key % self.size
    
    def __getitem__(self, key):
        #print(self.collision_lst, self.size)
        for i in self.collision_lst:
            if i[0] == key:
                if i[1] in self.data:
                    return i[1]
                else:
                    return None
        return self.data[key % self.size]
        
    def __setitem__(self, key, value):
        #print(self)
        hash_val = self.hashCode(key)
        if value is None:
            for i in self.collision_lst:
                if i[0] == key:
                    #print(self.collision_lst, self)
                    self.data[self.data.index(i[1])] = None
                    #print("after del", self,self[key])
                    return
            self.data[hash_val] = None
            return
        if self[hash_val] is None:
            self.data[hash_val] = value
        else:
            self.collision_lst.append([key,value])
            if self.probe == 0:
                while self.data[hash_val] is not None:
                    hash_val += 1
                self.data[hash_val] = value
                if self.loadfactor() == 1:
                    self.reallocate(self.size*2)
            if self.probe == 1:
                n = 1
                while self.data[hash_val] is not None:
                    hash_val += n**2
                    if hash_val >= self.size:
                        self.reallocate(self.size*2)
                self.data[hash_val] = value
            if self.probe == 2:
                while self.data[hash_val] is not None:
                    hash_val = random.randint(0, self.size-1)
                self.data[hash_val] = value
                if self.loadfactor() == 1:
                    print("oveload", self.size)
                    self.reallocate(self.size*2)
                    print("after", self.size, self)

    def __delitem__(self, key):
        #real_key = self.hashCode(key)
        self[key] = None

    def loadfactor(self,i=0):
        count = 0
        for i in self:
            if i != None:
                count += 1
        return count / self.size

if __name__ == "__main__":
    random.seed(0)
    for probe in [0, 1, 2]:
        for i in range(5, 10):
            ht = HashTable(i, probe)
            for j in range(i):
                ht[j * i] = j
            for j in range(i):
                print(ht[j * i], j)
