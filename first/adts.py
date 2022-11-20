from utils import *
from arrays import *
from linkedlist import *


class StackOrQueue():
    def __init__(self,  useLinkedList = False, isQueue = False):
        # if it is LinkedList, create a linkedlist
        self.useLinkedList = useLinkedList
        if self.useLinkedList:
            self.data = LinkedList()
        else:
            # if not, use the DynamicArray
            self.data = DynamicArray(10)
        self.isQueue = isQueue

    def peek(self):
        # queue peek the first value
        if self.isQueue:
            return self.data[0]
        else:
            # stack peek the last
            return self.data[self.data.__len__()-1]

    def push(self, value):
        # append
        self.data.append(value)

    def pop(self):
        # queue pops out the first and return it
        if self.isQueue:
            temp = self.data[0]
            del self.data[0]
            return temp
        else:
            # stack pops out the last and return it
            temp = self.data[self.data.__len__()-1]
            del self.data[self.data.__len__()-1]
            return temp

    def __repr__(self):
        pass


