from utils import Node, Collections, StaticArray

class LinkedList(Collections):
    def __init__(self, isSet = False, isDoubly = False, isCircular = False):
        super(LinkedList,self).__init__()
        self.head = None
        self.isSet = isSet
        self.isDoubly = isDoubly
        self.isCircular = isCircular

    def __getitem__(self, index):
        count = 0
        cur = self.head
        while True:
            if cur == None:
                break
            if count == index:
                return cur
            cur = cur.next
            count += 1
    '''for the setitem, the core logic is that when appending, which condition it might go to( especially in isCircular and isDoubly
'''

    def __setitem__(self, index, value):
        # when appending, the first element we append goes under this
        if self.head == None:
            self.head = Node(value)
        # change the first element, the head
        elif index == 0:
            # align the new node to head and make the new node the new head( heritage the existing link from the head)
            new_head = Node(value)
            new_head.next = self.head
            self.head = new_head
            # when inserting into the 0 position(shifting), just .next to the next node and the node.prev to the head
            if self.isDoubly:
                self.head.next.prev = self.head

        # this works for append(), when the index is None, we give the last node a new .next slot
        elif self[index] == None:
            cur = self[len(self)-1]
            cur.next = Node(value)
            # appending a circular link list, put the last slot.next to the head slot
            if self.isCircular:
                cur.next.next = self.head
            # for a doubly list, when appending, put the new node.prev to the last slot
            if self.isDoubly:
                cur.next.prev = cur

        # for a circular, when append, the self[self.get_size()] is head
        elif self[index] == self.head and self.isCircular:
            # go back to previous slot to insert a new node
            cur = self[index-1]
            # store the node( self.head)
            temp = cur.next
            # connect the new node to the cur
            cur.next = Node(value)
            # for a doubly, should connect the prev with new node.prev
            if self.isDoubly:
                cur.next.prev = cur
            # link the new node to the last( the head) and form new circular
            cur.next.next = temp
            # for a doubly circular
            if self.isDoubly:
                temp.prev = cur.next
        else:
            # index of the last element
            n = self.get_size()-1
            # the size after adding one element
            k = self.get_size()+1
            # shifting property: when adding, coming from the back and every element after the index move one forward
            while n >= index:
                # only allow k elements in the list, if equal, we going to change the slot after the n position to the
                # shifted one(n)
                if self.get_size() >= k:
                    temp = self[n].next.next
                    self[n].next = Node(self[n].data)
                    self[n].next.next = temp
                    if self.isDoubly:
                        temp.prev = self[n].next
                # at first, the length is one less than K, so copy a slot to the back
                else:
                    # store the last slot.next
                    temp = self[n].next
                    # copy a slot of the last slot
                    self[n].next = Node(self[n].data)
                    # link the new slot to the last slot.next
                    self[n].next.next = temp
                    if self.isDoubly:
                        temp.prev = self[n].next
                n -= 1
            # after shifting all the slots behind the index one slot back
            temp = self[index-1].next.next
            # insert the new node with the value to the index position
            self[index-1].next = Node(value)
            self[index-1].next.next = temp
            # for a doubly, link the new slot to its previous slot
            if self.isDoubly:
                temp.prev = self[index-1].next

    def __delitem__(self, index):
        # with the finished setitem, it is easy to use self[index] = value to change node
        if self.head == None:
            return
        # when it is in head, just use head.next = head.next.next to skip the slot
        elif index == 0:
            self.head = self.head.next

        # when it is last slot
        elif index == self.get_size()-1:
            cur = self[self.get_size()-2]
            cur.next = None
        # when it is among the slots
        else:
            # shift every slot behind the index one slot forward, covering the purpose index
            for i in range(index+1, self.get_size()):
                self[i-1].data = self[i].data
            # after shifting, fill the last slot with None so that it won't repeat
            self[self.get_size()-2].next = None

    def append(self, value):
        # a output to check whether the append slot is self.head, useful to debug circular
        print("app slot:", self.get_size(), "destination:", self[len(self)])
        self[self.get_size()] = value

    def extend(self, arr):
        # call append
        for i in arr:
            self.append(i)

    def remove(self, value):
        # traverse the list, find the value, and skip the node
        p = self.head
        while p.next:
            if p.next.data == value:
                p.next = p.next.next
                return
            p = p.next

    def argwhere(self, value):
        # set a counter to see how many repetitive value
        count = 0
        cur = self.head
        while cur != None:
            if cur.data == value:
                count += 1
            cur = cur.next
        # create a StaticArray to store the space and index
        ret = StaticArray(count)
        cur = self.head
        index = 0
        # traverse with index and store it to StaticArray once found
        while cur != None:
            if cur.data == value:
                ret.append(index)
            index += 1
            cur = cur.next
        return ret

    def __len__(self):
        # when it is circular, when it meets the self.head for a second time, stop traversing
        if self.isCircular:
            cur = self.head
            # counter
            count = 0
            # times of meeting the head
            meet = 0
            while cur != None:
                cur = cur.next
                count += 1
                if cur == self.head:
                    meet += 1
                    # end condition
                    if meet == 1:
                        return count
            return count
        # when it is not circular, just traverse
        count = 0
        cur = self.head
        while True:
            if cur is None:
                break
            count += 1
            cur = cur.next
        return count

    def get_size(self):
        return len(self)

    def __eq__(self, arr):
        count = 0
        p = self.head
        # check the type first
        if type(arr) == LinkedList:
            # traverse and see whether the value matches
            while p.next:
                if p == arr[count]:
                    count += 1
                    p = p.next
                    continue
                else:
                    return False
            return True
        else:
            return False

    def __repr__(self):
        node = self.head
        ret = node.__repr__()
        while node != None :
            node = node.next
            # a recursion to group all the nodes
            ret += "->" + node.__repr__()
            if node == self.head:
                break
        return ret

    def __iter__(self):
        if self.head == None:
            yield None
        cur = self.head
        # traverse and yield
        while True:
            yield cur
            cur = cur.next
            if cur == None :
                break



