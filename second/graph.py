import copy


class Graph():
    def __init__(self):
        # use a list to implement graph in this case
        self.adj = []
        # prepare the storing list used for traversing and searching later
        self.dfslst = []
        self.visited = []
        self.max = [0]
        self.begin = [True]
        self.cyclic = False
        self.times = [0]

    def addVertex(self, data):
        self.adj.append([data])

    def removeVertex(self, data):
        # use enumerate to find the value and its index
        for i, v in enumerate(self.adj):
            if v[0] == data:
                self.adj.pop(i)
        # do not forget to remove the connections it has between other vertexes
        for item in self.adj:
            for index, dest in enumerate(item[1:]):
                if dest[0] == data:
                    item.pop(index+1)

    def addEdge(self, src, dest, weight = 1):
        # search for the vertex during traversing
        for i in self.adj:
            if i[0] == src:
                # just append the [dest, weight] to the vertex
                i.append([dest, weight])

    def addUndirectedEdge(self, A, B, weight = 1):
        # it is just addEdge to both of the vertexes with the other one
        self.addEdge(A, B, weight)
        self.addEdge(B, A, weight)

    def removeEdge(self,src,dest):
        # search fot the src
        for i in self.adj:
            if i[0] == src:
                # in src's neighbors, search for the one where dest in
                for index, value in enumerate(i[1:]):
                    if value[0] == dest:
                        i.pop(index+1)

    def removeUndirectedEdge(self, A, B, weight = 1):
        # it is just remove the edge between A and B at the same time
        self.removeEdge(A,B)
        self.removeEdge(B,A)

    def V(self):
        # prepare a list
        lst = []
        # traverse the adj and append the vertexes
        for i in self.adj:
            lst.append(i[0])
        return lst

    def E(self):
        lst = []
        # traverse for vertexes
        for vertex in self.adj:
            # traverse the vertex's neighbor to find the dests
            for edge in vertex[1:]:
                # add them to a list and add the list to the result list( the PA2.pdf says they should be in tuple)
                tup = [vertex[0], edge[0], edge[1]]
                lst.append(tup)
        return lst

    def neighbors(self,value):
        lst = []
        # find the value and its neighbors
        for vertex in self.adj:
            if vertex[0] == value:
                for dest in vertex[1:]:
                    lst.append(dest[0])

        return lst

    # a self-written sort method to sort the dests due to their order
    def sort(self, lst):
        for i in range(1, len(lst)):
            for j in range(i+1, len(lst)):
                if lst[i] > lst[j]:
                    lst[i], lst[j] = lst[j], lst[i]

    '''
    My Style of DFS : make a list to store all the nodes and when it goes back to the root, clear all the list
    '''
    def dft(self, src):
        print(src)
        # at very first, store the root value for the clearing of storing lists later
        if self.begin[0] is True:
            self.max[0] = src
            # an end case : if it is a branch without descendants at beginning, just return the node itself
            for i in self.adj:
                if i[0] == src and len(i) == 1:
                    return [src]
            # set the begin[0] to False so that in recursion this will not change the stored root value
            self.begin[0] = False

        # end case 2 : it does not have any connections to other nodes
        for i in self.adj:
            if i[0] == src and len(i) == 1:
                # make it visited so that in further searching it will not be considered
                self.visited.append(i[0])
                # append the value to the dfs result list
                self.dfslst.append(src)
                # return to the last node to search more nodes
                return
        # normal case : go outward the graph
        for i in self.adj:
            if i[0] == src and i[0] in self.visited:
                self.cyclic = True
                return
            if i[0] == src and i[0] not in self.visited:
                # after being searched, make it to the visited list
                self.visited.append(i[0])
                # add nodes to the dfs list (Queue Mode Traverse (FIFO),
                # if we put this line after the recursion, it will be a stack mode(LIFO))
                self.dfslst.append(src)
                # call a written method to sort the node's neighbors(we want it to be ordered when searching)
                self.sort(i)
                # punchline : recursion to search all the nodes under father node until we reach the single branch
                for item in i[1:]:
                    self.dft(item[0])
                # endcase 3(destination where we get the results: when we go back to the root, all is done,
                # just clear or turn the lists to their origins and return the dfs list
                if src == self.max[0]:
                    # store the dfslst
                    k = self.dfslst
                    self.dfslst = []
                    self.visited = []
                    self.max = [0]
                    self.begin = [True]
                    return k
                # when we finish searching a whole son nondes of a node, just return to its father node
                return

        # end case 2: if cannot find the src, return to the last node
        #return

    def bft(self, src):
        # the return list
        self.bftlst = []
        # the processing queue useful for BFT
        queue = [src]
        # firstly append the root to the result list
        self.bftlst.append(src)
        # if we do not remove the last valid node from the queue, it should be working
        while queue:
            # take out the first element because it is BFT
            n = queue.pop(0)
            # make it visited so that no repetitive visit will be made
            self.visited.append(n)
            # traverse the neighbours of the popped out element
            for i in self.adj:
                # we want it to be an ordered search
                self.sort(i)
                if i[0] == n:
                    for j in i[1:]:
                        # append the valid nodes to the queue, waiting to be searched
                        if j[0] not in self.visited:
                            # append it to the result list
                            self.bftlst.append(j[0])
                            # push it to the queue, waiting to be searched
                            queue.append(j[0])
                            # make it visited
                            self.visited.append(j[0])
        # do not forget to set the visited empty
        self.visited = []
        # return the result
        return self.bftlst

    def isDirected(self):
        # first take a copy of self.adj
        copi = copy.deepcopy(self.adj)
        # delete the commute connections between the two nodes
        for i in copi:
            for dest in i[1:]:
                weight = dest[1]
                for j in copi:
                    if j[0] == dest[0]:
                        for desti in j[1:]:
                            # check if the weight are the same
                            if desti[0] == i[0] and desti[1] == weight:
                                # remove them from each node's neighbor
                                i.remove(dest)
                                j.remove(desti)
        # if it is directed, after removing all the commute connections, there should still be one-way connections
        for item in copi:
            if len(item) > 1:
                return True
        # if it is not, it should be an empty list
        return False

    def recursion(self, src1, lst1):
        # the first time of running, just store the root where we start traversing
        if self.begin[0] is True:
            # record the origin
            self.max[0] = src1
            # make it to False so that it will not be disturbed by recursion
            self.begin[0] = False

        for i in lst1:
            if i[0] == src1:
                # end case(terminal): if we get to the origin of where we start, return True
                if i[0] == self.max[0]:
                    # at first it will meet, so we must count twice
                    self.times[0] += 1
                    if self.times[0] == 2:
                        # set all the defined lists to empty
                        self.max[0] = 0
                        self.begin[0] = True
                        self.times[0] = 0
                        return True
                # when it reaches to the nodes where there is no further connections, return False
                if len(i) == 1:
                    return False
                # regular recursion
                for item in i[1:]:
                    f = item[0]
                    j = i.index(item)
                    # pop the examined edge between the nodes, make sure that the undirected won't be twice been
                    i.pop(j)
                    # pop the edge out in the connected node
                    for j in lst1:
                        if j[0] == f:
                            for thing in j[1:]:
                                if thing[0] == src1:
                                    # remove the edge
                                    j.remove(thing)
                    # recursion, if it is true, it will all the way send True to the last recursion and return True
                    if self.recursion(f, lst1):
                        return True
                # if cannot find the origin, return False
                return False

    def isCyclic(self):
        # first take a copy
        cop1 = copy.deepcopy(self.adj)
        # if it is directed, just use DFT to solve it, in the DFT, if go over a visited point, it is cyclic
        if self.isDirected():
            for i in self.adj:
                self.dft(i[0])
            return self.cyclic
        else:
            # if not directed, use the recursion function to traverse through the graph to see where from any point
            # in the graph we can get one way through all the nodes without repetition
            for fr in self.adj:
                if self.recursion(fr[0], cop1):
                    # if found, it must be a cyclic
                    return True
                else:
                    # if cannot find, set the storing list back
                    self.max[0] = 0
                    self.begin[0] = True
                    self.times[0] = 0
            return False

    def isConnected(self):
        # just check whether there is a vertex that not connected to any vertexes
        for i in self.adj:
            if len(i) == 1:
                return False
        return True

    def isTree(self):
        tree = False
        # if it is both acyclic and connected, it is a tree by definition
        if not self.isCyclic():
                if self.isConnected():
                    tree = True
        return tree

    # self-written dft using stack
    def dft1(self,src):
        final_lst = []
        stack = [src]
        templ = []
        while stack:
            n = stack.pop()
            if n not in final_lst:
                final_lst.append(n)
                for vertex in self.adj:
                    if vertex[0] == n:
                        for edge in vertex[1:]:
                            if edge[0] not in final_lst:
                                templ.append(edge[0])
                        templ.sort(reverse=True)
                        for item in templ:
                            stack.append(item)
                        templ.clear()
        return final_lst


if __name__ == "__main__":
    g = Graph()
    g.addVertex(0)
    g.addVertex(1)
    g.addVertex(2)
    g.addVertex(4)
    g.addEdge(0,1)
    g.addEdge(1,2)
    g.addEdge(0,2)
    g.addEdge(2,4)
    print(g.adj)
    print(g.isCyclic(), g.isDirected())
