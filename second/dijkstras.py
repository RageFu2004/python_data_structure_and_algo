from graph import *
import copy

# for a recursion way to implement a dijkstras, we need to prepare lists out of the function
first_time = [True]
# |used for condition where if it is first time running with the origin src
dest_lst = {}
# |prepare a list that we return during each run
start = [0]
# |store the origin src to check whether we have traversed all the possible shorter ways and return to origin
store_the_path = [0]
# |used to calculate, when we compare the new path and existing path, add the paths it comes from the origin


# for my prim, I write as the PA2.pdf tells me, but now I have a clear idea, I try recursion, not priority queue
def dijkstras(graph, src):
    # check if first time running
    if first_time[0] == True:
        # initialize the dest list with extremely big path
        for i in graph.V():
            dest_lst[i] = 10000
        # set the path src to src = 0
        dest_lst[src] = 0
        #  record the origin node
        start[0] = src
        # make the first_time to False so that the origin and the return list will not be changed
        first_time[0] = False
    # recursion begins
    for source in graph.adj:
        if source[0] == src:
            for dests in source[1:]:
                # regular case : if the path between a to the node is shorter than recorded, we use that node as new src
                if dests[1] + store_the_path[0] < dest_lst[dests[0]]:
                    # store the new path length to the dest_lst
                    dest_lst[dests[0]] = dests[1] + store_the_path[0]
                    # also record the length of paths that have come so far
                    store_the_path[0] = dests[1] + store_the_path[0]
                    # recursion
                    dijkstras(graph, dests[0])
                    # a detail : when you retrieve from a node where no more shorter can be found, you need to change
                    # the total path come so far back to its status before this recursion
                    store_the_path[0] -= dests[1]
            else:
                # end case : when go back to the origin, set all the lists back and return the result list
                if src == start[0]:
                    k = copy.deepcopy(dest_lst)
                    dest_lst.clear()
                    first_time[0] = True
                    store_the_path[0] = 0
                    start[0] = 0
                    return k
                else:
                    # return : if you cannot find any valid shorter path, just return to the last node
                    return


def runDijkstras():
    return_dict = {}
    # from this point, I copy from my prim.py because they are totally the same
    visted_key = []
    location = {"Wisconsin, USA": [44.5, -89.5], "West Virginia, USA": [39.0, -80.5],
                "Vermont, USA": [44.0, -72.699997], "Texas, USA": [31.0, -100.0], "South Dakota, US": [44.5, -100.0],
                "Rhode Island, US": [41.742325, -71.742332], "Oregon, US": [44.0, -120.5],
                "New York, USA": [43.0, -75.0], "New Hampshire, USA": [44.0, -71.5], "Nebraska, USA": [41.5, -100.0]}
    g = Graph()
    # first, just append all the vertexes
    for v in location:
        g.addVertex(v)
    # make connections between every two existing nodes
    for vertex in location:
        visted_key.append(vertex)
        for dest in location:
            if dest not in visted_key:
                # calculate the weight, but now use the float, because test.py will check between 2 int
                weight = float(((location[vertex][0] - location[dest][0]) ** 2 + (
                            location[vertex][1] - location[dest][1]) ** 2) ** 0.5)
                # add the undirected edge with this weight
                g.addUndirectedEdge(vertex, dest, weight)
    # change the return_dict to the demanded form
    for vertexes in g.V():
        return_dict[vertexes] = dijkstras(g, vertexes)
    return return_dict

