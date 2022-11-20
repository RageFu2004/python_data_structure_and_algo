from graph import *


def sort(dict):
    lst = []
    for keys in dict:
        lst.append(dict[keys])
    # sort the value of all the keys
    lst.sort()
    smallest = lst[0]
    for dest in dict:
        # cannot keep track of the value's key, so just traverse to find
        if dict[dest] == smallest:
            return [dest, smallest]


def search(lst, dest, weight):
    for i in lst.adj:
        for j in i[1:]:
            if j[0] == dest and j[1] == weight:
                return i[0]


def prim(graph):
    # set up a final list that we return a list of [src,dest,weight]
    final_lst = []
    # prepare two lists, one for the visited, one for making the remaining size smaller
    vertex_lst = graph.V()
    # the first source we begin should be added to the visited and removed from the whole vertexes
    visit_lst = [graph.adj[0][0]]
    vertex_lst.remove(graph.adj[0][0])
    # if all the vertex list is empty, we know it finishes its greedy and all the nodes have been visited
    while vertex_lst:
        # create a prepare list to compare the shorter path
        prepare_lst = {}
        # traverse all the elements out to find all their connections
        for i in visit_lst:
            # find it in g.adj
            for j in graph.adj:
                if j[0] == i:
                    # find all its neighbors
                    for dest in j[1:]:
                        # only if it is not visited, we can append this vertex and its path weight to the prepare list
                        if dest[0] not in visit_lst:
                            # a detail: do not ever trust traversing through the dict's keys, it is VERY STUPID and
                            # COSTLY in this case
                            key_lst = []
                            for item in prepare_lst:
                                key_lst.append(item)
                            # examine whether the dest is already in the prepare list
                            if dest[0] in key_lst:
                                # if so, we should compare the existing path's weight and the new one, choosing the
                                # shorter one
                                if dest[1] < prepare_lst[dest[0]]:
                                    prepare_lst[dest[0]] = dest[1]
                            else:
                                # if it is not repetitive, just append
                                prepare_lst[dest[0]] = dest[1]
        # self-written sorting method
        greed = sort(prepare_lst)
        # find the origin using the unique dest and its connection path's weight
        origin = search(graph, greed[0], greed[1])
        # change the vertex list to move it toward the end case
        vertex_lst.remove(greed[0])
        # append the visited to the visit list
        visit_lst.append(greed[0])
        # make these src, dest, weight to the final list
        final_lst.append([origin, greed[0], greed[1]])

    else:
        # when you jump out of the while loop, time to end the function and return the wanted
        return final_lst


def runPrim():
    # traversing a dict's keys and avoid repetition is a costly thing
    visted_key = []
    location = {"Wisconsin, USA": [44.5, -89.5], "West Virginia, USA": [39.0, -80.5], "Vermont, USA": [44.0, -72.699997], "Texas, USA": [31.0, -100.0], "South Dakota, US": [44.5, -100.0],"Rhode Island, US": [41.742325,-71.742332],"Oregon, US": [44.0,-120.5],"New York, USA": [43.0,-75.0],"New Hampshire, USA": [44.0,-71.5],"Nebraska, USA": [41.5,-100.0]}
    g = Graph()
    # first, just append all the vertexes
    for v in location:
        g.addVertex(v)
    # make connections between every two existing nodes
    for vertex in location:
        visted_key.append(vertex)
        for dest in location:
            if dest not in visted_key:
                # calculate the weight
                weight = int(((location[vertex][0] - location[dest][0])**2 + (location[vertex][1] - location[dest][1])**2)**0.5)
                # add the undirected edge with this weight
                g.addUndirectedEdge(vertex, dest, weight)
    
    return prim(g)

if __name__ == "__main__":
    print("by RageFu")

