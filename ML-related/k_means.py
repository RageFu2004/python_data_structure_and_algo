import random
import math
import numpy as np
import matplotlib.pyplot as mp
def k_mean(split):
    xs = []
    ys = []
    dot = []
    split_axis = []
    centroidx = []
    centroidy = []

    with open('test.txt', 'r') as k:
        for sets in k.read().split('\n'):
            xs.append(int(sets.split()[0]))
            ys.append(int(sets.split()[1]))
            dot.append([int(sets.split()[0]), int(sets.split()[1])])
    belong = [0] * len(dot)
    for i in range(split):
        ran_x = random.randint(min(xs), max(xs))
        ran_y = random.randint(min(ys), max(ys))
        split_axis.append([ran_x, ran_y])
        centroidx.append(ran_x)
        centroidy.append(ran_y)

    mp.ion()
    mp.subplots()
    for train_tims in range(100):
        mp.clf()
        mp.scatter(np.array(xs), np.array(ys), color='blue')
        mp.scatter(np.array(centroidx), np.array(centroidy), color='red')
        mp.pause(0.0001)

        for ind, dots in enumerate(dot):
            x = dots[0]
            y = dots[1]
            comp = []
            min_dist = math.inf
            for index, centroid in enumerate(split_axis):
                centroid_x = centroid[0]
                centroid_y = centroid[1]
                comp.append([index, (x-centroid_x)**2 + (y-centroid_y)**2])
            for distance in comp:
                if distance[1] < min_dist:
                    min_dist = distance[1]
                    belong[ind] = distance[0]
        for index, centroid in enumerate(split_axis):
            new_x, x_count, new_y, y_count = 0, 0, 0, 0
            for index2, ind in enumerate(belong):
                if ind == index:
                    new_x += dot[index2][0]
                    x_count += 1
                    new_y += dot[index2][1]
                    y_count += 1
            split_axis[index] = [new_x/x_count, new_y/y_count]
            centroidx[index] = new_x/x_count
            centroidy[index] = new_y/y_count
    mp.ioff()
    mp.show()
k_mean(3)








