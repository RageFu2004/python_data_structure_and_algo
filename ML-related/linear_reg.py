import numpy as np
import matplotlib.pyplot as mp

# linear regression
def linear_regression(times):
    xs = []
    ys = []
    plot_x = []
    plot_y = []
    with open('test.txt', 'r') as e:
        for sets in e.read().split('\n'):
            xs.append([1, int(sets.split()[1])])
            plot_x.append(int(sets.split()[1]))
            ys.append(int(sets.split()[0]))
            plot_y.append(int(sets.split()[0]))
    test_xs = np.array(xs)
    test_ys = np.array(ys)
    result = gradient_descent(test_xs, test_ys, times)
    mp.figure('result')
    mp.scatter(np.array(plot_x), np.array(plot_y))
    now_x = np.linspace(1, int(max(plot_x)*1.2), 100)
    mp.plot(now_x, now_x*result[1]+result[0])
    mp.show()
    return result
def gradient_descent(xs, ys, util_time):
    ws = xs.shape[1]
    w_org = []
    for i in range(ws):
        w_org.append(10)
    for time in range(util_time):
        tempo = []
        for index, w in enumerate(w_org):
            # set learning rate here
            storage = w - (1/ws) * (0.001) * diff(xs, w_org, ys, index)
            tempo.append(storage)
        w_org = tempo
    return w_org
def diff(arr1, w, arr2, n):
    flist = []
    times = arr1.shape[0]
    for i in range(times):
        flist.append((np.dot(arr1[i, :], w) - arr2[i])*arr1[i, n])
    return sum(flist)

linear_regression(10000)
