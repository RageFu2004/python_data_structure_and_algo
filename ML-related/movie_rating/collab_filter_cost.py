from file_open import *
import numpy as np
import tensorflow as tf
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

def cofi_cost_func(X, W, b, Y, R, lambda_):
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(nm):
            x = X[i, :]
            y = Y[i, j]
            r = R[i, j]
            J += np.square(r * (np.dot(w, x) + b_j - y))
    J += lambda_ * (np.sum(np.square(W)) + np.sum(np.square(X)))
    J = J / 2

    return J


def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J
