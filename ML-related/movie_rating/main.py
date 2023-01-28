import numpy as np
import tensorflow as tf
from tensorflow._api.v2.compat.v1 import keras
from collab_filter_cost import *
from file_open import *
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def main():
    # X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
    Y, R = load_ratings_small()
    #print(Y.shape, R.shape)
    user_rate = np.zeros(Y.shape[0])
    user_rate_bool = np.zeros(Y.shape[0])
    with open('./data/small_movie_list.csv') as f:
        movie_set = f.read().split('\n')
        #print(len(movie_set))
    times = int(input('How many movies you want to rate:'))
    for i in range(times):
        movie_index = random.randint(0, len(movie_set)-2)
        rate = int(input('how you rate{0} from 1-5'.format(movie_set[movie_index].split(',')[3])))
        user_rate[movie_index] = rate
        user_rate_bool[movie_index] = 1
    my_rated = [i for i in range(len(user_rate_bool)) if user_rate_bool[i] > 0]
    Y = np.column_stack((Y, user_rate))
    R = np.column_stack((R, user_rate_bool))
    #print(Y.shape, R.shape)
    Ynorm, Ymean = normalizeRatings(Y, R)

    num_movies, num_users = Y.shape
    num_features = 100

    tf.random.set_seed(1234)
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    iterations = 200
    lambda_ = 1
    for i in range(iterations):
        # Use TensorFlow’s GradientTape
        with tf.GradientTape() as tape:
            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        grads = tape.gradient(cost_value, [X, W, b])

        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        # Log periodically.
        if i % 20 == 0:
            print(f"Training loss at iteration {i}: {cost_value:0.1f}")

    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    # restore the mean
    pm = p + Ymean
    #print(pm.shape)

    my_predictions = pm[:, 443]

    # sort predictions
    ix = tf.argsort(my_predictions, direction='DESCENDING')

    for i in range(17):
        j = ix[i]
        if j not in my_rated:
            print(f"Predicting rating {my_predictions[j]:0.2f} for movie {movie_set[j].split(',')[3]}")

    print('\n\nOriginal vs Predicted ratings:\n')
    for i in range(len(user_rate)):
        if user_rate[i] > 0:
            print(f"Original {user_rate[i]}, Predicted {my_predictions[i]:0.2f} for {movie_set[i].split(',')[3]}")


if __name__ == '__main__':
    main()
