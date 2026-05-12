import pickle
import numpy as np


def get_moving_average(arr, window, mode="valid"):

    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=mode
    ) / window

def save_q_table(data, path):
    with open(path, "wb") as f:
        pickle.dump(dict(data), f)
