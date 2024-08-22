import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_noise(shape: (int, int), mu, sigma):
    """
    Creates an array of thermal noise with mean mu and variance sigma^2
    :param shape: Two-dimensional tuple
    :param mu: mean
    :param sigma: standard deviation
    :return: created thermal noise
    """
    array = np.clip(np.random.normal(loc=mu, scale=sigma, size=shape), a_min=-3 * sigma, a_max=3 * sigma)
    return array


def add_target(array, x=None, y=None, power=5., is_label=False):
    """
    Adds a target to an array
    :param is_label:
    :param y:
    :param x:
    :param array: numpy array
    :param power: power of the target
    :return: thermal noise with the target
    """
    if x is None:
        x = np.random.randint(0, array.shape[0])
    if y is None:
        y = np.random.randint(0, array.shape[1])
    array[x, y] = power
    if not is_label:
        array[x + 1, y + 1] = power
        array[x + 1, y - 1] = power
        array[x - 1, y + 1] = power
        array[x - 1, y - 1] = power
        array[x, y + 1] = power
        array[x, y - 1] = power
        array[x - 1, y] = power
        array[x + 1, y] = power
    return array


if __name__ == "__main__":

    params_path = "params_windows.json"
    with open(params_path, "r") as f:
        dic = json.load(f)
    PATH = dic["general"]["data_path"]
    PATH = f"{PATH}"

    n_images = 50000
    sigma = .5  # Std of noise
    for i in tqdm(range(n_images)):
        dava_card = create_noise((256, 256), mu=0, sigma=sigma)
        label = np.zeros((256, 256))
        for j in range(np.random.randint(1, 6)):
            x = np.random.randint(10, 250)
            y = np.random.randint(10, 250)
            dava_card = add_target(dava_card, x, y,
                                   power=float(np.random.normal(loc=1, scale=.7)))
            label = add_target(label, x, y, power=1, is_label=True)
        dava_card /= (3 * sigma)  # Normalizing between 0 and 1
        np.save(f"{PATH}/carte{i}.npy", dava_card)
        np.save(f"{PATH}/label{i}.npy", label)
