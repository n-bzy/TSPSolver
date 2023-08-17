from config import HNConfig as Config
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import pandas as pd
import util

window_size = 5
dpi = 100
eps = 500
record_moment = np.arange(0, eps, 10)
record = False
delta_t = 0.01
noise = 0.001
u_0 = 0.02
param_a = 1.0
param_b = 1.0
param_c = 2.0
param_d = 0.5


@np.vectorize
def sigmoid(input: float) -> float:
    sigmoid_range = 34.538776394910684
    if input <= -sigmoid_range:
        return 1e-15
    if input >= sigmoid_range:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + np.exp(-input / u_0))

# NEW
@np.vectorize
def tanh(input: float) -> float:
    lamb = 0.5
    return 1/2*(1+np.tanh(input/u_0))


def kronecker_delta(i: int, j: int) -> float:
    if i == j:
        return 1.0
    return 0.0


def calc_weight_matrix(city_array: np.array) -> np.array:
    city_num: int = city_array.shape[0]
    # number of neurons
    n: int = city_num ** 2
    tmp: np.array = np.zeros((n, n))
    for s0 in range(n):
        x: int = int(s0 / city_num)
        i: int = s0 % city_num
        for s1 in range(n):
            y: int = int(s1 / city_num)
            j: int = s1 % city_num
            # distance between every pair of cities
            dxy: float = util.dist(city_array[x, :], city_array[y, :])
            tmp[s0, s1] = (
                -param_a * kronecker_delta(x, y) * (1.0 - kronecker_delta(i, j))
                - param_b * kronecker_delta(i, j) * (1.0 - kronecker_delta(x, y))
                - param_c
                - param_d
                * dxy
                * (
                    kronecker_delta(j, (i - 1) % city_num)
                    + kronecker_delta(j, (i + 1) % city_num)
                )
            )
    return tmp


def calc_bias(city_array: np.array) -> np.array:
    city_num: int = city_array.shape[0]
    n: int = city_num ** 2
    tmp: np.array = param_c * city_num * np.ones(n)
    return tmp


def update_inner_vals(
    nodes_array: np.matrix,
    inner_vals: np.matrix,
    weight_matrix: np.matrix,
    biases: np.matrix,
) -> np.matrix:
    tau = 1.0
    asdf: np.matrix = np.matmul(weight_matrix, nodes_array)
    delta: np.matrix = (-inner_vals / tau + asdf + biases) * delta_t
    return inner_vals + delta


def hp_begin(
    inner_vals_array: np.matrix,
    nodes_array: np.matrix,
    weights_matrix: np.matrix,
    biases_array: np.matrix,
) -> None:
    if record:
        dir_name: str = util.make_directory(Config)
        for i in range(eps):
            if i in record_moment:
                filename: str = "iteration-" + str(i) + ".png"
                file_path: str = dir_name + filename
                plt.savefig(file_path)
            inner_vals_array = update_inner_vals(
                nodes_array, inner_vals_array, weights_matrix, biases_array
            )
            nodes_array = tanh(inner_vals_array)
            plt.title("iteration=" + str(i + 1))
            mat_visual.set_data(np.reshape(nodes_array, (city_num, city_num)))
            plt.pause(0.01)
    else:
        i = 1
        for i in range(eps):
            inner_vals_array = update_inner_vals(
                nodes_array, inner_vals_array, weights_matrix, biases_array
            )
            nodes_array = tanh(inner_vals_array)
            plt.title("iteration=" + str(i))
            mat_visual.set_data(np.reshape(nodes_array, (city_num, city_num)))
            i += 1
            plt.pause(0.01)


if __name__ == "__main__":
    if Config.read_file:
        np_cities = np.genfromtxt(Config.file_path + Config.city_file, delimiter=",")
        city_num = np_cities.shape[0]
        figsize = (window_size, window_size)
    else:
        city_num = Config.city_num
        # “continuous uniform” distribution random
        np_cities = np.random.random((city_num, 2))
        center_x = 0.5
        center_y = 0.5
        figsize = (window_size, window_size)
    # Initialize random starting values
    inner_vals = np.matrix((np.random.random((city_num ** 2)) - 0.5) * noise).T
    # outputs
    nodes = np.matrix(tanh(inner_vals))
    # Calculate weights and save them
    weights = np.matrix(calc_weight_matrix(np_cities))
    df = pd.DataFrame(weights)
    df.to_csv("weigths.csv", header=False, index=False)
    # Calculate biases
    biases = np.matrix(calc_bias(np_cities)).T
    fig = plt.figure(figsize=figsize, dpi=dpi)
    mat_visual = plt.matshow(
        np.reshape(nodes, (city_num, city_num)),
        fignum=0,
        cmap=cm.Greys,
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
    )
    fig.colorbar(mat_visual)
    plt.title("iteration=" + str(0))
    plt.pause(0.0001)
    hp_begin(inner_vals, nodes, weights, biases)
