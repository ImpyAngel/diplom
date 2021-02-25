import os

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable, List, Optional, Union

import random
import math

from matplotlib.axes import Axes

from plot_util import PlotWrapper

orange = '#FF8000'


# True if \ksi <= r/n
def random_u(r: int, n: int) -> bool:
    return random.randint(1, n) <= r


def find_new_optimum(r: int, n: int) -> int:
    i = 0
    while True:
        i += 1
        if random_u(r, n):
            j = 0
            for k in range(1, n):
                if random_u(n - r, n):
                    j += 1
            if j == n - 1:
                return i


def find_new_optimum_averaged(r: int, n: int, iterations: int) -> Tuple[float, float]:
    return averaged(lambda: find_new_optimum(r, n), iterations)


def t_bigger_than_new_optimum(T: int, r: int, n: int, iteration: int) -> float:
    i = 0

    for j in range(iteration):
        if find_new_optimum(r, n) < T:
            i += 1
    return i / iteration


def pt_bigger_than_new_optimum_averaged(T: int, r: int, n: int, upper_iterations: int, lower_iterations: int) \
        -> Tuple[float, float]:
    return averaged(lambda: t_bigger_than_new_optimum(T, r, n, lower_iterations), upper_iterations)


def averaged(f: Callable[[], float], iterations: int) -> Tuple[float, float]:
    xs: List[float] = list()
    for i in range(iterations):
        xs.append(f())
    e_x = np.mean(xs, axis=0)
    std_x = np.std(xs, axis=0)
    return e_x, std_x


def mu_upper(r, n) -> float:
    return math.exp(r) * (n - r) / r


def mu_lower(r, n) -> float:
    return math.exp(r) * (n - r) / r * (n ** 2 - (n + 1) * (r ** 2)) / (n ** 2 - r ** 2)


def plot_T_bigger_X(r, n, sqrt_iterations):
    name = f'plot_T_bigger_X_{n}_{r}_{sqrt_iterations}_{random.randint(1, n)}'

    ts = range(r + 1, n * 10, 10)

    fig = plt.figure()

    y_pairs = list(map(lambda t: pt_bigger_than_new_optimum_averaged(t, r, n, sqrt_iterations, sqrt_iterations), ts))
    yt = np.transpose(y_pairs)
    e = yt[0]
    sigma = yt[1]
    plt.plot(ts, e, 'b-')
    plt.plot(ts, e + sigma, 'y:')

    plt.xlabel('t', fontsize='xx-large')
    plt.ylabel('P(T > X)', fontsize='xx-large')
    plt.grid(True)

    ax = plt.gca()

    mu = math.exp(r) * n / r
    ax.vlines(mu, 0, 1, color='#FF8000')

    ax.legend(("Average", "σ", "μ"), loc='upper right')

    plt.plot(ts, e - sigma, 'y:')

    ax.text(0.1, 0.1, f'n = {n}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize='xx-large',
            bbox={'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.8})

    plt.savefig(os.path.join('grafics', (name + '.png')), dpi=200)
    plt.close(fig)


def plot_mu(r, n_max, iteration=1000):
    name = f'plot_mu_{r}_iter_{iteration}_{random.randint(1, n_max)}'

    fig = plt.figure()

    x = range(r ** 2 * 3, n_max + 1)
    y_pairs = list(map(lambda n_cur: find_new_optimum_averaged(r, n_cur, iteration), x))
    y_mu_upper = list(map(lambda n_cur: mu_upper(r, n_cur), x))
    y_mu_lower = list(map(lambda n_cur: mu_lower(r, n_cur), x))
    yt = np.transpose(y_pairs)
    e = yt[0]
    sigma = yt[1]
    plt.plot(x, e, 'b')
    # plt.plot(x, e + sigma, 'y:')
    plt.plot(x, y_mu_upper, 'r')
    plt.plot(x, y_mu_lower, orange)

    ax = plt.gca()

    ax.legend(("averaged",
               # "σ",
               "μ upper", "μ lower"), loc='upper right')

    # plt.plot(x, e - sigma, 'y:')

    plt.xlabel('n', fontsize='xx-large')
    plt.ylabel('μ', fontsize='xx-large')
    plt.grid(True)

    ax.text(0.1, 0.1, f'r = {r}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize='xx-large',
            bbox={'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.8})

    # plt.show()
    plt.savefig(os.path.join('grafics', (name + '.png')), dpi=200)
    plt.close(fig)


def plot_example():
    x = np.linspace(0, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.grid(True)
    plt.show()


class OneMax:
    def __init__(self, n: int, r: int, T: Optional[int]):
        self.n = n
        self.r = r
        self.T = T
        self.cur_vector = self.init_vector()
        self.max_vector = self.init_vector()
        self.i = 0

    def init_vector(self) -> List[bool]:
        return np.random.choice([False, True], size=(self.n,), p=[1. / 2, 1. / 2])

    def d(self) -> int:
        return self.vector_d(self.cur_vector)

    def vector_d(self, vector: List[bool]) -> int:
        ans = 0
        for i, x in enumerate(vector):
            if x == self.max_vector[i]:
                ans += 1
        return ans

    def emulate_p(self) -> bool:
        return random_u(self.r, self.n)

    def change_one_bit_in_max(self):
        i = random.randint(0, self.n - 1)
        self.max_vector[i] = not self.max_vector[i]

    def mutate_cur_vector(self) -> List[bool]:
        new_vector = list()
        for bit in self.cur_vector:
            new_vector.append(not bit if self.emulate_p() else bit)
        return new_vector

    def time(self) -> int:
        return self.start_algorithms(None, False)

    def time_with_stop(self) -> int:
        max_iter = self.n * round(math.log2(self.n)) * 20
        ans = self.start_algorithms(max_iter * 10, False)
        if ans > max_iter + 2:
            print(f'problem with {self.n} {self.T} d is {self.d()}')
            return ans
        return ans

    def ds(self, iterations) -> List[int]:
        return self.start_algorithms(iterations, True)

    def start_algorithms(self, max_iterations: Optional[int], need_ds: bool) -> Union[int, List[int]]:
        i = 0
        ds = list()
        while self.d() < self.n:
            if max_iterations is not None:
                if i >= max_iterations:
                    break
            if need_ds:
                ds.append(self.d())
            if self.T is not None and (i % self.T) == 0:
                if need_ds:
                    ds.append(self.d())
                self.change_one_bit_in_max()
            i += 1
            new_vector = self.mutate_cur_vector()
            if self.d() <= self.vector_d(new_vector):
                self.cur_vector = new_vector
        if need_ds:
            return ds
        else:
            return i


def time(n: int, r: int, T: Optional[int]) -> int:
    cur_time = OneMax(n, r, T).time_with_stop()
    return cur_time


def averaged_time(r: int, n: int, T: int, iterations: int) -> Tuple[float, float]:
    print(T)
    return averaged(lambda: time(n, r, T), iterations)


def plot_time(r: int, n: int):
    name = f'plot_time_{n}_as_{random.randint(1, n)}'
    iterations = 400

    lower_bound = round(n * 2.5)
    upper_bound = n * 5
    step = round(n / 8)

    ts = range(lower_bound, upper_bound, step)

    fig = plt.figure()

    y_pairs = list(map(lambda t: averaged_time(r, n, t, iterations), ts))
    yt = np.transpose(y_pairs)
    e = yt[0]
    sigma = yt[1]
    plt.plot(ts, e, 'b-')
    plt.xlabel('t', fontsize='xx-large')
    plt.ylabel('time', fontsize='xx-large')
    plt.grid(True)
    ax = plt.gca()

    without_t, _ = averaged_time_without_T(n, r, iterations * 10)
    plt.plot(ts, e - sigma, 'y:')
    ax.hlines(without_t, lower_bound, upper_bound, color='#FF8000')
    ax.legend(("Average time with T", "σ", "Without T"), loc='upper right')
    plt.plot(ts, e + sigma, 'y:')

    ax.text(0.1, 0.1, f'n = {n}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize='xx-large',
            bbox={'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.8})
    plt.savefig(os.path.join('grafics', (name + '.png')), dpi=200)
    plt.close(fig)


def averaged_time_without_T(n: int, r: int, iterations: int) -> Tuple[float, float]:
    return averaged(lambda: time(n, r, None), iterations)


def plot_d(r: int, n: int, T: Optional[int], iterations=2000):
    name = f'plot_d_{n}_{T}_{iterations}{random.randint(1, n)}'

    ds = OneMax(n, r, T).ds(iterations)
    x = range(0, len(ds))
    y = ds

    fig = plt.figure()
    plt.plot(x, y)
    plt.grid(True)

    ax = plt.gca()
    plt.xlabel('Iteration', fontsize='xx-large')
    plt.ylabel('d', fontsize='xx-large')

    ax.text(0.1, 0.1, f'T = {T}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize='xx-large',
            bbox={'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.8})
    plt.savefig(os.path.join('grafics', (name + '.png')), dpi=200)
    plt.close(fig)


def plot_d_averaged(r: int, n: int, T: Optional[int], upper_iterations=2000, down_iteration=20):
    acc = [[] for i in range(upper_iterations)]
    for i in range(down_iteration):
        ds = OneMax(n, r, T).ds(upper_iterations)
        print(f"for {n} - {i}")
        for j in range(upper_iterations):
            acc[j].append(n if len(ds) < j else ds[j])
    ys = list()
    for i in acc:
        ys.append(np.mean(i))
    xs = list(range(upper_iterations))
    plot = PlotWrapper(0, upper_iterations)
    plot.add_line(xs, ys, f"average by {down_iteration}", orange)
    name = f'plot_d_averaged_{n}_{r}_{T}_{upper_iterations}_{random.randint(1, n)}'
    plot.saveToFile(name)


def plot_all():
    # print("mu started")

    print("d started")
    plot_d(1, 100, 10)
    plot_d(1, 200, 10)
    plot_d(1, 400, 15)

    # print("time r = 1 started")
    # plot_time(1, 10)
    # plot_time(1, 20)
    # plot_time(1, 30)
    #
    # print("time r = 2 started")
    # plot_time(2, 10)
    # plot_time(2, 20)
    # plot_time(2, 30)
    #
    # print("T bigger X started")
    # plot_T_bigger_X(1, 20, 200)
    # plot_T_bigger_X(1, 40, 100)
    # plot_T_bigger_X(1, 100, 40)


if __name__ == '__main__':
    # plot_mu(1, 200, 3100)
    # plot_mu(2, 200, 3100)
    # plot_mu(5, 200, 3100)
    # plot_T_bigger_X(1, 100, 20)
    # print(100)
    # plot_d_averaged(1, 100, 20)

    print(200)
    plot_d_averaged(1, 200, 20)

    print(400)
    plot_d_averaged(1, 400, 20, upper_iterations=4000)

    print(1000)
    plot_d_averaged(1, 1000, 20, upper_iterations=10000)
