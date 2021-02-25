from math import exp

import scipy.special

from plot import orange
from plot_util import PlotWrapper


def bin(n, m):
    return scipy.special.binom(n, m)


def bernoulli(p, q, n, m):
    ans = bin(n, m) * (p ** m) * (q ** (n - m))
    # print(ans)
    return ans


class ExpectedDelta:
    def __init__(self, r, n):
        self.r = r
        self.n = n

    @property
    def p(self):
        return self.r / self.n

    @property
    def q(self):
        return 1 - self.p

    def expected_delta(self, I: int):
        ans = 0.
        for k in range(0, I + 1):
            loc = bin(k, self.n - I)
            for delta in range(1, I - k + 1):
                loc += delta * bernoulli(self.p, self.q, I, k + delta)
            ans += loc * bernoulli(self.p, self.q, self.n - I, k)
        return ans

    def lower_predicted(self, I: int):
        c = I / self.n
        return exp(c - 1) * c
        # return self.p * I * (self.q ** (self.n - I))

    def upper_prediction(self, I: int):
        c = I / self.n
        return self.lower_predicted(I) + exp(-1) * (1 - c) * c ** 2
        # 2 * I^2 * (n - I) * p^3 * q^{n-3}
        # return self.lower_predicted(I) + 2 * I ** 2 * (self.n - I) * (self.p ** 3) * self.q ** (self.n - 3)


def plot_expected_delta(n=100, r=1, upper=50):
    lower = 3
    model = ExpectedDelta(r, n)
    xs = range(lower, upper + 1)
    y1s = list(map(model.expected_delta, xs))
    y2s = list(map(model.lower_predicted, xs))
    y3s = list(map(model.upper_prediction, xs))
    # print(y1s)
    # print(y2s)

    plot = PlotWrapper(lower, upper)
    plot.add_line(xs, y1s, "mu", "b")
    plot.add_line(xs, y2s, "lower", orange)
    plot.add_line(xs, y3s, "upper", "r")
    plot.show()


def plot_bernoulli(n=100, r=1):
    upper = n
    p = r / n
    q = 1 - p
    xs = range(upper + 1)
    ys = list(map(lambda m: bernoulli(r / n, q, n, m), xs))
    print(sum(ys))
    wrapper = PlotWrapper(0, upper)
    wrapper.add_line(xs, ys, "Bernoullit", orange)
    wrapper.show()


def other():
    print((99 / 100) ** 90)


if __name__ == '__main__':
    plot_expected_delta(1000, 1, 100)
