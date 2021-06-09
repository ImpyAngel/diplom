import os
from time import strftime, gmtime

import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional

orange = '#FF8000'

from matplotlib.axes import Axes


class PlotWrapper:
    def __init__(self, lower, upper, legends_loc: str = 'upper right'):
        self.upper = upper
        self.lower = lower
        self.funcs: List[Callable[[Axes], None]] = list()
        self.fig = plt.figure()
        self.ax: Axes = plt.gca()
        self.legends: List[str] = list()
        self.legends_loc: str = legends_loc

    def run(self):
        for f in self.funcs:
            f(self.ax)
        if len(self.legends) != 0:
            self.ax.legend(tuple(self.legends), loc=self.legends_loc)
        self.ax.grid(True)

    def show(self):
        self.run()

        plt.show()
        plt.close(self.fig)

    def saveToFile(self, fileName: str):
        self.run()

        time = strftime("_%d-%m-%H:%M:%S", gmtime())
        plt.savefig(os.path.join('grafics', (fileName + time + '.png')), dpi=200)
        plt.close(self.fig)

    def add_text(self, text):
        plt.text(0.95, 0.1, text,
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform=self.ax.transAxes,
                 bbox={'facecolor': 'white', 'edgecolor': 'grey', 'alpha': 0.8})

    def add_line(self, xs, ys, color: Optional[str] = None):
        self.funcs.append(lambda _: self.ax.plot(xs, ys, color))
        # self.legends.append(legend)

    def add_max(self, min=None, max=None):
        self.funcs.append(lambda _: self.ax.set_ylim(bottom=min, top=max))

    def horizontal_line(self, value, legend: str, color: Optional[str] = None):
        self.funcs.append(lambda _: self.ax.hlines(value, self.lower, self.upper, color))
        self.legends.append(legend)

    def vertical_line(self, value, legend: str, color: Optional[str] = None):
        self.funcs.append(lambda _: self.ax.vlines(value))
        self.legends.append(legend)

    def x_label(self, label: str):
        self.funcs.append(lambda _: self.ax.set_xlabel(label, fontsize='xx-large'))

    def y_label(self, label: str):
        self.funcs.append(lambda _: self.ax.set_ylabel(label, fontsize='xx-large'))
