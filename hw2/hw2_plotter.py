import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from matplotlib.pyplot import FigureCanvasBase
from matplotlib.lines import Line2D

import seaborn as sns
sns.set_style("darkgrid")


''' plots to make:
      o Maybe x1 vs x2
      o Convergence - log(f(x)) over time with
        convergence goal line, multiple x0's

'''

class Plotter:
    def __init__(self):
        self.figs = []
        self.axes = []

        # plt.ion()

    def make_new_plot(self, nrows, ncols, title, xlabs, ylabs, sharex=True):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, squeeze=False)
        axes = axes[0].tolist()
        self.figs.append(fig)
        self.axes.append(axes)
        ind = len(self.figs)-1

        fig.suptitle(title)
        for i in range(len(axes)):
            axes[i].set_xlabel(xlabs[i])
            axes[i].set_ylabel(ylabs[i])

        return fig, axes

    def init_xy_data(self, fig_ind:int, ax_ind:int, xdata:ndarray, ydata:ndarray, lab:str):
        ax:Axes = self.axes[fig_ind][ax_ind]
        line, = ax.plot(xdata, ydata, label=lab)

        fig:Figure = self.figs[fig_ind]
        fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.0001)

    def update_plot(self, fig_ind:int, ax_ind:int, line_ind:int, xdata:ndarray, ydata:ndarray):
        ax: Axes = self.axes[fig_ind][ax_ind]
        line: Line2D = ax.lines[line_ind]

        line.set_data(xdata, ydata)
        ax.relim()
        ax.autoscale_view(True,True,True)
        ax.redraw_in_frame()
        plt.pause(0.00001)

    def hold_plot(self):
        plt.show(block=True)

    def close_plots(self):
        plt.close('all')