import matplotlib.pyplot as plt
import matplotlib
from typing import List
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoLocator, MaxNLocator, Locator, LogLocator

class Plotter:
    def __init__(self, combined_data: pd.DataFrame, fig_dir: str) -> None:
        self.combined_data: pd.DataFrame = combined_data
        self.fig_dir: str = fig_dir
        self.linewidth = 4
        self.alpha = 0.5
        self.stab_linewidth = 1
        self.linestyles: List[str] = ['solid', 'dashdot', 'dashed', 'dotted']
        self.colors: List[str] = ['#023047', '#ffb703', '#fb8500']
        self.discarded_size = 60
        self.window_size = self.discarded_size // 6
        
    def __plot_axis(self, ax: matplotlib.axes.Axes, x: pd.Series, y: pd.Series, label: str, source_index: int, col_index: int, loc: Locator = AutoLocator()) -> None:
        if 'scan' not in self.fig_dir:
            # Plot stabilization point
            stab_point, stab_index, stab_phase_start = self.__find_stabilization_point(y)
            ax.axhline(y = stab_point, 
                    #    xmin = (stab_index - self.window_size - self.discarded_size) / len(smoothed), xmax = (stab_index - self.discarded_size) / len(smoothed),
                    linestyle=self.linestyles[col_index], color=self.colors[source_index], linewidth=self.stab_linewidth)
            ax.text(1, stab_point, f'{stab_point:.2f}', color=self.colors[source_index], transform=ax.get_yaxis_transform())
            
            # Plot smoothed line
            smoothed = savgol_filter(y, self.window_size, 3)
            smoothed = smoothed[self.discarded_size:]
            x = x.values[self.discarded_size:]
            ax.yaxis.set_major_locator(loc)
            ax.plot(x, smoothed, label=label, 
                    linewidth=self.linewidth, linestyle=self.linestyles[col_index],
                    alpha=self.alpha, color=self.colors[source_index])
            # Plot scattered points
            y = y.values[self.discarded_size:].astype(float)
            # y_min, y_max = ax.get_ylim()
            # print(f'Y min: {y_min}, Y max: {y_max}')
            # for i in range(0, len(smoothed)):
            #     # only plot point if it deviates from the smoothed line
            #     if np.abs(y[i] - smoothed[i]) < 0.1 * (np.max(smoothed) - np.min(smoothed)):
            #         y[i] = np.nan
            ax.scatter(x, y, color=self.colors[source_index], s=1)
        else:
            ax.scatter(x[self.discarded_size:], y[self.discarded_size:], label=label, color=self.colors[source_index], s=2)

    def plot_chart(self, y_columns: List[str], title: str, y_label: str, secondary_y: bool = False) -> None:
        print(f'Plotting {title}')
        fig, ax = plt.subplots()
        if secondary_y:
            assert(len(y_columns) == 2)
            ax2: matplotlib.axes.Axes = ax.twinx()
        
        source_index = 0
        for key, grp in self.combined_data.groupby(level='Source'):
            for i, col in enumerate(y_columns):
                if secondary_y and i == 1:
                    self.__plot_axis(ax2, grp['t'], grp[col], f'{key} - {col}', source_index, i)
                elif col == 'GHz':
                    print('Custom locator for GHz')
                    self.__plot_axis(ax, grp['t'], grp[col], f'{key} - {col}', source_index, i, MaxNLocator(integer=True))
                else:
                    self.__plot_axis(ax, grp['t'], grp[col], f'{key} - {col}', source_index, i)
            source_index += 1
        
        ax.set_xlabel('Time Elapsed (seconds)')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if secondary_y:
            ax2.set_ylabel(y_columns[1])
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
            ax.legend(
                # lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.
                )
        else:
            ax.legend(
                # loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.
                )
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/{title}.png')
    
    def plot_all_charts(self) -> None:
        self.plot_chart(['OLTP TX'], 'Chart 1: Transaction Throughput', 'TXs/s')
        # self.plot_chart(['W MiB', 'R MiB'], 'Chart 2: W MiB, R MiB', 'MiB/s')
        self.plot_chart(['SSDReads/TX', 'SSDWrites/TX'], 'Chart 3: IO per TX', 'Operations/TX')
        self.plot_chart(['GHz', "Cycles/TX"], 'Chart 4: GHz and Cycles per TX', 'GHz', secondary_y=True)

    def __find_stabilization_point(self, series: pd.Series) -> (float, int, int):
        threshold1 = 0.01 # For finding stabilization point
        threshold2 = 0.1 # For finding stabliization phase, assuming the last 2 values are in the stabilization phase
        weights = np.linspace(0, 2, len(series))
        weighted_series = series * weights
        weighted_rolling_mean = weighted_series.rolling(window=self.window_size).mean()
        # print(weighted_rolling_mean)
        rolling_mean = series.rolling(window=self.window_size).mean()
        stab_phase_start = 0
        stab_point = np.nan
        stab_index = 0
        for i in range(len(weighted_rolling_mean) - 2, 31, -1): # start from the third last value because transactions may be stopping
            if np.isnan(stab_point) and np.abs(weighted_rolling_mean.values[i] - weighted_rolling_mean.values[i - 1]) <= threshold1 * weighted_rolling_mean.values[i - 1]:
                # stabilized from window i - 1
                print(f'Stabilization point found at {i - 1}: {rolling_mean.values[i - 1]}')
                stab_point = rolling_mean.values[i - 1]
                stab_index = i - 1
            if stab_phase_start == 0 and np.abs(weighted_rolling_mean.values[i] - weighted_rolling_mean.values[i - 1]) > threshold2 * weighted_rolling_mean.values[i - 1]:
                print(f'Stabilization phase start at {i}')
                stab_phase_start = i
            if np.isnan(stab_point) == False and stab_phase_start != 0:
                break
        
        if np.isnan(stab_point):
            print(f'Could not find stabilization point, use the second last value: {rolling_mean.values[-2]}')
            stab_point = rolling_mean.values[-2]
            stab_index = len(rolling_mean) - 2
        
        return stab_point, stab_index, stab_phase_start