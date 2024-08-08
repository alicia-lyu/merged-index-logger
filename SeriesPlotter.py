from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoLocator, MaxNLocator, Locator
import re
from args import args

class TrimOption(Enum):
    ADD = 1
    REMOVE_1 = 2
    REMOVE_2 = 3

def find_stabilization_point(discarded_size, window_size, series: pd.Series) -> Tuple[float, int]:
    series.replace([np.inf, -np.inf], np.nan, inplace=True)
    series = series.dropna(inplace=False)
    # print(f'Finding stabilization point for series of length {len(series)}')
    if len(series) > discarded_size:
        discarded = min(discarded_size, len(series) // 2)
        series = series[discarded:] # Remove the first half of the series
    else:
        discarded = 0
        
    norm_series = (series - series.min()) / (series.max() - series.min()) # Normalize the series
    rolling_var = norm_series.rolling(window=window_size).var()
    variance_threshold = 0.1

    for i in range(window_size, len(rolling_var)):
        last_N = len(rolling_var) - i
        if all(rolling_var.dropna().tail(last_N) < variance_threshold):
            # print(f'Stabilized after window starting from {i - window_size + discarded}')
            break
    else:
        i = max(0, len(rolling_var) - window_size)
        print(f'No stabilization point found, using the last window at {i + discarded}')
        mean = series[i:].mean()
        return mean, i + discarded
    
    i = max(0, i - window_size)
    mean = series[i:].mean()
    # print(f"Mean after stabilization: {mean}")
    return mean, i + discarded

class SeriesPlotter:
    def __init__(self, combined_data: pd.DataFrame, fig_dir: str) -> None:
        self.combined_data: pd.DataFrame = combined_data
        self.fig_dir: str = fig_dir
        self.linewidth = 3
        self.alpha = 0.7
        self.stab_linewidth = 1
        self.linestyles: List[str] = ['solid', 'dashdot', 'dashed', 'dotted']
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']
        self.discarded_size = 60
        self.window_size = self.discarded_size // 6
        
    def __plot_axis(self, ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, x: pd.Series, y: pd.Series, label: str, source_index: int, col_index: int, loc: Locator = AutoLocator(), logical_min = 0, logical_max = float('inf')) -> None:
        stab_point, stab_phase_start = find_stabilization_point(self.discarded_size, self.window_size, y)
        # Plot smoothed line
        smoothed = savgol_filter(y, self.window_size // 2, 3)
        smoothed = np.maximum(smoothed, np.full(smoothed.shape, logical_min))
        smoothed = np.minimum(smoothed, np.full(smoothed.shape, logical_max))
        x = x.values
        ax1.yaxis.set_major_locator(loc)
        ax1.plot(x, smoothed, label=label, 
                linewidth=self.linewidth, linestyle=self.linestyles[col_index],
                alpha=self.alpha, color=self.colors[source_index])
        # Plotted non-smoothed line after stabilization point
        ax2.plot(x[stab_phase_start:], y[stab_phase_start:], label=label, linewidth=2, linestyle=self.linestyles[col_index], color=self.colors[source_index])
        ax2.axhline(y = stab_point, linestyle=self.linestyles[col_index], color=self.colors[source_index], linewidth=self.stab_linewidth)
        ax2.text(1, stab_point, f'{stab_point:.2f}', color=self.colors[source_index], transform=ax2.get_yaxis_transform())

    def plot_chart(self, y_columns: List[str], title: str, y_label: str, secondary_y: bool = False, trim_option: TrimOption = TrimOption.ADD) -> None:
        print(f'Plotting {title}')
        
        # Check requirements for data
        if (len(y_columns) > 4):
            print("Max. 4 columns are supported. Exiting...")
            exit(1)
        trim = False
        if len(y_columns) * self.combined_data.groupby(level='Source').ngroups > 6:
            print("Too many stats. Trimming...")
            trim = True
        
        # Twinx for secondary y-axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1_twinx = None
        ax2_twinx = None
        if secondary_y and (not trim or trim_option != TrimOption.REMOVE_2):
            ax1_twinx: matplotlib.axes.Axes = ax1.twinx()
            ax2_twinx: matplotlib.axes.Axes = ax2.twinx()
        
        data_gen = self.combined_data.groupby(level='Source')
        for (key, grp), source_index in zip(data_gen, range(len(data_gen))):
            # Plot the sum of all columns
            if trim and trim_option == TrimOption.ADD:
                y = grp[y_columns[0]]
                for i in range(1, len(y_columns)):
                    y += grp[y_columns[i]]
                self.__plot_axis(ax1, ax2, grp['t'], y, f'{key}', source_index, 0)
                continue
            # Plot each column
            for i, col in enumerate(y_columns):
                # Skip trimmed columns
                if trim and trim_option == TrimOption.REMOVE_1 and i == 0:
                    continue
                elif trim and trim_option == TrimOption.REMOVE_2 and i == 1:
                    continue
                
                if secondary_y and i == 1:
                    self.__plot_axis(ax1_twinx, ax2_twinx, grp['t'], grp[col], f'{key} - {col}', source_index, i)
                elif col == 'GHz':
                    print('Custom locator for GHz')
                    self.__plot_axis(ax1, ax2, grp['t'], grp[col], f'{key} - {col}', source_index, i, MaxNLocator(integer=True), logical_max=4)
                else:
                    self.__plot_axis(ax1, ax2, grp['t'], grp[col], f'{key} - {col}', source_index, i)
        
        # Set legends
        for ax, ax_twinx in zip([ax1, ax2], [ax1_twinx, ax2_twinx]):
            matches = re.match(r'Chart (\d+): (.+)', title)
            subtitle = matches.group(2) if ax == ax1 else f'After Stabilization Point'
            ax.set_xlabel('Time Elapsed (seconds)')
            ax.set_ylabel(y_label)
            ax.set_title(subtitle)
            if secondary_y and (not trim or trim_option != TrimOption.REMOVE_2):
                ax_twinx.set_ylabel(y_columns[1])
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twinx.get_legend_handles_labels()
                lines += lines2
                labels += labels2
                ax.legend(lines, labels)
            else:
                ax.legend()
                
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/{title}.png', dpi=300)
    
    def plot_all_charts(self) -> None:
        self.plot_chart(['OLTP TX'], 'Chart 1: Transaction Throughput', 'TXs/s')
        if args.rocksdb is False:
            self.plot_chart(['SSDReads/TX', 'SSDWrites/TX'], 'Chart 2: IO per TX', 'Operations/TX')
        else:
            self.plot_chart(['SSTRead(ms)/TX', 'SSTWrite(ms)/TX'], 'Chart 2: IO per TX', 'ms/TX')
        self.plot_chart(['GHz', "Cycles/TX"], 'Chart 3: CPU Information', 'GHz', True, TrimOption.REMOVE_2)