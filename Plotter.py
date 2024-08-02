from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoLocator, MaxNLocator, Locator, LogLocator
import re

class TrimOption(Enum):
    ADD = 1
    REMOVE_1 = 2
    REMOVE_2 = 3

def find_stabilization_point(discarded_size, window_size, series: pd.Series) -> Tuple[float, int]:
    series.replace([np.inf, -np.inf], np.nan, inplace=True)
    series = series.dropna(inplace=False)
    print(f'Finding stabilization point for series of length {len(series)}')
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
            print(f'Stabilized after window starting from {i - window_size + discarded}')
            break
    else:
        i = max(0, len(rolling_var) - window_size)
        print(f'No stabilization point found, using the last window at {i + discarded}')
        mean = series[i:].mean()
        return mean, i + discarded
    
    i = max(0, i - window_size)
    mean = series[i:].mean()
    print(f"Mean after stabilization: {mean}")
    return mean, i + discarded

class Plotter:
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
        
    def __plot_axis(self, ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, x: pd.Series, y: pd.Series, label: str, source_index: int, col_index: int, loc: Locator = AutoLocator()) -> None:
        stab_point, stab_phase_start = find_stabilization_point(self.discarded_size, self.window_size, y)
        # Plot smoothed line
        smoothed = savgol_filter(y, self.window_size // 2, 3)
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
        
        if (len(y_columns) > 4):
            print("Max. 4 columns are supported. Exiting...")
            exit(1)
        
        trim = False
        if len(y_columns) * self.combined_data.groupby(level='Source').ngroups > 6:
            print("Too many stats. Trimming...")
            trim = True
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        if secondary_y and (not trim or trim_option != TrimOption.REMOVE_2):
            assert(len(y_columns) == 2)
            ax1_twinx: matplotlib.axes.Axes = ax1.twinx()
            ax2_twinx: matplotlib.axes.Axes = ax2.twinx()
        
        source_index = -1
        for key, grp in self.combined_data.groupby(level='Source'):
            source_index += 1
            if trim and trim_option == TrimOption.ADD:
                y = grp[y_columns[0]]
                for i in range(1, len(y_columns)):
                    y += grp[y_columns[i]]
                self.__plot_axis(ax1, ax2, grp['t'], y, f'{key}', source_index, 0)
                continue
                
            for i, col in enumerate(y_columns):
                if trim and trim_option == TrimOption.REMOVE_1 and i == 0:
                    continue
                elif trim and trim_option == TrimOption.REMOVE_2 and i == 1:
                    continue
                
                if secondary_y and i == 1:
                    self.__plot_axis(ax1_twinx, ax2_twinx, grp['t'], grp[col], f'{key} - {col}', source_index, i)
                elif col == 'GHz':
                    print('Custom locator for GHz')
                    self.__plot_axis(ax1, ax2, grp['t'], grp[col], f'{key} - {col}', source_index, i, MaxNLocator(integer=True))
                else:
                    self.__plot_axis(ax1, ax2, grp['t'], grp[col], f'{key} - {col}', source_index, i)
        

        for i in range(2):
            ax = ax1 if i == 0 else ax2
            matches = re.match(r'Chart (\d+): (.+)', title)
            subtitle = matches.group(2) if i == 0 else f'After Stabilization'
            ax.set_xlabel('Time Elapsed (seconds)')
            ax.set_ylabel(y_label)
            ax.set_title(subtitle)
            if secondary_y and (not trim or trim_option != TrimOption.REMOVE_2):
                ax_twinx = ax1_twinx if i == 0 else ax2_twinx
                ax_twinx.set_ylabel(y_columns[1])
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twinx.get_legend_handles_labels()
                lines += lines2
                labels += labels2
                ax.legend(
                    # lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.
                    )
            else:
                ax.legend(
                    # loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.
                    )
                
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/{title}.png', dpi=300)
    
    def plot_all_charts(self) -> None:
        self.plot_chart(['OLTP TX'], 'Chart 1: Transaction Throughput', 'TXs/s')
        # self.plot_chart(['W MiB', 'R MiB'], 'Chart 2: W MiB, R MiB', 'MiB/s')
        self.plot_chart(['SSDReads/TX', 'SSDWrites/TX'], 'Chart 3: IO per TX', 'Operations/TX')
        self.plot_chart(['GHz', "Cycles/TX"], 'Chart 4: CPU Information', 'GHz', True, TrimOption.REMOVE_2)
    
    def plot_agg(self, title: str) -> None:
        # Normalize the bar heights for the same column
        paths = self.combined_data.index
        x_label_dir1 = {}
        x_label_dir2 = {}
        
        if title == 'selectivity':
            default_val = '-sel100'
        elif title == 'update-size':
            default_val = '-size5'
        elif title == 'included-columns':
            default_val = '-col1'
        else:
            default_val = ''
            
        for i, p in enumerate(paths):
            matches = re.match(r'(join|merged)-\d+-\d+-(read|scan|write|mixed-\d+-\d+-\d+)(-\w+)?$', p)
            if matches is None:
                print(f'Invalid path: {p}')
                exit(1)
            if matches.group(3) is None:
                key = f'{matches.group(1)}{default_val}'
            else:
                key = f'{matches.group(1)}{matches.group(3)}'
            if matches.group(1) == 'join':
                x_label_dir = x_label_dir1
            else:
                x_label_dir = x_label_dir2
                
            if key in x_label_dir.keys():
                x_label_dir[key].append((i, matches.group(2)))
            else:
                x_label_dir[key] = [(i, matches.group(2))]
        
        variance_size = 0
        for x_label_dir in [x_label_dir1, x_label_dir2]:
            for key, value in x_label_dir.items():
                x_label_dir[key] = sorted(value, key=lambda x: x[1])
                if variance_size == 0:
                    variance_size = len(x_label_dir[key])
                else:
                    assert(variance_size == len(x_label_dir[key]))
            
            
            sorted_items = sorted(
                x_label_dir.items(),
                key=lambda x: (
                    re.match(r'(join|merged)(-[A-Za-z]+([0-9]+))?$', x[0]).group(1),
                    '' if re.match(r'(join|merged)(-[A-Za-z]+([0-9]+))?$', x[0]).group(3) is None else int(re.match(r'(join|merged)(-[A-Za-z]+([0-9]+))?$', x[0]).group(3))
                )
            )
            x_label_dir.clear()
            x_label_dir.update(sorted_items)
        
        assert(len(x_label_dir1) == len(x_label_dir2))
        
        bar_width = 0.5
        label_width = bar_width * (len(self.combined_data.columns) + 1)
        label_starts = np.arange(len(x_label_dir1)) * label_width
        fig, (ax_up, ax_dn) = plt.subplots(2, 1, figsize=(len(x_label_dir) * label_width * 1.5, 6))
        
        ax_up.set_xticks(label_starts + (label_width - bar_width) / 2, labels=x_label_dir1.keys())
        ax_dn.set_xticks(label_starts + (label_width - bar_width) / 2, labels=x_label_dir2.keys())
        
        i = 0
        for col in self.combined_data.columns:
            col_data = self.combined_data[col]
            if col == 'TXs/s' or col == 'Reads/TX' or col == 'Writes/TX':
                norm_col = np.log10(col_data + 1)
                min_val = 0
                max_val = max(np.log10(col_data.max() + 1), 4)
                norm_col = (norm_col - min_val) / (max_val - min_val)
            elif col == 'GHz':
                min_val = 0
                max_val = 4
                norm_col = (col_data - min_val) / (max_val - min_val)
                
            if col == 'Reads/TX' or col == 'Writes/TX':
                norm_col = -norm_col
                va = 'top'
            else:
                va = 'bottom'
            
            for x_label_dir, ax in zip([x_label_dir1, x_label_dir2], [ax_up, ax_dn]):
                heights = [0 for _ in range(len(x_label_dir))]
                for j in range(variance_size):
                    j_norm_col = [norm_col.iloc[x[j][0]] for x in x_label_dir.values()]
                    bars = ax.bar(label_starts + i * bar_width * 0.5, j_norm_col, bar_width, bottom=heights, label=col, color=self.colors[j])
                    for bar, pair, k in zip(bars, x_label_dir.items(), range(len(x_label_dir))):
                        key = pair[0]
                        value = pair[1]
                        stat = col_data.iloc[value[j][0]]
                        if stat > 99999:
                            text = f'{stat:.2e}'
                        elif stat > 99:
                            text = f'{int(stat):d}'
                        elif stat > 0.1 or stat == 0:
                            text = f'{stat:.2f}'
                        else:
                            text = f'{stat:.2e}'
                        if len(value) > 1 and stat != 0:
                            text = f'{value[j][1]}:\n' + text
                        t = ax.text(bar.get_x() + bar.get_width() / 2.0, heights[k] + bar.get_height() * 0.5, text, ha='center', va='center', color = 'white', fontsize='x-small', fontfamily='monospace', fontweight='bold')
                        t.set_bbox(dict(facecolor=self.colors[j], alpha=0.5, edgecolor='none', pad=0.1))
                    heights = [x + bar.get_height() for x, bar in zip(heights, bars)]
                
                # Adding the text labels on each bar
                for bar, value, k in zip(bars, col_data, range(len(x_label_dir))):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, heights[k] * 1.1, f'{col}', ha='center', va=va)
                i += 1
        
        for ax in [ax_up, ax_dn]:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # ax.legend()
        fig.suptitle(f'Means for Experiment on {title}')
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/Aggregates.png', dpi=300)
        # Plot bars side by side for the same row
        # Add text on top of each column