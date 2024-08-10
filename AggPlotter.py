from matplotlib import transforms
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import re
import matplotlib.pyplot as plt
from args import args

class AggPlotter:
    def __init__(self, agg_data: pd.DataFrame, fig_dir: str) -> None:
        self.agg_data: pd.DataFrame = agg_data
        self.fig_dir: str = fig_dir
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']
    
    def plot(self) -> Tuple[dict, dict]:
        if args.type == 'all-tx':
            self.__plot_type()
        else:
            self.__plot_x()
    
    # Plot data with no extra variable. Type becomes the x-axis.
    def __plot_type(self) -> Tuple[dict, dict]:
        for col in self.agg_data.columns:
            join_scatter_points = []
            merged_scatter_points = []
            for i, p in enumerate(self.agg_data.index):
                pattern = r'(join|merged)-[\d\.]+-\d+-(read|scan|write|mixed-\d+-\d+-\d+)'
                matches = re.match(pattern, p)
                
                if matches is None:
                    print(f'Invalid path {p} does not match {pattern}.')
                    exit(1)
                
                if matches.group(1) == 'join':
                    method_points = join_scatter_points
                else:
                    method_points = merged_scatter_points
                
                if matches.group(2) == 'read':
                    tx_type = 'Point Query\n(Read Only)'
                elif matches.group(2) == 'write':
                    tx_type = 'Read-Write'
                elif matches.group(2) == 'scan':
                    tx_type = 'Analytical Query\n(Scan)'
                else:
                    tx_type = matches.group(2).capitalize()
                method_points.append((tx_type, self.agg_data[col].iloc[i]))
            join_scatter_points.sort()
            merged_scatter_points.sort()
            self.__plot(col, None, join_scatter_points, merged_scatter_points)
        
    # Plot data with an extra variable (e.g., selectivity, update size, included columns)
    def __plot_x(self) -> Tuple[dict, dict]:
        
        default_val, suffix = args.get_default()
        
        rows_per_type: dict[str, Tuple[List, List]] = {}
            
        for i, p in enumerate(self.agg_data.index):
            pattern = r'(join|merged)-[\d\.]+-\d+-(read|scan|write|mixed-\d+-\d+-\d+)' + f'({suffix})?' + r'(\d+)?'
            matches = re.match(pattern, p)
            if matches is None:
                print(f'Invalid path {p} does not match {pattern}.')
                exit(1)
            
            if matches.group(4) is None:
                x = default_val
            else:
                x = int(matches.group(4))
                
            tp = matches.group(2)
            
            if tp in rows_per_type.keys():
                join_rows, merged_rows = rows_per_type[tp]
            else:
                join_rows = []
                merged_rows = []
                rows_per_type[tp] = (join_rows, merged_rows)
                
            if matches.group(1) == 'join':
                method_rows = join_rows
            else:
                method_rows = merged_rows
                
            method_rows.append((x, i))
        
        for tp, (join_rows, merged_rows) in rows_per_type.items():
            assert(len(join_rows) == len(merged_rows))
            
        self.__plot_all(rows_per_type)
    
    def __plot_all(self, rows_per_type: dict[str, Tuple[List, List]]):
        for col in self.agg_data.columns:
            for tp, (join_rows, merged_rows) in rows_per_type.items():
                join_scatter_points = []
                merged_scatter_points = []
                for x, i in join_rows:
                    join_scatter_points.append((x, self.agg_data[col].iloc[i]))
                for x, i in merged_rows:
                    merged_scatter_points.append((x, self.agg_data[col].iloc[i]))
                join_scatter_points.sort()
                merged_scatter_points.sort()
                self.__plot(col, tp, join_scatter_points, merged_scatter_points)
        
    def __find_breakpoints(self, values: np.array):
        sorted_values = np.sort(values)
        sqrt_values = np.sqrt(np.sqrt(sorted_values))
        diffs = np.diff(sqrt_values)
        breakpoint_idx = np.argmax(diffs)
        lower = sorted_values[breakpoint_idx]
        upper = sorted_values[breakpoint_idx + 1]
        lower = max(lower * 1.2, 1)
        upper = max((upper + lower) / 2, 
                    upper - max(sorted_values.max() - upper, upper) * 0.2
                    )
        if lower >= upper:
            return False
        return lower, upper

    
    def __plot(self, col: str, tp: str, join_scatter_points: List[Tuple], merged_scatter_points: List[Tuple]) -> None:
        
        print(f'**************************************** Plotting {col} for {tp} ****************************************')
        print(f'Join: {join_scatter_points}')
        print(f'Merged: {merged_scatter_points}')
        
        join_scatter_points = pd.DataFrame(join_scatter_points, columns=['x', 'y'])
        merged_scatter_points = pd.DataFrame(merged_scatter_points, columns=['x', 'y'])
        
        join_values = join_scatter_points['y'].values
        merged_values = merged_scatter_points['y'].values
        all_values = np.concatenate((join_values, merged_values))
        ret = self.__find_breakpoints(all_values)
        
        if ret is False:
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
            if col == 'GHz' and all_values.min() > 3.55:
                ax.set_ylim(3.5, 4.05)
            self.__plot_axis(ax, col, tp, join_scatter_points, merged_scatter_points)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4.5))
            fig.subplots_adjust(hspace=0.05)
            lower, upper = ret
            print(f'Lower: {lower}, Upper: {upper}')
            self.__plot_broken_axis(ax1, ax2, col, tp, join_scatter_points, merged_scatter_points, lower, upper)
            
        fig.tight_layout()
        col_name = col.replace('/', '-')
        file_name = f'{col_name}-{tp}.png' if tp is not None else f'{col_name}.png'
        fig.savefig(
            f'{self.fig_dir}/{file_name}',
            dpi=300)
            
    def __plot_axis(self, ax: plt.Axes, col: str, tp: str, join_scatter_points: pd.DataFrame, merged_scatter_points: pd.DataFrame, broken: bool = False) -> None:
        ax.scatter(join_scatter_points['x'], join_scatter_points['y'], marker='x', label='Join', color=self.colors[0], s=60, clip_on=True)
        ax.scatter(merged_scatter_points['x'], merged_scatter_points['y'], marker='+', label='Merged', color=self.colors[1], s=60, clip_on=True)
        
        # Connect join v.s. merged
        for join_row, merged_row in zip(join_scatter_points.itertuples(), merged_scatter_points.itertuples()):
            if join_row.x != merged_row.x:
                print(f'x values do not match: {join_row.x} != {merged_row.x}')
                exit(1)
                
            ax.plot([join_row.x, merged_row.x], [join_row.y, merged_row.y], color='black', alpha=0.3, linewidth=3, linestyle='dotted')
            
            if not broken:
                self.__add_text(ax, join_row.x, join_row.y, 0)
                self.__add_text(ax, merged_row.x, merged_row.y, 1)
        
        ax.set_xlabel(args.get_xlabel())
        
        ax.set_ylabel(f'{col}')
        ax.set_xticks(join_scatter_points['x'].unique())
        ax.set_xticklabels([str(x) for x in join_scatter_points['x'].unique()])
        ax.legend()
        
    def __get_text(self, y: float) -> str:
        if y > 1e6 or y < 0.1:
            return f'{y:.2e}'
        elif y > 1e3:
            return f'{int(y):d}'
        else:
            return f'{y:.2f}'
    
    def __add_text(self, ax: plt.Axes, x: float, y: float, method: int):
        offset_transform = transforms.ScaledTranslation((method - 0.5) * 0.4, 0, ax.figure.dpi_scale_trans)
        ax.text(
            x, y, self.__get_text(y), color=self.colors[0], 
            fontsize=8, ha='right' if method == 0 else 'left', va='center', 
            bbox=dict(facecolor=self.colors[method], alpha=0.3, edgecolor='none', boxstyle='square'),
            transform=ax.transData + offset_transform)
        
    def __plot_broken_axis(self, ax1: plt.Axes, ax2: plt.Axes, col: str, tp: str, join_scatter_points: pd.DataFrame, merged_scatter_points: pd.DataFrame, lower: float, upper: float) -> None:
        self.__plot_axis(ax1, col, tp, join_scatter_points, merged_scatter_points, True)
        self.__plot_axis(ax2, col, tp, join_scatter_points, merged_scatter_points, True)

        ax2.set_ylim(-lower * 0.05, lower)
        
        vmax = max(join_scatter_points['y'].values.max(), merged_scatter_points['y'].values.max()) * 1.1 if col != 'GHz' else 4.05
        ax1.set_ylim(upper, vmax)
        
        ax1_yticks = ax1.get_yticks()
        print(f'ax1_yticks: {ax1_yticks}')
        if ax1_yticks[0] == 0:
            ax1.set_yscale('log')
            ax1.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
            print("ax1_yticks: ", ax1.get_yticks())
        
        for join_row, merged_row in zip(join_scatter_points.itertuples(), merged_scatter_points.itertuples()):
            
            if join_row.y < lower:
                join_ax = ax2
            elif join_row.y > upper:
                join_ax = ax1
            else:
                print(f'Join row {join_row} is in the middle.')
                exit(1)
                
            if merged_row.y < lower:
                merged_ax = ax2
            elif merged_row.y > upper:
                merged_ax = ax1
            else:
                print(f'Merged row {merged_row} is in the middle.')
                exit(1)
            
            self.__add_text(join_ax, join_row.x, join_row.y, 0)
            self.__add_text(merged_ax, merged_row.x, merged_row.y, 1)
        
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        
        ax1.set_xlabel('')
        ax1.set_xticks([], [])
        ax2.legend().remove()
        ax2.set_ylabel('')