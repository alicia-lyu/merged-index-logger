from matplotlib import transforms
import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from args import args
from Reaggregator import Reaggregator

class AggPlotter:
    def __init__(self, agg_data: pd.DataFrame, fig_dir: str) -> None:
        self.agg_data: pd.DataFrame = agg_data
        self.fig_dir: str = fig_dir
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']
        self.markers: List[str] = ['x', '+', 'o', 's', 'D', 'v']
        self.labels: List[str] = ['Base', 'Join', 'Merged']
    
    def plot(self) -> Tuple[dict, dict]:
        if args.type == 'all-tx':
            self.__plot_type()
        else:
            self.__plot_x()
    
    # Plot data with no extra variable. Type becomes the x-axis.
    def __plot_type(self) -> Tuple[dict, dict]:
        reaggregator = Reaggregator(self.agg_data)
        base_rows, join_rows, merged_rows = reaggregator() # Sorted dfs
        for col in self.agg_data.columns:
            scatter_points_list, scatter_points_size_list = self.__get_scatter_points(base_rows, join_rows, merged_rows, col)
            self.__plot(col, None, *scatter_points_list)
    
    def __get_scatter_points(self, base_rows: pd.DataFrame, join_rows: pd.DataFrame, merged_rows: pd.DataFrame, col: str, size_df = None) -> Tuple[List, List]:
        scatter_points_list = []
        if "core_size" in base_rows.columns:
            scatter_points_size_list = []
        for rows in [base_rows, join_rows, merged_rows]: # Sorted on x
            scatter_points = []
            if "core_size" in base_rows.columns:
                scatter_points_size = []
            for row in rows.itertuples():
                x = row.Index
                data_row = self.agg_data.iloc[row.i_col]
                y = data_row[col]
                scatter_points.append((x, y))
                if "core_size" in base_rows.columns:
                    scatter_points_size.append((row.core_size, y, x))
            if "core_size" in base_rows.columns:
                scatter_points_size.sort()
            scatter_points_list.append(scatter_points)
            if "core_size" in base_rows.columns:
                scatter_points_size_list.append(scatter_points_size)
        if "core_size" in base_rows.columns:
            return scatter_points_list, scatter_points_size_list
        else:
            return scatter_points_list, None
        
    # Plot data with an extra variable (e.g., selectivity, update size, included columns)
    def __plot_x(self) -> Tuple[dict, dict]:
        reaggregator = Reaggregator(self.agg_data)
        type_to_rows = reaggregator()
        for tp, (base_rows, join_rows, merged_rows) in type_to_rows.items(): # Sorted
            for col in self.agg_data.columns:
                scatter_points_list, scatter_points_size_list = self.__get_scatter_points(base_rows, join_rows, merged_rows, col)
                self.__plot(col, tp, *scatter_points_list)
                self.__plot_by_size(col, tp, *scatter_points_size_list)
    
    def __set_locator_by_col(self, ax: plt.Axes, col: str) -> None:
        if col == 'Utilized CPU (%)':
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        else:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
        
    def __plot_by_size(self, col: str, tp: str, base_scatter_points_size: List[Tuple], join_scatter_points_size: List[Tuple], merged_scatter_points_size: List[Tuple]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
        
        for i, scatter_points in enumerate([base_scatter_points_size, join_scatter_points_size, merged_scatter_points_size]):
            df = self.__create_df_size(scatter_points)
            ax.plot(df['size'], df['y'], marker=self.markers[i], label=self.labels[i], color=self.colors[i], linewidth=2)
            for row in df.itertuples():
                self.__add_text(ax, row.size, row.y, i, f'{row.x}%')
            
        ax.set_xlabel('Space Consumption (GB)')
        ax.set_ylabel(f'{col}')
        self.__set_locator_by_col(ax, col)
        ax.legend()
        
        fig.tight_layout()
        col_name = col.replace('/', '-')
        file_name = f'{col_name}-{tp}-size.png' if tp is not None else f'{col_name}-size.png'
        
        fig.savefig(
            f'{self.fig_dir}/{file_name}',
            dpi=300)
    
    def __create_df_size(self, size_list: List[Tuple]) -> pd.DataFrame:
        df = pd.DataFrame(size_list, columns=['size', 'y', 'x'])
        return df
        
    def __create_df_xy(self, xy_list: List[Tuple]) -> pd.DataFrame:
        df = pd.DataFrame(xy_list, columns=['x', 'y'])
        return df
        
    def __find_breakpoints_in_array(self, values: np.array):
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

    
    def __find_breakpoints(self, base_scatter_points: pd.DataFrame, join_scatter_points: pd.DataFrame, merged_scatter_points: pd.DataFrame) -> Tuple[float, float]:
        all_values = np.concatenate([df['y'].values for df in [base_scatter_points, join_scatter_points, merged_scatter_points]])
        return self.__find_breakpoints_in_array(all_values)
    
    def __plot(self, col: str, tp: str, base_scatter_points: List[Tuple], join_scatter_points: List[Tuple], merged_scatter_points: List[Tuple]) -> None:
        
        print(f'**************************************** Plotting {col} for {tp} ****************************************')
        print(f'Join: {join_scatter_points}')
        print(f'Merged: {merged_scatter_points}')
        
        base_scatter_points = self.__create_df_xy(base_scatter_points)
        join_scatter_points = self.__create_df_xy(join_scatter_points)
        merged_scatter_points = self.__create_df_xy(merged_scatter_points)
        
        assert(list(base_scatter_points.index) == list(join_scatter_points.index) and list(join_scatter_points.index) == list(merged_scatter_points.index))
        
        breakpoint_ret = self.__find_breakpoints(base_scatter_points, join_scatter_points, merged_scatter_points)
        
        # All TX types except Point Query with an extra column
        all_tx_mask = join_scatter_points.index != 'Point Query\n(with an extra column)'
        
        if breakpoint_ret is False:
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
            self.__plot_axis(ax, col, tp, 
                             base_scatter_points[all_tx_mask],
                             join_scatter_points[all_tx_mask], 
                             merged_scatter_points[all_tx_mask])
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4))
            fig.subplots_adjust(hspace=0.05)
            lower, upper = breakpoint_ret
            print(f'Lower: {lower}, Upper: {upper}')
            self.__plot_broken_axis(ax1, ax2, col, tp, 
                                    base_scatter_points[all_tx_mask],
                                    join_scatter_points[all_tx_mask], 
                                    merged_scatter_points[all_tx_mask],
                                    lower, upper)
            
        fig.tight_layout()
        col_name = col.replace('/', '-')
        file_name = f'{col_name}-{tp}.png' if tp is not None else f'{col_name}.png'
        fig.savefig(
            f'{self.fig_dir}/{file_name}',
            dpi=300)
        
        # Compare two point queries: with or without an extra column
        if tp is None: # TX Type is x-axis
            fig_read, ax_read = plt.subplots(1, 1, figsize=(4, 4))
            pq_mask = join_scatter_points.index.str.contains('Point Query')
            self.__plot_axis(ax_read, col, tp,
                                base_scatter_points[pq_mask],
                                join_scatter_points[pq_mask],
                                merged_scatter_points[pq_mask])
            fig_read.tight_layout()
            fig_read.savefig(
                f'{self.fig_dir}/{file_name.replace(".png", "-point-query.png")}',
                dpi=300)
            
    def __plot_axis(self, ax: plt.Axes, col: str, tp: str, base_scatter_points: pd.DataFrame, join_scatter_points: pd.DataFrame, merged_scatter_points: pd.DataFrame, broken: bool = False) -> None:
        for i, scatter_points in enumerate([base_scatter_points, join_scatter_points, merged_scatter_points]):
            ax.scatter(scatter_points['x'], scatter_points['y'], marker=self.markers[i], label=self.labels[i], color=self.colors[i], linewidth=2)
            if not broken:
                for row in scatter_points.itertuples():
                    self.__add_text(ax, row.x, row.y, i)
            # else defer to __plot_broken_axis
        
        ax.set_xlabel(args.get_xlabel())
        
        ax.set_ylabel(f'{col}')
        ax.set_xticks(join_scatter_points['x'].unique())
        ax.set_xticklabels([str(x) for x in join_scatter_points['x'].unique()], fontsize=9, fontfamily='monospace')
        tick_positions = ax.get_xticks()
        tick_width = tick_positions[1] - tick_positions[0]
        ax.set_xlim(tick_positions[0] - tick_width * 0.5, tick_positions[-1] + tick_width * 0.5)
        ax.legend()
        
    def __get_text(self, y: float) -> str:
        if y > 1e6 or y < 0.05:
            return f'{y:.2e}'
        elif y > 1e3:
            return f'{int(y):d}'
        elif y == 0:
            return '0'
        else:
            return f'{y:.2f}'
    
    def __add_text(self, ax: plt.Axes, x: float, y, method: int, t: str = None) -> None:
        h_offset = (method - 1) * 0.1
        v_offset = (0.5 - method % 2) * 0.2
        # base(0)      merged(2)
        #        join(1)
        offset_transform = transforms.ScaledTranslation(h_offset, v_offset, ax.figure.dpi_scale_trans)
        ax.text(
            x, y, t if t is not None else self.__get_text(y), color=self.colors[method], 
            fontsize=8, ha='right' if method == 0 else 'left', va='center', 
            bbox=dict(facecolor=self.colors[method], alpha=0.3, edgecolor='none', boxstyle='square'),
            transform=ax.transData + offset_transform)
        
    def __plot_broken_axis(self, ax1: plt.Axes, ax2: plt.Axes, col: str, tp: str, base_scatter_points: pd.DataFrame, join_scatter_points: pd.DataFrame, merged_scatter_points: pd.DataFrame, lower: float, upper: float) -> None:
        self.__plot_axis(ax1, col, tp, base_scatter_points, join_scatter_points, merged_scatter_points, True)
        self.__plot_axis(ax2, col, tp, base_scatter_points, join_scatter_points, merged_scatter_points, True)

        ax2.set_ylim(-lower * 0.05, lower)
        vmax = max(join_scatter_points['y'].values.max(), merged_scatter_points['y'].values.max()) * 1.1
        ax1.set_ylim(upper, vmax)
        
        ax1_yticks = ax1.get_yticks()
        print(f'ax1_yticks: {ax1_yticks}')
        if ax1_yticks[0] == 0: # Upper broken axis' minimum is 0. This happens when the largest value dominate the scale of upper axis.
            ax1.set_yscale('log')
            ax1.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
            print("ax1_yticks: ", ax1.get_yticks())
        
        # Add text to the right ax
        for base_row, join_row, merged_row in zip(base_scatter_points.itertuples(), join_scatter_points.itertuples(), merged_scatter_points.itertuples()):
            for i, row in enumerate([base_row, join_row, merged_row]):
                if row.y < lower:
                    ax = ax2
                elif row.y > upper:
                    ax = ax1
                else:
                    print(f'Row {row} is in the middle.')
                    exit(1)
                self.__add_text(ax, row.x, row.y, i)
        
        # Tick ax1's x axis at the top and ax2's x axis at the bottom
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        # Remove labels, ticks, and legends set by __plot_axis
        ax1.set_xlabel('')
        ax1.set_xticks([], [])
        ax2.legend().remove()
        ax2.set_ylabel('')

        # Add slanted line to suggest brokenness
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)