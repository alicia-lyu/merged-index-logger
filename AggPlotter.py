import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text

class AggPlotter:
    def __init__(self, agg_data: pd.DataFrame, fig_dir: str, title: str) -> None:
        self.agg_data: pd.DataFrame = agg_data
        self.fig_dir: str = fig_dir
        self.title: str = title
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']
    
    def __calc_mean(self, method_dir: dict):
        scatter_dir = {}
        for col in self.agg_data.columns:
            col_data = self.agg_data[col]
            scatter_points = []
            for x, value in method_dir.items():
                s = 0
                for i, _ in value:
                    s += col_data.iloc[i]
                scatter_points.append((x, s / len(value)))
            scatter_points.sort()
            scatter_dir[col] = pd.DataFrame(scatter_points, columns=['x', 'y'])
        return scatter_dir
    
    def __get_scatter_dirs_all_tx(self, x_labels: List[str]) -> Tuple[dict, dict]:
        join_scatter_dir = {}
        merged_scatter_dir = {}
        
        for col in self.agg_data.columns:
            join_scatter_points = []
            merged_scatter_points = []
            for i, p in enumerate(x_labels):
                matches = re.match(r'(join|merged)-[\d\.]+-\d+-(read|scan|write|mixed-\d+-\d+-\d+)', p)
                
                if matches is None:
                    print(f'Invalid path: {p}')
                    exit(1)
                
                if matches.group(1) == 'join':
                    method_dir = join_scatter_points
                else:
                    method_dir = merged_scatter_points
                
                method_dir.append((matches.group(2), self.agg_data[col].iloc[i]))
            join_scatter_points.sort()
            merged_scatter_points.sort()
            join_scatter_dir[col] = pd.DataFrame(join_scatter_points, columns=['x', 'y'])
            merged_scatter_dir[col] = pd.DataFrame(merged_scatter_points, columns=['x', 'y'])
            
        return join_scatter_dir, merged_scatter_dir
        
    def __get_scatter_dirs(self, x_labels: List[str]) -> Tuple[dict, dict]:
            
        join_dir = {}
        merged_dir = {}
        
        if self.title == 'selectivity':
            default_val = 100
            suffix = '-sel'
        elif self.title == 'update-size':
            default_val = 5
            suffix = '-size'
        elif self.title == 'included-columns':
            default_val = 1
            suffix = '-col'
        elif self.title == 'all-tx':
            return self.__get_scatter_dirs_all_tx(x_labels)
        else:
            print(f'Invalid title: {self.title}')
            exit(1)
            
        for i, p in enumerate(x_labels):
            matches = re.match(
                r'(join|merged)-[\d\.]+-\d+-(read|scan|write|mixed-\d+-\d+-\d+)' + f'({suffix})?' + r'(\d+)?$', 
                p)
            if matches is None:
                print(f'Invalid path: {p}')
                exit(1)
            
            if matches.group(4) is None:
                x = default_val
            else:
                x = int(matches.group(4))
                
            if matches.group(1) == 'join':
                method_dir = join_dir
            else:
                method_dir = merged_dir
                
            if x not in method_dir:
                method_dir[x] = [(i, matches.group(2))]
            else:
                method_dir[x].append((i, matches.group(2)))
        
        intersection = set(join_dir.keys()) & set(merged_dir.keys())
        
        join_dir = {k: v for k, v in join_dir.items() if k in intersection}
        merged_dir = {k: v for k, v in merged_dir.items() if k in intersection}
                
        join_scatter_dir = self.__calc_mean(join_dir)
        merged_scatter_dir = self.__calc_mean(merged_dir)
        
        return join_scatter_dir, merged_scatter_dir
        
    def __find_breakpoints(self, values: np.array):
        sorted_values = np.sort(values)
        sqrt_values = np.sqrt(np.sqrt(sorted_values))
        diffs = np.diff(sqrt_values)
        breakpoint_idx = np.argmax(diffs)
        lower = sorted_values[breakpoint_idx]
        upper = sorted_values[breakpoint_idx + 1]
        lower *= 1.2
        upper = max((upper + lower) / 2, upper - max(sorted_values.max() - upper, upper) * 0.2)
        if lower >= upper:
            return False
        return lower, upper

    def plot_agg(self) -> None:
        # Normalize the bar heights for the same column
        paths = self.agg_data.index
        join_scatter_dir, merged_scatter_dir = self.__get_scatter_dirs(paths)
        fig, axes = plt.subplots(2, len(self.agg_data.columns), figsize=(12, 6))
        fig.subplots_adjust(hspace=0.05)

        for (col_join, scatter_df_join), (col_merged, scatter_df_merged), ax1, ax2 in zip(join_scatter_dir.items(), merged_scatter_dir.items(), axes[0], axes[1]):
            assert(col_join == col_merged)
            
            join_values = scatter_df_join['y'].values
            merged_values = scatter_df_merged['y'].values
            all_values = np.concatenate((join_values, merged_values))
            
            ret = self.__find_breakpoints(all_values)
            
            if ret is False:
                ax2.remove()
                ax2 = None
            else:
                lower, upper = ret
                print(f'Lower: {lower}, Upper: {upper}')
            
            
            ax1.scatter(scatter_df_join['x'], scatter_df_join['y'], marker='x', label='Join', color=self.colors[0], alpha=0.8, s=60, clip_on=True)
            ax1.scatter(scatter_df_merged['x'], scatter_df_merged['y'], marker='+', label='Merged', color=self.colors[1], s=60, clip_on=True)
            
            if ax2 is not None:
                ax2.scatter(scatter_df_join['x'], scatter_df_join['y'], marker='x', label='Join', color=self.colors[0], alpha=0.8, s=60, clip_on=True)
                ax2.scatter(scatter_df_merged['x'], scatter_df_merged['y'], marker='+', label='Merged', color=self.colors[1], s=60, clip_on=True)

            for join_row, merged_row in zip(scatter_df_join.itertuples(), scatter_df_merged.itertuples()):
                if join_row.x != merged_row.x:
                    print(f'x values do not match: {join_row.x} != {merged_row.x}')
                    exit(1)
                ax1.plot([join_row.x, merged_row.x], [join_row.y, merged_row.y], color='black', alpha=0.3, linewidth=3, linestyle='dotted')
                if ax2 is not None:
                    ax2.plot([join_row.x, merged_row.x], [join_row.y, merged_row.y], color='black', alpha=0.3, linewidth=3, linestyle='dotted')

            if ax2 is not None:
                ax2.set_ylim(0, lower if lower > 0 else 1)
                vmax = all_values.max() * 1.1 if col_join != 'GHz' else 4
                ax1.set_ylim(upper, vmax)
                
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

            
            if self.title != 'all-tx':
                if ax2 is not None:
                    ax2.set_xlabel(f'{self.title.capitalize()} (%)')
                else:
                    ax1.set_xlabel(f'{self.title.capitalize()} (%)')
                ax1.set_ylabel(f'{col_join} (Mean across all TX types)')
            else:
                if ax2 is not None:
                    ax2.set_xlabel('Transaction Type')
                else:
                    ax1.set_xlabel('Transaction Type')
                ax1.set_ylabel(f'{col_join}')

            ax1.set_xticks(scatter_df_join['x'].unique())
            ax1.set_xticklabels([str(x) for x in scatter_df_join['x'].unique()])
            ax1.legend()

        fig.suptitle(f'{self.title.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/Aggregates.png', dpi=300)