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
        else:
            default_val = None
            suffix = ''
            
        for i, p in enumerate(x_labels):
            matches = re.match(
                r'(join|merged)-\d+-\d+-(read|scan|write|mixed-\d+-\d+-\d+)' + f'({suffix})?' + r'(\d+)?$', 
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
        
    def plot_agg(self) -> None:
        # Normalize the bar heights for the same column
        paths = self.agg_data.index
        join_scatter_dir, merged_scatter_dir = self.__get_scatter_dirs(paths)
        fig, axes = plt.subplots(1, len(self.agg_data.columns), figsize=(12, 5))
        
        for (col_join, scatter_df_join), (col_merged, scatter_df_merged), ax in zip(join_scatter_dir.items(), merged_scatter_dir.items(), axes):
            assert(col_join == col_merged)
            ax.scatter(scatter_df_join['x'], scatter_df_join['y'], marker='x', label='Join', color=self.colors[0], alpha=0.8, edgecolors='none', s=60)
            ax.scatter(scatter_df_merged['x'], scatter_df_merged['y'], marker='+', label='Merged', color=self.colors[1], edgecolors='none', s=60)
            
            for join_row, merged_row in zip(scatter_df_join.itertuples(), scatter_df_merged.itertuples()):
                if join_row.x != merged_row.x:
                    print(f'x values do not match: {join_row.x} != {merged_row.x}')
                    exit(1)
                
                ax.plot([join_row.x, merged_row.x], [join_row.y, merged_row.y], color='black', alpha=0.3, linewidth=3, linestyle='dotted')
            
            ax.xaxis.set_major_locator(plt.FixedLocator(scatter_df_join['x'].unique()))
            ax.set_xlabel(f'{self.title.capitalize()} (%)')
            ax.set_ylabel(f'{col_join} (Mean across all TX types)')
            
            ax.legend()
        
        fig.suptitle(f'{self.title.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/Aggregates.png', dpi=300)