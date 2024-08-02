import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import re
import matplotlib.pyplot as plt

class AggPlotter:
    def __init__(self, agg_data: pd.DataFrame, fig_dir: str, title: str) -> None:
        self.agg_data: pd.DataFrame = agg_data
        self.fig_dir: str = fig_dir
        self.title: str = title
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']
        
    def __reorganize_x_labels(self, x_labels: str) -> Tuple[dict, dict]:
        
        def __get_sort_key(x: str) -> Tuple[str, int]:
            matches = re.match(r'(join|merged)(-[A-Za-z]+([0-9]+))?$', x)
            if matches is None:
                print(f'Invalid key: {x}')
                exit(1)
            return (f'{matches.group(1)}', 0 if matches.group(3) is None else int(matches.group(3)))
            
        x_label_dir1 = {}
        x_label_dir2 = {}
        
        if self.title == 'selectivity':
            default_val = '-sel100'
        elif self.title == 'update-size':
            default_val = '-size5'
        elif self.title == 'included-columns':
            default_val = '-col1'
        else:
            default_val = ''
            
        for i, p in enumerate(x_labels):
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
            sorted_items = sorted(x_label_dir.items(), key=lambda x: __get_sort_key(x[0]))
            x_label_dir.clear()
            x_label_dir.update(sorted_items)
        
        assert(len(x_label_dir1) == len(x_label_dir2))
        
        return x_label_dir1, x_label_dir2
        
    def __get_and_normalize_column(self, col: str) -> Tuple[pd.Series, pd.Series]:
        col_data = self.agg_data[col]
        if col == 'TXs/s' or col == 'Reads/TX' or col == 'Writes/TX':
            norm_col = np.log10(col_data + 1)
            min_val = 0
            max_val = max(np.log10(col_data.max() + 1), 4)
            norm_col = (norm_col - min_val) / (max_val - min_val)
        elif col == 'GHz':
            min_val = 0
            max_val = 4
            norm_col = (col_data - min_val) / (max_val - min_val)
        return [col_data, norm_col]
    
    def __get_stats_representation(self, col_data: pd.Series) -> Callable[..., str]:
        max_value = col_data.max()
        if max_value > 99999:
            return lambda x: f'{x:.2e}'
        elif max_value > 99:
            return lambda x: f'{int(x):d}'
        elif max_value > 0.1 or max_value == 0:
            return lambda x: f'{x:.2f}'
        else:
            return lambda x: f'{x:.2e}'
        
    def plot_agg(self) -> None:
        # Normalize the bar heights for the same column
        paths = self.agg_data.index
        x_label_dir1, x_label_dir2 = self.__reorganize_x_labels(paths)
        M = len(x_label_dir1)
        N = len(list(x_label_dir1.values())[0])
        n = len(self.agg_data.columns)
        
        bar_width = 0.5
        label_width = bar_width * (n + 1)
        print(f'M: {M}, N: {N}, bar_width: {bar_width}, label_width: {label_width}')
        
        label_starts = np.arange(M) * label_width
        
        fig, (ax_up, ax_dn) = plt.subplots(2, 1, figsize=(M * label_width * 1.5, 6))
        
        ax_up.set_xticks(label_starts + (label_width - bar_width) / 2, labels=x_label_dir1.keys())
        ax_dn.set_xticks(label_starts + (label_width - bar_width) / 2, labels=x_label_dir2.keys())
        
        i = 0
        for col in self.agg_data.columns:
            col_data, norm_col = self.__get_and_normalize_column(col)
            stats_representation = self.__get_stats_representation(col_data)
            
            if col == 'Reads/TX' or col == 'Writes/TX':
                norm_col = -norm_col
                va = 'top'
            else:
                va = 'bottom'
            
            for x_label_dir, ax in zip([x_label_dir1, x_label_dir2], [ax_up, ax_dn]):
                heights = [0 for _ in range(M)]
                for j in range(N):
                    j_norm_col = [norm_col.iloc[x[j][0]] for x in x_label_dir.values()]
                    bars = ax.bar(label_starts + i * bar_width * 0.5, j_norm_col, bar_width, bottom=heights, label=col, color=self.colors[j])
                    for bar, value, k in zip(bars, x_label_dir.values(), range(M)):
                        stat = col_data.iloc[value[j][0]]
                        text = stats_representation(stat)
                        if N > 1 and stat != 0:
                            text = f'{value[j][1]}:\n' + text
                        t = ax.text(
                            bar.get_x() + bar.get_width() / 2.0, 
                            heights[k] + bar.get_height() * 0.5, 
                            text, 
                            ha='center', va='center', color = 'white', 
                            fontsize='x-small', fontfamily='monospace', fontweight='bold')
                        t.set_bbox(dict(facecolor=self.colors[j], alpha=0.5, edgecolor='none', pad=0.1))
                    
                    heights = [x + bar.get_height() for x, bar in zip(heights, bars)]
                    
                # Adding the text labels on each bar
                for bar, value, k in zip(bars, col_data, range(M)):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, heights[k] * 1.2, f'{col}', ha='center', va=va)
                i += 1
        
        for ax in [ax_up, ax_dn]:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            
        fig.suptitle(f'{self.title}')
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/Aggregates.png', dpi=300)