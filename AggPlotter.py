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
        
    def __get_and_normalize_column(self, col_data: pd.Series, col: str) -> Tuple[pd.Series, float, float]:
        if col == 'TXs/s' or col == 'Reads/TX' or col == 'Writes/TX' or col == "IO/TX":
            norm_col = np.log10(col_data + 1)
            min_val = 0
            max_val = max(norm_col.max(), 4)
            norm_col = (norm_col - min_val) / (max_val - min_val)
        elif col == 'GHz':
            min_val = 0
            max_val = 4
            norm_col = (col_data - min_val) / (max_val - min_val)
        else:
            raise ValueError("Unsupported column name for normalization")
        return norm_col, min_val, max_val
    
    def __unnorm(self, norm_val: float, min_val: float, max_val: float, col: str) -> float:
        val = norm_val * (max_val - min_val) + min_val
        if col == 'TXs/s' or col == 'Reads/TX' or col == 'Writes/TX' or col == "IO/TX":
            return 10 ** val - 1
        else:
            return val
        
    def __stats_representation(self, major_rep: Callable[..., str]) -> Callable[..., str]:
        return major_rep
    
    def __get_stats_representation(self, col_data: pd.Series) -> Callable[..., str]:
        max_value = col_data.max()
        if max_value > 99999:
            major_rep = lambda x: f'{x:.2e}'
        elif max_value > 99:
            major_rep = lambda x: f'{int(x):d}'
        elif max_value > 0.1 or max_value == 0:
            major_rep = lambda x: f'{x:.2f}'
        else:
            major_rep = lambda x: f'{x:.2e}'
        return self.__stats_representation(major_rep)
        
    def plot_agg(self) -> None:
        # Normalize the bar heights for the same column
        paths = self.agg_data.index
        x_label_dir1, x_label_dir2 = self.__reorganize_x_labels(paths)
        keys = list(x_label_dir1.keys()) + list(x_label_dir2.keys())
        sublabel_lists = list(x_label_dir1.values()) + list(x_label_dir2.values())
        M = len(x_label_dir1)
        N = len(list(x_label_dir1.values())[0])
        n = len(self.agg_data.columns)
        
        bar_width = 0.5
        # bar_locs = [i * 0.5 for i in range(M * 2)]
        
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(2 * n * 1.5, (n + 1) // 2 * (N + 1)))
        
        axes = axes.flatten()

        texts = []
        for i_col, col in enumerate(self.agg_data.columns):
            ax = axes[i_col]
            ax.set_title(col)
            # ax.set_xticks(bar_locs, keys)
            
            col_data = self.agg_data[col]
            norm_col, min_val, max_val = self.__get_and_normalize_column(col_data, col)
            # print(col_data, norm_col, min_val, max_val)
            stats_representation = self.__get_stats_representation(col_data)
            
            heights = [0 for _ in range(M * 2)]
            
            for i_sublabel in range(N):
                col_locs = [sublabel_list[i_sublabel][0] for sublabel_list in sublabel_lists]
                sublabels = [sublabel_list[i_sublabel][1] for sublabel_list in sublabel_lists]
                
                sublabel = sublabels[0]
                for i in range(1, M):
                    assert(sublabel == sublabels[i])

                bars = ax.bar(keys, norm_col.iloc[col_locs], bar_width, bottom=heights, color=self.colors[i_sublabel], label = sublabel)
                
                for bar, col_loc, sublabel, k in zip(bars, col_locs, sublabels, range(M * 2)):
                    stat = col_data.iloc[col_loc]
                    text = stats_representation(stat)
                    if N > 1 and norm_col.iloc[col_loc] < 0.01:
                        va = 'top'
                        height = heights[k] + bar.get_height()
                    else:
                        va='center'
                        height = heights[k] + bar.get_height() * 0.5
                    if stat > 0.001 or N == 1:
                        t = ax.text(
                            bar.get_x() + bar.get_width() * 0.5, 
                            height, 
                            text, 
                            ha='center', va=va, color = 'white', 
                            fontsize='small', fontfamily='monospace', fontweight='bold')
                        t.set_bbox(dict(facecolor=self.colors[i_sublabel], alpha=0.5, edgecolor='none', pad=0.2))
                        texts.append(t)
                
                heights = [x + bar.get_height() for x, bar in zip(heights, bars)]
            
            ax.set_ylim(0)
            y_ticks = ax.get_yticks()
            y_labels = []
            for y_tick in y_ticks:
                y_label = self.__unnorm(y_tick, min_val, max_val, col)
                y_labels.append(stats_representation(y_label))
            ax.set_yticks(y_ticks, y_labels)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])
        
        # adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle='-', color='black'), force_points=0.1, lim=100)
        fig.suptitle(f'{self.title}')
        fig.tight_layout()
        fig.savefig(f'{self.fig_dir}/Aggregates.png', dpi=300)