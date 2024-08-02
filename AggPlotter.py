import pandas as pd
import numpy as np
from typing import List, Tuple
import re
import matplotlib.pyplot as plt

class AggPlotter:
    def __init__(self, combined_data: pd.DataFrame, fig_dir: str) -> None:
        self.combined_data: pd.DataFrame = combined_data
        self.fig_dir: str = fig_dir
        
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