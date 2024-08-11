import pandas as pd
from typing import Tuple, List
from enum import Enum
import re, os
import matplotlib.pyplot as plt
import matplotlib

class TableOption(Enum):
    CORE = "TPC-C Tables"
    JM = "Materialized Join or Merged Index"

colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']

def process_data(size_df: pd.DataFrame, table_option: TableOption) -> pd.DataFrame:
    config_to_size = {}
    for index, row in size_df.iterrows():
        if table_option == TableOption.CORE:
            if row['table(s)'] != 'core':
                continue
        elif table_option == TableOption.JM:
            if row['table(s)'] != 'join_results' and row['table(s)'] != 'merged_index':
                continue
        config_to_size[row['config']] = row['size'], row['time']
    scatter_stats = []
    # Pick out stats and reorder
    for config, (size, duration) in config_to_size.items():
        patten = r'(\d+)\|(\d+)\|(\d+)'
        matches = re.match(patten, config)
        assert(matches is not None)
        scatter_stats.append((int(matches.group(2)), int(matches.group(3)), float(size), float(duration)))
    scatter_stats.sort()
    scatter_stats = pd.DataFrame(scatter_stats, columns=['selectivity', 'included_columns', 'size', 'time'])
    return scatter_stats

def plot_ax(ax: matplotlib.axes.Axes, scatter_stats: pd.DataFrame, label: str, col: str, marker: str, offset: int = 0) -> None:
    print("Plotting", label)
    print(scatter_stats)
    y_offset = -((offset - 0.5) * 0.3)
    size_scale = 500 / scatter_stats[col].median()
    ax.scatter(scatter_stats['selectivity'], scatter_stats['included_columns'] + y_offset, scatter_stats[col] * size_scale, marker=marker, label=label, color=colors[offset], alpha=0.5, edgecolors='none', clip_on=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(scatter_stats['included_columns'].unique()))
    ax.set_yticklabels(["None", "All"])
    xticks = list(scatter_stats['selectivity'].unique())
    xticks.append(0)
    xticks.sort()
    print(xticks)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))
    for index, row in scatter_stats.iterrows():
        t = f'{row[col]:.2f}\nGiB' if col == 'size' else f'{int(row[col]):d} ms'
        ax.text(row['selectivity'], row['included_columns'] + y_offset, t, color=colors[offset], ha='center', va='center', fontsize=8)

def plot_both(join_size: pd.DataFrame, merged_size: pd.DataFrame, table_option: TableOption) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    assert(list(join_size.columns) == list(merged_size.columns))
    join_scatters = process_data(join_size, table_option)
    merged_scatters = process_data(merged_size, table_option)
    
    plot_ax(ax, join_scatters, 'Materialized Join', 'size', 'o', 0)
    plot_ax(ax, merged_scatters, 'Merged Index', 'size', 'o', 1)
    
    ax.set_xlabel('Selectivity (%)')
    ax.set_ylabel('Included Columns')
    ax.set_title(f"Size of {table_option.value}")
    ax.legend()
    fig.tight_layout()
    filename = table_option.value.replace(' ', '_').lower()
    fig.savefig(f'size/{filename}.png', dpi=300)
    
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    plot_ax(ax2, join_scatters, 'Materialized Join', 'time', '$⧗$', 0)
    plot_ax(ax2, merged_scatters, 'Merged Index', 'time', '$⧗$', 1)
    
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Included Columns')
    ax2.set_title(f"Time to Generate {table_option.value}")
    ax2.legend()
    fig2.tight_layout()
    filename2 = table_option.value.replace(' ', '_').lower() + '_time'
    fig2.savefig(f'size/{filename2}.png', dpi=300)
    
if __name__ == '__main__':
    join_size = pd.read_csv('./size/join_size.csv')
    merged_size = pd.read_csv('./size/merged_size.csv')
    os.makedirs('./size', exist_ok=True)
    plot_both(join_size, merged_size, TableOption.JM)
    plot_both(join_size, merged_size, TableOption.CORE)
    
    