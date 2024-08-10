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

def plot_scatters(ax: matplotlib.axes.Axes, size_df: pd.DataFrame, label: str, table_option: TableOption, offset: int = 0) -> None:
    print("Plotting", label)
    # Eliminate duplicates
    config_to_size = {}
    for index, row in size_df.iterrows():
        if table_option == TableOption.CORE:
            if row['table(s)'] != 'core':
                continue
        elif table_option == TableOption.JM:
            if row['table(s)'] != 'join_results' and row['table(s)'] != 'merged_index':
                continue
        config_to_size[row['config']] = row['size']
    scatter_stats = []
    # Pick out stats and reorder
    for config, size in config_to_size.items():
        patten = r'(\d+)\|(\d+)\|(\d+)'
        matches = re.match(patten, config)
        assert(matches is not None)
        scatter_stats.append((int(matches.group(2)), int(matches.group(3)), float(size)))
    scatter_stats.sort()
    # Plot
    scatter_stats = pd.DataFrame(scatter_stats, columns=['selectivity', 'included_columns', 'size'])
    print(scatter_stats)
    y_offset = -((offset - 0.5) * 0.3)
    ax.scatter(scatter_stats['selectivity'], scatter_stats['included_columns'] + y_offset, scatter_stats['size'] * 1000, marker='o', label=label, color=colors[offset], alpha=0.5, edgecolors='none')
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(scatter_stats['included_columns'].unique()))
    ax.set_yticklabels(["None", "All"])
    xticks = list(scatter_stats['selectivity'].unique())
    xticks.append(0)
    xticks.sort()
    print(xticks)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))
    for index, row in scatter_stats.iterrows():
        ax.text(row['selectivity'], row['included_columns'] + y_offset, f'{row["size"]:.2f}\nGiB', color=colors[offset], ha='left', va='center')


def plot_both(join_size: pd.DataFrame, merged_size: pd.DataFrame, table_option: TableOption) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    assert(list(join_size.columns) == list(merged_size.columns))
    
    plot_scatters(ax, join_size, 'Materialized Join', table_option)
    plot_scatters(ax, merged_size, 'Merged Index', table_option, 1)
    
    ax.set_xlabel('Selectivity (%)')
    ax.set_ylabel('Included Columns')
    ax.set_title(f"Size of {table_option.value}")
    ax.legend()
    fig.tight_layout()
    filename = table_option.value.replace(' ', '_').lower()
    fig.savefig(f'size/{filename}.png', dpi=300)
    
if __name__ == '__main__':
    join_size = pd.read_csv('join_size.csv')
    merged_size = pd.read_csv('merged_size.csv')
    os.makedirs('./size', exist_ok=True)
    plot_both(join_size, merged_size, TableOption.JM)
    plot_both(join_size, merged_size, TableOption.CORE)
    
    