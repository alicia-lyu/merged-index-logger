import pandas as pd
from args import args
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os

'''
Reaggregate the Agg DataFrame (2D, index=source, columns=metrics) into
1. all-tx: 3 DataFrames (2D, index=tx_type, columns=[i_col]), one for join, one for merged, one for base.
2. An extra-x: 3 DataFrames (2D, index=extra-x, columns=[i_col, core_size, rest_size]), one for join, one for merged, one for base.
Each DataFrame can be used for plotting given a metric type: x = index, y = agg_data.loc[metric, i_col].
'''

class Reaggregator:
    def __init__(self, agg_data: pd.DataFrame) -> None:
        self.agg_data: pd.DataFrame = agg_data
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']

    def __call__(self) -> pd.DataFrame:
        if args.type == 'all-tx':
            return self.__reagg_all_tx()
        elif args.type in ['selectivity', 'included-columns']:
            return self.__reagg_extra()
        else:
            print(f'Invalid type {args.type} for reaggregation')
            exit(1)
            
    def __reagg_all_tx(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        join_rows = []
        merged_rows = []
        base_rows = []
        for i, p in enumerate(self.agg_data.index):
            method, tx_type = self.parse_path(p)
            
            if method == 'join':
                method_points = join_rows
            elif method == 'merged':
                method_points = merged_rows
            else:
                method_points = base_rows
            
            if tx_type == 'read-locality':
                tx_type = 'Point Lookup'
            elif tx_type == 'read':
                tx_type = 'Point Lookup\n(with an extra lookup key)'
            elif tx_type == 'write':
                tx_type = 'Read-Write TX'
            elif tx_type == 'scan':
                tx_type = 'Range Scan'
            else:
                tx_type = tx_type.capitalize()
            
            method_points.append((tx_type, i))
        
        dfs = []
        for points in [base_rows, join_rows, merged_rows]:
            points.sort()
            points_df = pd.DataFrame(points, columns=['tx_type', 'i_col'])
            points_df.set_index('tx_type', inplace=True)
            dfs.append(points_df)
        return tuple(dfs)
    
    def __reagg_extra(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def get_size(size_df: pd.DataFrame, x: int) -> float:
            row = size_df[size_df[args.type] == x]
            if row.empty:
                print(f'No data for {args.type} = {x}')
                exit(1)
            return row['size'].iloc[-1]

        type_to_rows = defaultdict(lambda: ([], [], []))
        
        for i, p in enumerate(self.agg_data.index):
            
            method, tx_type, extr_val = self.parse_path(p)
            
            base_rows, join_rows, merged_rows = type_to_rows[tx_type]
            
            if method == 'join':
                method_rows = join_rows
                core_size_filename = "join_materialized_join_or_merged_index.csv"
                rest_size_filename = "join_tpc-c_tables.csv"
            elif method == 'merged':
                method_rows = merged_rows
                core_size_filename = "merged_materialized_join_or_merged_index.csv"
                rest_size_filename = "merged_tpc-c_tables.csv"
            else:
                method_rows = base_rows
                core_size_filename = "base_materialized_join_or_merged_index.csv"
                rest_size_filename = "base_tpc-c_tables.csv"
                
            size_dir = os.path.join(os.path.dirname(p), 'size_rocksdb' if args.rocksdb else 'size')
                
            core_size_df = pd.read_csv(os.path.join(size_dir, core_size_filename))
            rest_size_df = pd.read_csv(os.path.join(size_dir, rest_size_filename))
            
            core_size = get_size(core_size_df, extr_val)
            rest_size = get_size(rest_size_df, extr_val)
                
            method_rows.append((extr_val, i, core_size, rest_size))
        
        type_to_dfs = {}
        
        for type, (base_rows, join_rows, merged_rows) in type_to_rows.items():
            join_rows.sort()
            join_df = pd.DataFrame(join_rows, columns=['x', 'i_col', 'core_size', 'rest_size'])
            join_df.set_index('x', inplace=True)
            merged_rows.sort()
            merged_df = pd.DataFrame(merged_rows, columns=['x', 'i_col', 'core_size', 'rest_size'])
            merged_df.set_index('x', inplace=True)
            base_rows.sort()
            base_df = pd.DataFrame(base_rows, columns=['x', 'i_col', 'core_size', 'rest_size'])
            base_df.set_index('x', inplace=True)
            type_to_dfs[type] = (join_df, merged_df, base_df)
        
        # Use the last set of dfs to plot size (tx_type independent)  
        self.__plot_size(join_df, merged_df, base_df)
        
        return join_df, merged_df, base_df
        
    def __plot_size(self, join_rows: pd.DataFrame, merged_rows: pd.DataFrame, base_rows: pd.DataFrame) -> None: # TODO: Test this
        if join_rows.index != merged_rows.index or merged_rows.index != base_rows.index:
            print('Extra-x values do not match')
            print(f'Join:\n{join_rows.index}')
            print(f'Merged:\n{merged_rows.index}')
            print(f'Base:\n{base_rows.index}')
            exit(1)
            
        bar_width = 0.3
        base_x = range(len(base_rows.index)) - bar_width
        join_x = range(len(join_rows.index))
        merged_x = range(len(merged_rows.index)) + bar_width
        
        fig, ax = plt.subplots(figsize=(4.5, 4))
        
        for df, x, color, label in zip([base_rows, join_rows, merged_rows], [base_x, join_x, merged_x], self.colors, ['Base', 'Join', 'Merged']):
            ax.bar(x, df['core_size'], width=bar_width, color=color)
            ax.bar(x, df['rest_size'], width=bar_width, color=color, hatch='//', bottom=df['core_size'])
            
        ax.set_xticks(range(len(base_rows.index)), base_rows.index)
        ax.set_xlabel(args.get_xlabel())
        ax.set_ylabel('Size (GB)')
        
        color_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(self.colors[:3], ['Base', 'Join', 'Merged'])]
        hatch_patches = [mpatches.Patch(facecolor='white', hatch=hatch, label=label, edgecolor='black') for hatch, label in zip(['', '//'], ['Core', 'Rest'])]
        
        combined_handles = color_patches + hatch_patches
        ax.legend(handles=combined_handles)
        fig.tight_layout()
        fig.savefig(os.path.join(args.get_dir(), f'{args.type}_size.png'))