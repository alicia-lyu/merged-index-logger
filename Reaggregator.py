import pandas as pd
from args import args
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os, re
import numpy as np

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
        self.size_dir = os.path.join(os.path.dirname(p), 'size_rocksdb' if args.rocksdb else 'size')
        self.__parse_size()
        
    def __parse_size(self):
        for method in ['join', 'merged', 'base']:
            size_core, size_rest, total_time = self.__agg_size_by_config(f'{method}_rest.csv')
            size_core.to_csv(os.path.join(self.size_dir, f'{method}_core.csv'), index=False)
            size_rest.to_csv(os.path.join(self.size_dir, f'{method}_rest.csv'), index=False)
            total_time.to_csv(os.path.join(self.size_dir, f'{method}_time.csv'), index=False)
    
    def __agg_size_by_config(self, size_filename: str):
        def parse_config(config):
            patten = r'([\.\d]+)\|(\d+)\|(\d+)\|(\d+)'
            matches = re.match(patten, config)
            assert(matches is not None)
            dram = float(matches.group(1))
            target = int(matches.group(2))
            selectivity = int(matches.group(3))
            included_columns = int(matches.group(4))
            return dram, target, selectivity, included_columns
        
        def process_config_dict(config_dict: dict, func: Callable, column_name: str) -> pd.DataFrame:
            config_df = []
            for config, value in config_dict.items():
                dram, target, selectivity, included_columns = parse_config(config)
                config_df.append((dram, target, selectivity, included_columns, func(value)))
            return pd.DataFrame(config_df, columns=['dram', 'target', 'selectivity', 'included_columns', column_name])
        
        size_df = pd.read_csv(os.path.join(self.size_dir, size_filename))
        config_to_size_core = defaultdict(list) # config -> [size], list to be averaged
        config_to_size_rest = defaultdict(list) # config -> [size], list to be averaged
        config_to_total_time = defaultdict(defaultdict(list)) # config -> table -> [time], list to be averaged, tables will be summed
        
        for index, row in size_df.iterrows():
            if row['time(ms)'] > 0:
                config_to_total_time[row['config']][row['table(s)']].append(row['time(ms)'])
            
            if row['table(s)'] == 'core': # ATTN: name mismatch
                size_dir = config_to_size_rest
            elif row['table(s)'] in ['join_results', 'merged_index', 'stock+orderline_secondary']:
                size_dir = config_to_size_core
            elif row['table(s)'] == 'orderline_secondary':
                config_to_size_core[row['config']][-1] += row['size']
                continue
            else:
                raise ValueError(f'Invalid table name: {row["table(s)"]}')
                
            size_dir[row['config']].append(row['size'])
            
        size_core = process_config_dict(config_to_size_core, np.mean, 'size')
        size_rest = process_config_dict(config_to_size_rest, np.mean, 'size')
        total_time = process_config_dict(config_to_total_time, lambda table_dict: sum([np.mean(time_list) for time_list in table_dict.values()]), 'time(ms)')
        
        return size_core, size_rest, total_time

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
            elif method == 'merged':
                method_rows = merged_rows
            else:
                method_rows = base_rows
                
            core_size_filename = f"{method}_core.csv"
            rest_size_filename = f"{method}_rest.csv"
            
            core_size_df = pd.read_csv(os.path.join(self.size_dir, core_size_filename))
            rest_size_df = pd.read_csv(os.path.join(self.size_dir, rest_size_filename))
            
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
        
        for df, x, color in zip([base_rows, join_rows, merged_rows], [base_x, join_x, merged_x], self.colors[:3]):
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