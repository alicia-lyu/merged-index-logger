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
Each DataFrame can be used for plotting given a metric type: x = index, y = agg_data.iloc[i_col][metric]
'''

class Reaggregator:
    def __init__(self, agg_data: pd.DataFrame) -> None:
        self.agg_data: pd.DataFrame = agg_data
        self.colors: List[str] = ['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00', '#70e000']
        
        self.size_file_base = 'size_outer' if args.outer_join else 'size'
        self.method_names = ['Base Tables', 'Materialized Join', 'Merged Index']
        self.method_names_short = ['base', 'join', 'merged']
        self.__parse_size()
    
    def __get_size_paths(self, method: str) -> Tuple[str, str, str]:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        size_path = os.path.join(current_dir, f'{method}', self.size_file_base + '.csv')
        core_path = os.path.join(current_dir, f'{method}', f'{self.size_file_base}_core.csv')
        rest_path = os.path.join(current_dir, f'{method}', f'{self.size_file_base}_rest.csv')
        time_path = os.path.join(current_dir, f'{method}', f'{self.size_file_base}_time.csv')
        return size_path, core_path, rest_path, time_path
        
    def __parse_size(self):
        for method in self.method_names_short:
            method = 'rocksdb_' + method if args.rocksdb else method
            size_path, core_path, rest_path, time_path = self.__get_size_paths(method)
            
            for derivative_path in [core_path, rest_path, time_path]:
                if not os.path.exists(derivative_path) or os.path.getmtime(derivative_path) < os.path.getmtime(size_path): # derivative is older
                    break
            else: # no break
                code_path = os.path.realpath(__file__)
                if os.path.getmtime(code_path) < os.path.getmtime(size_path): # code is older
                    print(f'{method} size is up-to-date')
                    continue
            
            print(f'Parsing Size for {method}...')
            size_core, size_rest, total_time = self.__agg_size_by_config(size_path)
            size_core.to_csv(core_path, index=False)
            size_rest.to_csv(rest_path, index=False)
            total_time.to_csv(time_path, index=False)
    
    def __agg_size_by_config(self, size_path: str):
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
        
        size_df = pd.read_csv(size_path)
        config_to_size_core = defaultdict(list) # config -> [size], list to be averaged
        config_to_size_rest = defaultdict(list) # config -> [size], list to be averaged
        config_to_total_time = defaultdict(lambda: defaultdict(list)) # config -> table -> [time], list to be averaged, tables will be summed
        
        for index, row in size_df.iterrows():
            if row['time(ms)'] > 0:
                config_to_total_time[row['config']][row['table(s)']].append(row['time(ms)'])
            
            if row['table(s)'] == 'core': # ATTN: name mismatch
                size_dir = config_to_size_rest
            elif row['table(s)'] in ['join_results', 'merged_index', 'stock+orderline_secondary']:
                size_dir = config_to_size_core
            elif row['table(s)'] == 'orderline_secondary':
                config_to_size_rest[row['config']][-1] += row['size']
                continue
            elif row['table(s)'] == 'total':
                continue
            else:
                raise ValueError(f'Invalid table name: {row["table(s)"]}')
                
            size_dir[row['config']].append(row['size'])
            
        size_core = process_config_dict(config_to_size_core, np.mean, 'size')
        size_rest = process_config_dict(config_to_size_rest, np.mean, 'size')
        total_time = process_config_dict(config_to_total_time, lambda table_dict: sum([np.mean(time_list) for time_list in table_dict.values()]), 'time(ms)')
        
        return size_core, size_rest, total_time

    def __call__(self) -> pd.DataFrame:
        print(f'Reaggregating for {args.type}...')
        if args.type == 'all-tx':
            return self.__reagg_all_tx()
        elif args.type in ['selectivity', 'included-columns']:
            return self.__reagg_extra()
        else:
            print(f'Invalid type {args.type} for reaggregation')
            exit(1)
            
    def __reagg_all_tx(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.__plot_size_single()
        join_rows = []
        merged_rows = []
        base_rows = []
        for i, p in enumerate(self.agg_data.index):
            method, tx_type = args.parse_path(p)
            
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
    
    def __get_size(self, size_df: pd.DataFrame, x: int) -> float:
            suffix_label, suffix_val = args.get_filter_for_size_df()
            size_df_key = args.type.replace('-', '_')
            row = size_df[(size_df[size_df_key] == x) & (size_df[suffix_label] == suffix_val)]
            if row.empty:
                print(f'No data for {args.type} = {x}')
                exit(1)
            return row['size'].iloc[-1]
        
    def __convert_to_df(self, rows: List[Tuple[int, int, float, float]]) -> pd.DataFrame:
        rows.sort()
        df = pd.DataFrame(rows, columns=["x", 'i_col', 'core_size', 'rest_size'])
        df.set_index("x", inplace=True)
        return df
    
    def __reagg_extra(self):

        type_to_rows = defaultdict(lambda: ([], [], []))
        
        for i, p in enumerate(self.agg_data.index):
            
            method, tx_type, extr_val = args.parse_path(p)
            
            base_rows, join_rows, merged_rows = type_to_rows[tx_type]
            
            if method == 'join':
                method_rows = join_rows
            elif method == 'merged':
                method_rows = merged_rows
            else:
                method_rows = base_rows
            
            _, core_path, rest_path, _ = self.__get_size_paths(method)
            
            core_size_df = pd.read_csv(core_path)
            rest_size_df = pd.read_csv(rest_path)
            
            core_size = self.__get_size(core_size_df, extr_val)
            rest_size = self.__get_size(rest_size_df, extr_val)
                
            method_rows.append((extr_val, i, core_size, rest_size))
        
        type_to_dfs = {}
        
        for type, (base_rows, join_rows, merged_rows) in type_to_rows.items():
            type_to_dfs[type] = (self.__convert_to_df(base_rows), self.__convert_to_df(join_rows), self.__convert_to_df(merged_rows))
        
        # Use the last set of dfs to plot size (tx_type independent)  
        self.__plot_size(*(type_to_dfs[type]))
        
        return type_to_dfs
        
    def __plot_size(self, base_rows: pd.DataFrame, join_rows: pd.DataFrame, merged_rows: pd.DataFrame) -> None:
        if list(join_rows.index) != list(merged_rows.index) or list(merged_rows.index) != list(base_rows.index):
            print('Extra-x values do not match')
            print(f'Join:\n{join_rows.index}')
            print(f'Merged:\n{merged_rows.index}')
            print(f'Base:\n{base_rows.index}')
            exit(1)
        
        print(f'**************************************** Plotting Size for extra {args.type} ****************************************')
        print('Base:', base_rows)
        print('Join:', join_rows)
        print('Merged:', merged_rows)
            
        bar_width = 0.2
        base_x = [x - bar_width for x in range(len(base_rows.index))]
        join_x = range(len(join_rows.index))
        merged_x = [x + bar_width for x in range(len(merged_rows.index))]
        
        fig, ax = plt.subplots(figsize=(len(base_rows.index) * bar_width * 6, 3))
        
        for df, x, color in zip([base_rows, join_rows, merged_rows], [base_x, join_x, merged_x], self.colors[:3]):
            ax.bar(x, df['core_size'], width=bar_width, color=color, edgecolor='black')
            # ax.bar(x, df['rest_size'], width=bar_width, color=color, hatch="xx", bottom=df['core_size'], edgecolor='black')
            
        ax.set_xticks(range(len(base_rows.index)), base_rows.index)
        ax.set_xlabel(args.get_xlabel())
        ax.set_ylabel('Size (GB)')
        
        color_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(self.colors[:3], self.method_names)]
        # hatch_patches = [mpatches.Patch(facecolor='white', hatch=hatch, label=label, edgecolor='black') for hatch, label in zip(['', "xx"], ['Core', 'Supporting'])]
        
        combined_handles = color_patches
            # + hatch_patches
        ax.legend(handles=combined_handles)
        fig.tight_layout()
        fig.savefig(os.path.join(args.get_dir(), f'{args.type}_size.png'), dpi=300)
        
    def plot_size(self):
        # Plot 4 selectivities: 5, 19, 50, 100
        args.type = 'selectivity'
        methods_df = []
        for method in self.method_names_short:
            method_rows = []
            for selectivity in [5, 19, 50, 100]:
                _, core_path, _, _ = self.__get_size_paths(method)
                core_size_df = pd.read_csv(core_path)
                core_size = self.__get_size(core_size_df, selectivity)
                method_rows.append((selectivity, core_size))
            methods_df.append(pd.DataFrame(method_rows, columns=['x', 'core_size'], index=[5, 19, 50, 100]))
        self.__plot_size(*methods_df)
        # Plot 3 included_columns: 0, 1, 2
        args.type = 'included-columns'
        methods_df = []
        for method in self.method_names_short:
            method_rows = []
            for included_columns in [0, 1, 2]:
                _, core_path, _, _ = self.__get_size_paths(method)
                core_size_df = pd.read_csv(core_path)
                core_size = self.__get_size(core_size_df, included_columns)
                method_rows.append((included_columns, core_size))
            methods_df.append(pd.DataFrame(method_rows, columns=['x', 'core_size'], index=[0, 1, 2]))
        self.__plot_size(*methods_df)
        
    def __plot_size_single(self):
        (filter_text1, filter_val1), (filter_text2, filter_val2) = args.get_filter_for_size_df()
        def get_size(size_df: pd.DataFrame) -> float:
            row = size_df[(size_df[filter_text1] == filter_val1) & (size_df[filter_text2] == filter_val2)]
            if row.empty:
                print(f'No data for {size_df}')
                exit(1)
            return row['size'].iloc[-1]
        fig, ax = plt.subplots(figsize=(3, 4))
        for i, method in enumerate(['base', 'join', 'merged']):
            _, core_path, _, _ = self.__get_size_paths(method)
            core_size_df = pd.read_csv(core_path)
            core_size = get_size(core_size_df)
            ax.bar(i, core_size, color=self.colors[i], label=self.method_names[i])
            
        ax.set_xticks([0, 1, 2], self.method_names, rotation=60, ha='right')
        ax.set_ylabel('Size (GB)')
        fig.tight_layout()
        fig.savefig(os.path.join(args.get_dir(), f'size.png'), dpi=300)
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("Reaggregator")

    parser.add_argument(
        '--suffix', type=str, required=False, help='Suffix of the dir names', default='')
    parser.add_argument(
        '--rocksdb', type=bool, required=False, help='Use RocksDB', default=False
    )
    parser.add_argument(
        '--outer_join', type=bool, required=False, help='Use outer join', default=False
    )
    
    # Parse the arguments
    parser.parse_args(namespace=args)
    
    print(args)
    reagg = Reaggregator(None)
    reagg.plot_size()