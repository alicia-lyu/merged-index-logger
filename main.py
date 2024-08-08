from typing import Tuple
from DataProcessor import DataProcessor
from SeriesPlotter import SeriesPlotter
from AggPlotter import AggPlotter
import os, argparse, re

DEFAULT_DRAM_GIB = 1
DEFAULT_TARGET_GIB = 4
DEFAULT_TYPE = 'all-tx'
DEFAULT_SUFFIX = ''
DEFAULT_IN_MEMORY = False
DEFAULT_ROCKSDB = False

class Args():
    def __init__(self) -> None:
        self.dram_gib: float = DEFAULT_DRAM_GIB
        self.target_gib: int = DEFAULT_TARGET_GIB
        self.type: str = DEFAULT_TYPE
        self.suffix: str = DEFAULT_SUFFIX
        self.in_memory: bool = DEFAULT_IN_MEMORY
        self.rocksdb: bool = DEFAULT_ROCKSDB
        
    def __str__(self) -> str:
        return f'dram_gib: {self.dram_gib}, target_gib: {self.target_gib}, type: {self.type}, suffix: {self.suffix}, in_memory: {self.in_memory}, rocksdb: {self.rocksdb}'
    
    def get_default(self) -> Tuple[int, str]:
        if self.type == 'selectivity':
            default_val = 100
            suffix = '-sel'
        elif self.type == 'update-size':
            default_val = 5
            suffix = '-size'
        elif self.type == 'included-columns':
            default_val = 1
            suffix = '-col'
        else:
            print(f'Invalid title: {args.type}')
            exit(1)
        return default_val, suffix
    
    def get_title(self) -> str:
        if self.type == 'selectivity':
            return 'Selectivity'
        elif self.type == 'update-size':
            return 'Update Size'
        elif self.type == 'included-columns':
            return 'Included Columns'
        elif self.type == 'all-tx':
            return 'All Transaction Types'
        elif self.type == 'read':
            return 'Read Transactions'
        elif self.type == 'write':
            return 'Write Transactions'
        elif self.type == 'scan':
            return 'Scan Transactions'
        else:
            print(f'Invalid title: {args.type} to be set as title')
            exit(1)
            
    def get_xlabel(self) -> str:
        if self.type == 'selectivity':
            return 'Selectivity (%)'
        elif self.type == 'update-size':
            return 'Update Size ([x, 3x] lines in one order)'
        elif self.type == 'included-columns':
            return 'Included Columns' # expected to have 0 and 1 ticked as none and all
        elif self.type == 'all-tx':
            return 'Transaction Type'
        else:
            print(f'Invalid title: {args.type} to be set as xlabel')
            exit(1)
            
    def get_dir(self) -> str:
        dir_name = f'plots-{args.dram_gib}-{args.target_gib}-{args.type}{args.suffix}'
        if self.in_memory:
            dir_name += '-in-memory' 
        if self.rocksdb:
            dir_name += '-rocksdb'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            
    def get_pattern(self) -> str:
        common_prefix = r'(join|merged)-' + f'{args.dram_gib:.1f}|{int(args.dram_gib):d}' + f'-{args.target_gib}-'
        print(f'Common prefix: {common_prefix}')
        
        pattern: str = ''
        
        match args.type:
            case 'read':
                pattern = common_prefix + r'read' + args.suffix + r'$'
            case 'write':
                pattern = common_prefix + r'write' + args.suffix + r'$'
            case 'scan': # Throughput too low. Only plot aggregates.
                pattern = common_prefix + r'scan' + args.suffix + r'$'
            case 'all-tx':
                pattern = common_prefix + r'(read|write|scan)' + args.suffix + r'$'
            case 'update-size':
                pattern = common_prefix + r'write(-size\d+)?$'
            case 'selectivity': # Too many stats. Only plot aggregates.
                pattern = common_prefix + r'(read|write|scan)(-sel\d+)?$'
            case 'included-columns':
                pattern = common_prefix + r'(read|write|scan)(-col\d+)?$'
            case _:
                raise ValueError(f'Invalid type: {args.type}')
            
        return pattern

args = Args()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plotter")

    parser.add_argument(
        '--dram_gib', type=float, required=False, help='', default=DEFAULT_DRAM_GIB)
    parser.add_argument(
        '--target_gib', type=int, required=False, help='', default=DEFAULT_TARGET_GIB)
    parser.add_argument(
        '--type', type=str, required=True, help='read, write, scan, update-size, selectivity, all-tx')
    parser.add_argument(
        '--suffix', type=str, required=False, help='Suffix of the dir names', default=DEFAULT_SUFFIX)
    parser.add_argument(
        '--in_memory', type=bool, required=False, help='Load all data in memory', default=DEFAULT_IN_MEMORY)
    parser.add_argument(
        '--rocksdb', type=bool, required=False, help='Use RocksDB', default=DEFAULT_ROCKSDB
    )
    
    # Parse the arguments
    parser.parse_args(namespace=args)
    print(args)
    
    pattern = args.get_pattern()
    # iterate on all directories in the current directory
    file_paths = []
    for dir in os.listdir('.'):
        matches = re.match(pattern, dir)
        if matches is not None:
            assert(os.path.isdir(dir))
            for file in os.listdir(dir):
                if file == 'log_sum.csv':
                    file_paths.append(f'{dir}/{file}')
                    break
            else:
                print(f'Skipping directory {dir} because it does not contain log_sum.csv.')
    
    print(file_paths)
    if (len(file_paths) == 0):
        print(f'No directories found with pattern {pattern}. Exiting...')
        exit(1)
        
    dir_name = args.get_dir()
    
    processor = DataProcessor(file_paths)
        
    if args.type == 'read' or args.type == 'write' or args.type == 'update-size' and len(file_paths) < 6:
        combined_data = processor.get_combined_data()    
        plotter = SeriesPlotter(combined_data, dir_name)
        plotter.plot_all_charts()

    if args.type in ['read', 'write', 'scan']:
        print(f'For aggregates, use --type all-tx')
        exit(0)
    
    agg_df = processor.get_agg()
    plotter = AggPlotter(agg_df, dir_name, file_paths)
    plotter.plot()
