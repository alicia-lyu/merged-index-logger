from typing import Tuple
from DataProcessor import DataProcessor
from SeriesPlotter import SeriesPlotter
from AggPlotter import AggPlotter
import os, argparse, re
import args

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plotter")

    parser.add_argument(
        '--dram_gib', type=float, required=False, help='', default=args.DEFAULT_DRAM_GIB)
    parser.add_argument(
        '--target_gib', type=int, required=False, help='', default=args.DEFAULT_TARGET_GIB)
    parser.add_argument(
        '--type', type=str, required=True, help='read, write, scan, update-size, selectivity, all-tx')
    parser.add_argument(
        '--suffix', type=str, required=False, help='Suffix of the dir names', default=args.DEFAULT_SUFFIX)
    parser.add_argument(
        '--in_memory', type=bool, required=False, help='Load all data in memory', default=args.DEFAULT_IN_MEMORY)
    parser.add_argument(
        '--rocksdb', type=bool, required=False, help='Use RocksDB', default=args.DEFAULT_ROCKSDB
    )
    
    # Parse the arguments
    parser.parse_args(namespace=args.args)
    print(args.args)
    
    pattern = args.args.get_pattern()
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
        
    dir_name = args.args.get_dir()
    
    processor = DataProcessor(file_paths)
        
    if args.args.type == 'read' or args.args.type == 'write' or args.args.type == 'update-size' and len(file_paths) < 6:
        combined_data = processor.get_combined_data()    
        plotter = SeriesPlotter(combined_data, dir_name)
        plotter.plot_all_charts()

    if args.args.type in ['read', 'write', 'scan']:
        print(f'For aggregates, use --type all-tx')
        exit(0)
    
    agg_df = processor.get_agg()
    plotter = AggPlotter(agg_df, dir_name, file_paths)
    plotter.plot()
