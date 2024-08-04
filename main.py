from typing import List
from DataProcessor import DataProcessor
from SeriesPlotter import SeriesPlotter
from AggPlotter import AggPlotter
import os, argparse, re

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plotter")

    parser.add_argument(
        '--dram_gib', type=int, required=False, help='', default=1)
    parser.add_argument(
        '--target_gib', type=int, required=False, help='', default=4)
    parser.add_argument(
        '--type', type=str, required=True, help='read, write, scan, update-size, selectivity')
    
    # Parse the arguments
    args = parser.parse_args()
    common_prefix = r'(join|merged)-' + f'{args.dram_gib}-{args.target_gib}-'
    
    pattern: str = ''
    
    match args.type:
        case 'read':
            pattern = common_prefix + r'read$'
        case 'write':
            pattern = common_prefix + r'write$'
        case 'scan': # Throughput too low. Only plot aggregates.
            pattern = common_prefix + r'scan$'
        case 'update-size':
            pattern = common_prefix + r'write(-size\d+)?$'
        case 'selectivity': # Too many stats. Only plot aggregates.
            pattern = common_prefix + r'(read|write|scan)(-sel\d+)?$'
        case 'included-columns':
            pattern = common_prefix + r'(read|write|scan)(-col\d+)?$'
        case _:
            raise ValueError(f'Invalid type: {args.type}')
    
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
        
    dir_name = f'plots-{args.dram_gib}-{args.target_gib}-{args.type}'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    processor = DataProcessor(file_paths)
        
    if args.type != 'scan' and args.type != 'selectivity' and args.type != 'included-columns' and len(file_paths) < 6:
        combined_data = processor.get_combined_data()    
        plotter = SeriesPlotter(combined_data, dir_name)
        plotter.plot_all_charts()

    agg_df = processor.get_agg()
    plotter = AggPlotter(agg_df, dir_name, args.type)
    plotter.plot_agg()
