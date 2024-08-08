from typing import List
from DataProcessor import DataProcessor
from SeriesPlotter import SeriesPlotter
from AggPlotter import AggPlotter
import os, argparse, re

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plotter")

    parser.add_argument(
        '--dram_gib', type=float, required=False, help='', default=1)
    parser.add_argument(
        '--target_gib', type=int, required=False, help='', default=4)
    parser.add_argument(
        '--type', type=str, required=True, help='read, write, scan, update-size, selectivity')
    parser.add_argument(
        '--suffix', type=str, required=False, help='Suffix of the dir names', default='')
    
    # Parse the arguments
    args = parser.parse_args()
    common_prefix = r'(join|merged)-' + (f'{args.dram_gib:.1f}' if args.dram_gib < 1 else f'{int(args.dram_gib):d}') + f'-{args.target_gib}-'
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
        
    dir_name = f'plots-{args.dram_gib}-{args.target_gib}-{args.type}{args.suffix}' 
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    processor = DataProcessor(file_paths)
        
    if args.type == 'read' or args.type == 'write' or args.type == 'update-size' and len(file_paths) < 6:
        combined_data = processor.get_combined_data()    
        plotter = SeriesPlotter(combined_data, dir_name)
        plotter.plot_all_charts()

    agg_df = processor.get_agg()
    plotter = AggPlotter(agg_df, dir_name, args.type)
    if args.type == 'selectivity' or args.type == 'included-columns' or args.type == 'update-size':
        plotter.plot_x(file_paths)
    else:
        plotter.plot_type(file_paths)
