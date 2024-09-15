from typing import Tuple
from DataProcessor import DataProcessor
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
        '--type', type=str, required=False, help='read, write, scan, update-size, selectivity, all-tx, included-columns', default=args.DEFAULT_TYPE)
    parser.add_argument(
        '--suffix', type=str, required=False, help='Suffix of the dir names', default=args.DEFAULT_SUFFIX)
    parser.add_argument(
        '--in_memory', type=bool, required=False, help='Load all data in memory', default=args.DEFAULT_IN_MEMORY)
    parser.add_argument(
        '--rocksdb', type=bool, required=False, help='Use RocksDB', default=args.DEFAULT_ROCKSDB
    )
    parser.add_argument(
        '--outer_join', type=bool, required=False, help='Use outer join', default=False
    )
    
    # Parse the arguments
    parser.parse_args(namespace=args.args)
    print(args.args)
    
    pattern = args.args.get_pattern()
    # iterate on all directories in the current directory
    file_paths = []
    for dir0 in os.listdir('.'):
        if not os.path.isdir(dir0):
            continue
        for dir1 in os.listdir(dir0):
            path = f'{dir0}/{dir1}'
            matches = re.match(pattern, path)
            if matches is not None:
                assert(os.path.isdir(path))
                for file in os.listdir(path):
                    if file.endswith('sum.csv'):
                        file_paths.append(f'{path}/{file}')
    
    print(f"Found {len(file_paths)} directories with pattern {pattern}")
    if (len(file_paths) == 0):
        print(f'No directories found with pattern {pattern}. Exiting...')
        exit(1)
        
    dir_name = args.args.get_dir()
    
    processor = DataProcessor(file_paths)

    if args.args.type in ['read', 'write', 'scan']:
        print(f'For aggregates, use --type all-tx')
        exit(0)
    
    agg_df = processor.get_agg()
    plotter = AggPlotter(agg_df, dir_name)
    plotter.plot()
