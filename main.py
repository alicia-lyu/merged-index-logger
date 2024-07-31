from DataProcessor import DataProcessor
from Plotter import Plotter
import os, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plotter")

    parser.add_argument(
        '--dram_gib', type=int, required=False, help='', default=1)
    parser.add_argument(
        '--target_gib', type=int, required=False, help='', default=4)
    parser.add_argument(
        '--read_pct', type=int, required=False, help='')
    parser.add_argument(
        '--write_pct', type=int, required=False, help='')
    parser.add_argument(
        '--scan_pct', type=int, required=False, help='')
    parser.add_argument(
        '--type', type=str, required=False, help='read, write, or scan')
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.read_pct is None or args.write_pct is None or args.scan_pct is None:
        assert(args.type is not None)
        if args.type == 'read':
            args.read_pct = 100
            args.write_pct = 0
            args.scan_pct = 0
        elif args.type == 'write':
            args.read_pct = 0
            args.write_pct = 100
            args.scan_pct = 0
        elif args.type == 'scan':
            args.read_pct = 0
            args.write_pct = 0
            args.scan_pct = 100
        else:
            raise ValueError('Invalid type')
    else:
        if args.read_pct == 100 and args.write_pct == 0 and args.scan_pct == 0:
            args.type = 'read'
        elif args.read_pct == 0 and args.write_pct == 100 and args.scan_pct == 0:
            args.type = 'write'
        elif args.read_pct == 0 and args.write_pct == 0 and args.scan_pct == 100:
            args.type = 'scan'
        else:
            args.type = f"custom-{args.read_pct}-{args.scan_pct}-{args.write_pct}"
    
    join_file_path = f'./join-{args.dram_gib}-{args.target_gib}-{args.read_pct}-{args.scan_pct}-{args.write_pct}/log_sum.csv'
    merged_file_path = f'./merged-{args.dram_gib}-{args.target_gib}-{args.read_pct}-{args.scan_pct}-{args.write_pct}/log_sum.csv'
    
    file_paths = [join_file_path, merged_file_path]
    processor = DataProcessor(file_paths)
    combined_data = processor.get_combined_data()
    
    dir_name = f'{args.type}-{args.target_gib}g'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    plotter = Plotter(combined_data, dir_name)
    plotter.plot_all_charts()
