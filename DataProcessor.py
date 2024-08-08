import pandas as pd
from typing import Callable, List, Tuple
import os
import numpy as np
from SeriesPlotter import find_stabilization_point
from main import args

class DataProcessor:
    def __init__(self, file_paths: List[str]) -> None:
        self.file_paths: List[str] = file_paths
        self.data_frames: List[pd.DataFrame] = []
        self.__load_data()

    def __load_data(self) -> None:
        # TODO
        print(f'Loading data from {len(self.file_paths)} files...')
        min_rows = float('inf')
        files_to_remove = []
        for file in self.file_paths:
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                print(f'Empty data in {file}. Skipping...')
                files_to_remove.append(file)
                continue
            if args.in_memory is False and len(df) < 50:
                print(f'Not enough data in {file}. Skipping...')
                files_to_remove.append(file)
                continue
            for label, data in df.items():
                df[label] = pd.to_numeric(data, errors='coerce')
            # Replace inf with NaN and drop rows containing NaN
            if args.in_memory:
                df = df[(df["W MiB"] == 0) & (df["R MiB"] == 0)]
            self.data_frames.append(df)
            min_rows = min(min_rows, len(df))
        
        self.file_paths = [file for file in self.file_paths if file not in files_to_remove]
        
        print(f'Minimum rows: {min_rows}')
        self.data_frames = [df.tail(min_rows) for df in self.data_frames]
        self.data_frames = [df.assign(t=df['t'] - df['t'].iloc[0]) for df in self.data_frames]

    def get_combined_data(self) -> pd.DataFrame:
        combined_df = pd.concat(self.data_frames, keys=[os.path.basename(os.path.dirname(path)) for path in self.file_paths], names=['Source', 'Index'])
        return combined_df

    def get_agg(self) -> pd.DataFrame:
        # Calculate 4 aggregates: TX throughput, Reads per TX, Writes per TX, CPU GHz
        if args.rocksdb is False:
            read_col = 'SSDReads/TX'
            write_col = 'SSDWrites/TX'
        else:
            read_col = 'SSTRead(ms)/TX'
            write_col = 'SSTWrite(ms)/TX'
            
        data = {
            'TXs/s': [ find_stabilization_point(60, 10, df['OLTP TX'])[0] for df in self.data_frames ],
            'IO/TX': [ 
                find_stabilization_point(60, 10, df[read_col])[0] + find_stabilization_point(60, 10, df[write_col])[0] for df in self.data_frames ],
            'GHz': [ find_stabilization_point(60, 10, df['GHz'])[0] for df in self.data_frames ],
            'Cycles/TX': [ find_stabilization_point(60, 10, df['Cycles/TX'])[0] for df in self.data_frames ],
        }
        agg_df = pd.DataFrame(data, 
                              index=[os.path.basename(os.path.dirname(path)) for path in self.file_paths])
        agg_df.replace([np.nan], 0, inplace=True)
        return agg_df