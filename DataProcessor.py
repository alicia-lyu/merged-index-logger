import pandas as pd
from typing import Callable, List, Tuple
import os
import numpy as np
from SeriesPlotter import find_stabilization_point
from args import args

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
        return self.__merge_unique_settings_df(self.data_frames)

    def get_agg(self) -> pd.DataFrame:
        # Calculate 4 aggregates: TX throughput, Reads per TX, Writes per TX, CPU GHz
        if args.rocksdb is False:
            read_col = 'SSDReads/TX'
            write_col = 'SSDWrites/TX'
        else:
            read_col = 'SSTRead(ms)/TX'
            write_col = 'SSTWrite(ms)/TX'
        
        stable_start = 0
        for col in ['OLTP TX', read_col, write_col, 'GHz', 'Cycles/TX']:
            for p, df in zip(self.file_paths, self.data_frames):
                # print(p)
                _, start = find_stabilization_point(60, 10, df[col])
                if 'scan' not in p and args.in_memory is False:
                    stable_start = max(stable_start, start)
        print(f'Stable start: {stable_start}')
            
        data = {
            'TXs/s': self.__merge_unique_settings_val([ df['OLTP TX'].iloc[stable_start:].mean() for df in self.data_frames ]),
            'IO/TX': self.__merge_unique_settings_val([ 
                df[read_col].iloc[stable_start:].mean() + df[write_col].iloc[stable_start:].mean() 
                for df in self.data_frames
            ]),
            'GHz': self.__merge_unique_settings_val([ df['GHz'].iloc[stable_start:].mean() for df in self.data_frames ]),
            'Cycles/TX': self.__merge_unique_settings_val([ df['Cycles/TX'].iloc[stable_start:].mean() for df in self.data_frames ])
        }
            
        agg_df = pd.DataFrame(data, index=self.__get_unique_settings())
        
        return agg_df
    
    def __get_unique_settings(self):
        unique_settings = {}
        for i, p in enumerate(self.file_paths):
            parent_dir = os.path.basename(os.path.dirname(p))
            if parent_dir not in unique_settings.keys():
                unique_settings[parent_dir] = [i]
            else:
                unique_settings[parent_dir].append(i)
        return unique_settings
    
    def __merge_unique_settings_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        unique_settings = self.__get_unique_settings()
        print("Files grouped into unique settings:", unique_settings)
        if isinstance(dfs[0], pd.DataFrame):
            merged_dfs = []
            for setting, indices in unique_settings.items():
                dfs_to_merge = [dfs[i].values for i in indices]
                stacked = np.stack(dfs_to_merge, axis=0)
                means = pd.DataFrame(np.mean(stacked, axis=0), columns=dfs[0].columns)
                merged_dfs.append(means)
            combined_df = pd.concat(merged_dfs, keys=unique_settings.keys(), names=['Source', 'Index'])
            return combined_df
    
    def __merge_unique_settings_val(self, values: List[float]) -> List[float]:
        unique_settings = self.__get_unique_settings()
        print("Files grouped into unique settings:", unique_settings)
        merged_means = []
        for setting, indices in unique_settings.items():
            values_to_merge = [values[i] for i in indices]
            merged_means.append(np.mean(values_to_merge))
        return merged_means
        