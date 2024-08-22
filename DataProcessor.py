import pandas as pd
from typing import Callable, List, Tuple
import os
import numpy as np
from SeriesPlotter import find_stabilization_point
from args import args

'''
Read all csv files of time series data and use one of the following methods to process the data:
1. get_combined_data: 
    a. Merge all time series data by calculating a time series mean for the same setting
    b. Combine all time series (2D, index=time, columns=metrics) data into one DataFrame (3D, keys=source, index=time, columns=metrics)
2. get_agg: 
    a. Find a global stabilization point for all time series data
    b. For each time series, calculate the stabilized mean of the following metrics: TXs/s, IOs/TX, Utilized CPU (%), CPUTime/TX (ms)
    c. Merge all the means whose parent directories are the same.
    d. Reorganize the final means into a DataFrame (2D, index=source, columns=metrics)
'''
class DataProcessor:
    def __init__(self, file_paths: List[str]) -> None:
        self.file_paths: List[str] = file_paths
        self.data_frames: List[pd.DataFrame] = []
        self.__load_data()
        if args.rocksdb is False:
            self.read_col = 'SSDReads/TX'
            self.write_col = 'SSDWrites/TX'
        else:
            self.read_col = 'SSTRead(ms)/TX'
            self.write_col = 'SSTWrite(ms)/TX'

    def __load_data(self) -> None:
        print(f'Loading data from {len(self.file_paths)} files...')
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
            # print(df)
            self.data_frames.append(df)
        
        self.file_paths = [file for file in self.file_paths if file not in files_to_remove]

    def get_combined_data(self) -> pd.DataFrame:
        print(f"**************************************** Getting combined data ****************************************")
        dfs = []
        min_rows = min([len(df) for df in self.data_frames])
        for df in self.data_frames:
            dfs.append(df.iloc[:min_rows])
        return self.__merge_unique_settings_df(dfs)

    def get_agg(self) -> pd.DataFrame:
        print(f"**************************************** Getting aggregated data ****************************************")
        # Find the stabilization point
        min_rows = min([len(df) for df in self.data_frames])
        if args.in_memory:
            for i, df in enumerate(self.data_frames):
                df = df[(df["W MiB"] == 0) & (df["R MiB"] == 0)]
                min_rows = min(min_rows, len(df))
                self.data_frames[i] = df.iloc[len(df) - min_rows:]
            stable_start = min_rows // 2
        else:
            stable_start = 0
            for col in ['OLTP TX', self.read_col, self.write_col, 'Utilized CPUs', 'CPUTime/TX (ms)']:
                for p, df in zip(self.file_paths, self.data_frames):
                    # print(p, df)
                    _, start = find_stabilization_point(30, 30, df[col])
                    # TODO: Is it a good idea to only rely on read TXs for stabilization?
                    if args.in_memory is False and 'write' in p:
                        continue
                    elif 'scan' not in p:
                        if start > 200:
                            print(f'File {p} has a stabilization point at {start}. Skipping...')
                            continue
                        stable_start = max(stable_start, start)
        print(f'Stable start: {stable_start}')
            
        min_rows = min([len(df) for df in self.data_frames])
        aggs = []
        
        for i in range(len(self.data_frames)):
            agg = self.agg_each(i, stable_start, min_rows)
            aggs.append(agg)
            
        data = self.__merge_unique_settings_series(aggs)
        return data
    
    def agg_each(self, index, stable_start, min_rows):
        df = self.data_frames[index]
        p = self.file_paths[index]
        
        if 'scan' in p:
            start = 0
        else:
            start = stable_start
        
        result = []
        
        io_col = 'IOs/TX' if not args.rocksdb else 'IO time (ms)/TX'
        
        cols = ['TXs/s', io_col, 'Utilized CPU (%)', 'CPUTime/TX (ms)']
            
        for col in cols:
            tx = df['OLTP TX'].iloc[start:]
            if tx.sum() == 0:
                print(f'{p} has no transactions.')
                print(tx)
                return None
            if col == 'TXs/s':
                curtailed_tx = tx.iloc[:min_rows]
                mean_val = curtailed_tx.mean()
            elif col == 'Utilized CPU (%)':
                curtailed_ghz = df['Utilized CPUs'].iloc[start:min_rows]
                mean_val = curtailed_ghz.mean() / 4 * 100
            elif col == io_col:
                reads_per_tx = df[self.read_col].iloc[start:]
                reads = reads_per_tx * tx
                writes_per_tx = df[self.write_col].iloc[start:] 
                writes = writes_per_tx * tx
                read_mean = reads.sum() / tx.sum()
                write_mean = writes.sum() / tx.sum()
                mean_val = read_mean + write_mean
            else: # do not curtail
                cycles_per_tx = df['CPUTime/TX (ms)'].iloc[start:] 
                cycles = cycles_per_tx * tx
                mean_val = cycles.sum() / tx.sum()
            
            result.append(mean_val)
            
        # if 'scan' in p:
        #     print(f'{p} tx sum over {len(tx)} seconds: {tx.sum()}')
        
        result = pd.Series(result, index=cols)
        
        # print(f'{p} result:\n{result}')
                
        return result
        
    
    def __get_unique_settings(self):
        unique_settings = {}
        for i, p in enumerate(self.file_paths):
            parent_dir = os.path.basename(os.path.dirname(p))
            if parent_dir not in unique_settings.keys():
                unique_settings[parent_dir] = [i]
            else:
                unique_settings[parent_dir].append(i)
        print("Files grouped into unique settings:", unique_settings)
        return unique_settings
    
    def __merge_unique_settings_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        unique_settings = self.__get_unique_settings()
        merged_dfs = []
        for setting, indices in unique_settings.items():
            dfs_to_merge = [dfs[i].values for i in indices]
            stacked = np.stack(dfs_to_merge, axis=0)
            means = pd.DataFrame(np.mean(stacked, axis=0), columns=dfs[0].columns)
            merged_dfs.append(means)
        combined_df = pd.concat(merged_dfs, keys=unique_settings.keys(), names=['Source', 'Index'])
        return combined_df
    
    def __merge_unique_settings_series(self, series: List[pd.Series]) -> pd.DataFrame:
        unique_settings = self.__get_unique_settings()
        merged_series = []
        for setting, indices in unique_settings.items():
            stacked = np.stack([series[i] for i in indices], axis=0)
            mean_series = pd.Series(np.mean(stacked, axis=0))
            merged_series.append(mean_series)
        result = pd.concat(merged_series, axis=1).T
        result.index = unique_settings.keys()
        result.columns = series[0].index
        return result