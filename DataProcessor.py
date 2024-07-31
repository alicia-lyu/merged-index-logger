import pandas as pd
from typing import List
import os
import re

class DataProcessor:
    def __init__(self, file_paths: List[str]):
        self.file_paths: List[str] = file_paths
        self.data_frames: List[pd.DataFrame] = []

    def __load_data(self) -> None:
        min_rows = float('inf')
        for file in self.file_paths:
            df = pd.read_csv(file)
            for label, data in df.items():
                df[label] = pd.to_numeric(data, errors='coerce')
            self.data_frames.append(df)
            min_rows = min(min_rows, len(df))
        
        print(f'Minimum rows: {min_rows}')
        for i in range(len(self.data_frames)):
            self.data_frames[i] = self.data_frames[i].tail(min_rows) # discard the first few rows to make all dataframes have the same length
            self.data_frames[i]['t'] = self.data_frames[i]['t'] - self.data_frames[i]['t'].iloc[0]

    def get_combined_data(self) -> pd.DataFrame:
        self.__load_data()
        keys: List[str] = []
        for path in self.file_paths:
            dirname = os.path.dirname(path)
            dirname = os.path.basename(dirname)
            # print(dirname)
            # pattern = r'^(\w+)' + r'-\d+' * 2 + r'-(.+)$' # It is assumed that dram_gib and target_gib are consistent across all directories
            # matches = re.match(pattern, dirname)
            # keys.append(matches.group(1) + '-' + matches.group(2))
            keys.append(dirname)
            
        combined_df = pd.concat(self.data_frames, keys=keys, names=['Source', 'Index'])
        return combined_df
