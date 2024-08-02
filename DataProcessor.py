import pandas as pd
from typing import Callable, List, Tuple
import os
import numpy as np
from SeriesPlotter import find_stabilization_point

class DataProcessor:
    def __init__(self, file_paths: List[str]):
        self.file_paths: List[str] = file_paths
        self.data_frames: List[pd.DataFrame] = []
        self.__load_data()

    def __load_data(self) -> None:
        min_rows = float('inf')
        for file in self.file_paths:
            df = pd.read_csv(file)
            for label, data in df.items():
                df[label] = pd.to_numeric(data, errors='coerce')
            # Replace inf with NaN and drop rows containing NaN
            self.data_frames.append(df)
            min_rows = min(min_rows, len(df))
        
        print(f'Minimum rows: {min_rows}')
        for i in range(len(self.data_frames)):
            self.data_frames[i] = self.data_frames[i].tail(min_rows) # discard the first few rows to make all dataframes have the same length
            self.data_frames[i]['t'] = self.data_frames[i]['t'] - self.data_frames[i]['t'].iloc[0]

    def get_combined_data(self) -> pd.DataFrame:
        keys: List[str] = []
        for path in self.file_paths:
            dirname = os.path.dirname(path)
            dirname = os.path.basename(dirname)
            keys.append(dirname)
            
        combined_df = pd.concat(self.data_frames, keys=keys, names=['Source', 'Index'])
        return combined_df

    def get_agg(self) -> pd.DataFrame:
        # Calculate 4 aggregates: TX throughput, Reads per TX, Writes per TX, CPU GHz
        data = {
            'TXs/s': [ find_stabilization_point(60, 10, df['OLTP TX'])[0] for df in self.data_frames ],
            'Reads/TX': [ find_stabilization_point(60, 10, df['SSDReads/TX'])[0] for df in self.data_frames ],
            'Writes/TX': [ find_stabilization_point(60, 10, df['SSDWrites/TX'])[0] for df in self.data_frames ],
            'GHz': [ find_stabilization_point(60, 10, df['GHz'])[0] for df in self.data_frames ]
        }
        print(data)
        agg_df = pd.DataFrame(data, index=[os.path.basename(os.path.dirname(path)) for path in self.file_paths])
        agg_df.replace([np.nan], 0, inplace=True)
        return agg_df