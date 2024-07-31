import pandas as pd
from typing import List
import os
import re

class DataProcessor:
    def __init__(self, file_paths: List[str]):
        self.file_paths: List[str] = file_paths
        self.data_frames: List[pd.DataFrame] = []

    def __load_data(self) -> None:
        for file in self.file_paths:
            df = pd.read_csv(file)
            for label, data in df.items():
                df[label] = pd.to_numeric(data, errors='coerce')
            self.data_frames.append(df)

    def get_combined_data(self) -> pd.DataFrame:
        self.__load_data()
        keys: List[str] = []
        for path in self.file_paths:
            dirname = os.path.dirname(path)
            dirname = os.path.basename(dirname)
            print(dirname)
            pattern = r'^(\w+)-'
            matches = re.match(pattern, dirname)
            keys.append(matches.group(1))
            
        combined_df = pd.concat(self.data_frames, keys=keys, names=['Source', 'Index'])
        return combined_df
