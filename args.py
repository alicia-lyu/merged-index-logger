from typing import Tuple
import os, re

DEFAULT_DRAM_GIB = 1
DEFAULT_TARGET_GIB = 4
DEFAULT_TYPE = 'all-tx'
DEFAULT_SUFFIX = ''
DEFAULT_IN_MEMORY = False
DEFAULT_ROCKSDB = False

class Args():
    def __init__(self) -> None:
        self.dram_gib: float = DEFAULT_DRAM_GIB
        self.target_gib: int = DEFAULT_TARGET_GIB
        self.type: str = DEFAULT_TYPE
        self.suffix: str = DEFAULT_SUFFIX
        self.in_memory: bool = DEFAULT_IN_MEMORY
        self.rocksdb: bool = DEFAULT_ROCKSDB
        self.outer_join: bool = False
        self.pattern = None

    
    def get_non_default_args(self):
        default_args = {'dram_gib': DEFAULT_DRAM_GIB, 'target_gib': DEFAULT_TARGET_GIB, 'type': DEFAULT_TYPE, 'suffix': DEFAULT_SUFFIX, 'in_memory': DEFAULT_IN_MEMORY, 'rocksdb': DEFAULT_ROCKSDB}
        return {k: v for k, v in self.__dict__.items() if k is not 'pattern' and v != default_args[k]}
    
    def get_pattern(self) -> str:
        if self.pattern is None:
            self.pattern = self.format_pattern()
        return self.pattern
        
    def __str__(self) -> str:
        return f'dram_gib: {self.dram_gib}, target_gib: {self.target_gib}, type: {self.type}, suffix: {self.suffix}, in_memory: {self.in_memory}, rocksdb: {self.rocksdb}'
    
    def get_default(self) -> Tuple[int, str]:
        if self.type == 'selectivity':
            default_val = 100
            suffix = '-sel'
        elif self.type == 'update-size':
            default_val = 5
            suffix = '-size'
        elif self.type == 'included-columns':
            default_val = 1
            suffix = '-col'
        else:
            raise ValueError(f'Invalid type: {self.type}')
        return default_val, suffix
    
    def get_filter_for_size_df(self): # Based on suffix
        if self.suffix == '':
            if self.type == 'selectivity':
                return 'included_columns', 1
            elif self.type == 'included-columns':
                return 'selectivity', 100
            elif self.type != 'all-tx':
                raise ValueError(f'Invalid type: {self.type} to call get_filter_for_size_df')

        if self.type == 'all-tx':
            filters = {'selectivity': 100, 'included_columns': 1}
            suffix_text, suffix_val = self.get_suffix_val()
            filters[suffix_text] = suffix_val
            return ('selectivity', filters['selectivity']),('included_columns', filters['included_columns'])
            
        return self.get_suffix_val()
    
    def get_title(self) -> str:
        if self.type == 'selectivity':
            return 'Selectivity'
        elif self.type == 'update-size':
            return 'Update Size'
        elif self.type == 'included-columns':
            return 'Included Columns'
        elif self.type == 'all-tx':
            return 'All Transaction Types'
        elif self.type == 'read':
            return 'Read Transactions'
        elif self.type == 'write':
            return 'Write Transactions'
        elif self.type == 'scan':
            return 'Scan Transactions'
        else:
            print(f'Invalid title: {self.type} to be set as title')
            exit(1)
            
    def get_xlabel(self) -> str:
        if self.type == 'selectivity':
            return 'SO (%)'
        elif self.type == 'update-size':
            return 'Update Size ([x, 3x] lines in one order)'
        elif self.type == 'included-columns':
            return 'Included Columns' # expected to have 0 and 1 ticked as none and all
        elif self.type == 'all-tx':
            return 'Transaction Type'
        elif self.type == 'manual':
            return ''
        else:
            print(f'Invalid title: {self.type} to be set as xlabel')
            exit(1)
            
    def get_dir(self) -> str:
        if self.dram_gib < self.target_gib:
            prefix = ''
        elif self.dram_gib > self.target_gib * 2:
            assert(self.in_memory)
            prefix = 'in-memory-'
        else:
            prefix = f'{self.dram_gib}-{self.target_gib}-'
        dir_name = f'plots/{prefix}{self.type}{self.suffix}'
        if self.outer_join:
            dir_name += '-outer'
        if self.rocksdb:
            dir_name += '-rocksdb'
        os.makedirs(dir_name, exist_ok=True)
        return dir_name
            
    def format_pattern(self) -> str:
        pattern = r'(join|merged|base)/' + f'({self.dram_gib:.1f})'.replace('.', '\.') + f'-{self.target_gib}-'
        if self.rocksdb:
            pattern = 'rocksdb_' + pattern
            
        pattern = r'^' + pattern
        
        match self.type:
            case 'read':
                pattern += r'read'
            case 'write':
                pattern += r'write'
            case 'scan': # Throughput too low. Only plot aggregates.
                pattern += r'scan'
            case 'update-size':
                pattern += r'write'
            case _: # all-tx, selectivity, included-columns
                pattern += r'(read-locality|read|write|scan)'
        
        if self.outer_join:
            pattern += r'-outer'
            
        match self.type:
            case 'update-size':
                pattern += r'(-size\d+)?'
            case 'selectivity':
                pattern += r'(-sel\d+)?'
            case 'included-columns':
                pattern += r'(-col\d+)?'
            case _: # all-tx, read, write, scan
                pass
        
        pattern += self.suffix + r'$'
        
        return pattern
    
    def get_suffix_val(self):
        if self.suffix == '':
            return '', 0
        matches = re.match(r'([^\d]+)(\d+)', self.suffix)
        suffix_val = int(matches.group(2))
        suffix_text = matches.group(1)
        match suffix_text:
            case '-col':
                return 'included_columns', suffix_val
            case '-sel':
                return 'selectivity', suffix_val
            case '-size':
                return 'update_size', suffix_val
            case _:
                raise ValueError(f'Invalid suffix: {self.suffix}')
    
    def parse_path(self, path: str):
        matches = re.match(self.get_pattern(), path)
        method = matches.group(1)
        tx_type = matches.group(3)
        if len(matches.groups()) > 3 and matches.group(4) is not None:
            extra = matches.group(4)
            extra_matches = re.match(r'[^\d]+(\d+)', extra)
            extra_val = int(extra_matches.group(1))
            return method, tx_type, extra_val
        else:
            try:
                default_val, suffix = self.get_default()
                return method, tx_type, default_val
            except ValueError:
                return method, tx_type
        
args = Args()