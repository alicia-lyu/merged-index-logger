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
            raise ValueError(f'Invalid type: {args.type}')
        return default_val, suffix
    
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
            print(f'Invalid title: {args.type} to be set as title')
            exit(1)
            
    def get_xlabel(self) -> str:
        if self.type == 'selectivity':
            return 'Selectivity as represented by SO (%)'
        elif self.type == 'update-size':
            return 'Update Size ([x, 3x] lines in one order)'
        elif self.type == 'included-columns':
            return 'Included Columns' # expected to have 0 and 1 ticked as none and all
        elif self.type == 'all-tx':
            return 'Transaction Type'
        else:
            print(f'Invalid title: {args.type} to be set as xlabel')
            exit(1)
            
    def get_dir(self) -> str:
        dir_name = f'plots-{args.dram_gib}-{args.target_gib}-{args.type}{args.suffix}'
        if self.in_memory:
            dir_name += '-in-memory' 
        if self.rocksdb:
            dir_name += '-rocksdb'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
            
    def get_pattern(self) -> str:
        common_prefix = r'(join|merged|base)-' + f'({args.dram_gib:.1f}|{int(args.dram_gib):d})' + f'-{args.target_gib}-'
        print(f'Common prefix: {common_prefix}')
        if self.rocksdb:
            common_prefix = 'rocksdb_' + common_prefix
        
        pattern: str = ''
        
        match args.type:
            case 'read':
                pattern = common_prefix + r'read' + args.suffix + r'$'
            case 'write':
                pattern = common_prefix + r'write' + args.suffix + r'$'
            case 'scan': # Throughput too low. Only plot aggregates.
                pattern = common_prefix + r'scan' + args.suffix + r'$'
            case 'all-tx':
                pattern = common_prefix + r'(read-locality|read|write|scan)' + args.suffix + r'$'
            case 'update-size':
                pattern = common_prefix + r'write(-size\d+)?$'
            case 'selectivity': # Too many stats. Only plot aggregates.
                pattern = common_prefix + r'(read-locality|read|write|scan)(-sel\d+)?$'
            case 'included-columns':
                pattern = common_prefix + r'(read-locality|read|write|scan)(-col\d+)?$'
            case _:
                raise ValueError(f'Invalid type: {args.type}')
            
        return pattern
    
    def parse_path(self, path: str):
        matches = re.match(self.get_pattern(), path)
        method = matches.group(1)
        tx_type = matches.group(3)
        if matches.group(4) is not None:
            extra = matches.group(4)
            extra_matches = re.match(r'.*(\d+)', extra)
            extra_val = int(extra_matches.group(1))
            return method, tx_type, extra_val
        else:
            try:
                default_val, suffix = self.get_default()
                return method, tx_type, default_val
            except ValueError:
                return method, tx_type
        
args = Args()