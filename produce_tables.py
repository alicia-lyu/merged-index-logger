from process_data import collect_size_data, collect_tx_data, DEFAULT_COLUMNS, DEFAULT_SELECTIVITY, DEFAULT_UPDATE_SIZE
from draw import safe_loc, get_storage_indexing_values, METHODS, STORAGES, TX_TYPES
import pandas as pd

leanstore_tx, rocksdb_tx = collect_tx_data()
size_df = collect_size_data()

def get_cpu_utilization():
    headers = STORAGES
    rows = []
    for method in METHODS:
        row = []
        for storage in STORAGES:
            indexing_values = [get_storage_indexing_values(storage, method, tx) for tx in TX_TYPES]
            all_cpu_data = [safe_loc(rocksdb_tx if storage == STORAGES[2] else leanstore_tx, i, "utilized_cpus") for i in indexing_values]
            all_cpu_utilization = [c/4 for c in all_cpu_data]
            row.append(f"{min(all_cpu_utilization) * 100:.2f}%--{max(all_cpu_utilization) * 100:.2f}%")
        rows.append(row)
    df = pd.DataFrame(rows, columns=headers, index=METHODS)
    df.to_csv("charts/cpu_utilization.csv")
    
if __name__ == "__main__":
    get_cpu_utilization()
            
            
            