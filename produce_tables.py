from process_data import read_collected_data, DEFAULT_COLUMNS, DEFAULT_SELECTIVITY, DEFAULT_UPDATE_SIZE
from draw import safe_loc, get_storage_indexing_values, METHODS, STORAGES, TX_TYPES, compute_heatmap
import pandas as pd

leanstore_tx, rocksdb_tx, size_df = read_collected_data()

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
    fn = "charts/cpu_utilization.csv"
    df.to_csv(fn)
    return fn

def lsm_speedup():
    # Heatmap Data Preparation
    def heatmap_fn(storage, method):
        tx_df = rocksdb_tx if storage == STORAGES[-1] else leanstore_tx
        read_col = "sst_read_per_tx" if storage == STORAGES[-1] else "ssd_reads_per_tx"
        write_col = "sst_write_per_tx" if storage == STORAGES[-1] else "ssd_writes_per_tx"
        read_val = safe_loc(tx_df, get_storage_indexing_values(storage, method, "write"), read_col)
        write_val = safe_loc(tx_df, get_storage_indexing_values(storage, method, "write"), write_col)
        return read_val / (read_val + write_val)
    
    heatmap = compute_heatmap(
        STORAGES[1:], # rows
        METHODS, # columns
        heatmap_fn
    )
    
    X = [m.capitalize() for m in METHODS]
    Y = [
        safe_loc(rocksdb_tx, get_storage_indexing_values(STORAGES[2], m, "write"), "tput") /
        safe_loc(leanstore_tx, get_storage_indexing_values(STORAGES[1], m, "write"), "tput")
        for m in METHODS
    ]
    fn = "charts/lsm_speedup.csv"
    with open(fn, "w") as f:
        f.write("Method,Speedup,Read ratio in b-trees,Read ratio in lsm-forests\n")  
        for i, m in enumerate(METHODS):
            read_ratios = [f"{r*100:.2f}%" for r in heatmap[:,i]]
            speedup = f"{Y[i]:.2f}x"
            f.write(f"{m.capitalize()},{speedup},{','.join(read_ratios)}\n")
    return fn

def csv2textab(csv_fn):
    with open(csv_fn, "r") as f:
        lines = f.readlines()
    headers = lines[0].strip().split(",")
    rows = [l.strip().split(",") for l in lines[1:]]
    with open(csv_fn.replace(".csv", ".tex"), "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|" + "|".join(["c" for _ in headers]) + "|}\n")
        f.write("\\hline\n")
        formatted_headers = [r"\textbf{" + h.capitalize() + "}" for h in headers]
        f.write(" & ".join(formatted_headers) + "\\\\\n")
        f.write("\\hline\n")
        for row in rows:
            items = [r.replace("%", "\%") for r in row]
            items[0] = r"\textbf{" + items[0].capitalize() + "}"
            f.write(" & ".join(items) + "\\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

if __name__ == "__main__":
    fn1 = get_cpu_utilization()
    fn2 = lsm_speedup()
    csv2textab(fn1)
    csv2textab(fn2)
            
            
            