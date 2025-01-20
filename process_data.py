import os, re
import pandas as pd
import numpy as np
from collections import defaultdict

DEFAULT_SELECTIVITY = 100
DEFAULT_COLUMNS = 1
DEFAULT_UPDATE_SIZE = 5

TX_INDEX_COLS = ["method", "tx", "selectivity", "included_columns", "join", "dram_gib", "target_gib", "update_size"]

def collect_tx_data():
    leanstore_rows = []
    rocksdb_rows = []
    params = TX_INDEX_COLS
    file_identifiers = ["timestamp", "file_path"]
    leanstore_metrics = ["tput", "w_mib", "r_mib", "cycles_per_tx", "utilized_cpus", "cpu_time_per_tx", "ssd_reads_per_tx", "ssd_writes_per_tx"]
    rocksdb_metrics = ["tput", "sst_read_per_tx", "sst_write_per_tx", "cpu_time_per_tx", "utilized_cpus"]
    leanstore_headers = params + file_identifiers + leanstore_metrics
    rocksdb_headers = params + file_identifiers + rocksdb_metrics
    
    leading_pattern = r"^([\d\.]+)-(\d+)-(read-locality|read|write|scan)"
    parameter_pattern = r"-(sel|col|size)(\d+)"
    for d in os.listdir("."):
        if not os.path.isdir(d):
            continue
        if "base" in d:
            method = "traditional indexes"
        elif "merged" in d:
            method = "merged index"
        elif "join" in d:
            method = "materialized join view"
        else:
            print("Ignoring directory: ", d)
            continue
        if "rocksdb" in d:
            rows = rocksdb_rows
        else:
            rows = leanstore_rows
        for exp in os.listdir(d):
            leading_match = re.match(leading_pattern, exp)
            if leading_match is None:
                print("Ignoring experiment: ", os.path.join(d, exp))
                continue
            dram_gib, target_gib, tx = leading_match.groups()
            dram_gib = float(dram_gib)
            target_gib = int(target_gib)
            selectivity = DEFAULT_SELECTIVITY
            included_columns = DEFAULT_COLUMNS
            update_size = DEFAULT_UPDATE_SIZE
            parameters = re.findall(parameter_pattern, exp)
            for p in parameters:
                p_type, p_value = p
                p_value = int(p_value)
                if p_type == "sel":
                    selectivity = p_value
                elif p_type == "col":
                    included_columns = p_value
                elif p_type == "size":
                    update_size = p_value
            if "outer" in exp:
                join = "outer"
            else:
                join = "inner"
            summary_files = [f for f in os.listdir(os.path.join(d, exp)) if f.endswith("_sum.csv")]
            summary_files.sort() # sort by timestamp
            if len(summary_files) == 0:
                print("No summary files found in: ", os.path.join(d, exp))
                continue
            f = summary_files[-1]
            timestamp = f.split("_")[0]
            try:
                metrics_data = synthesize(os.path.join(d, exp, f))
                if "rocksdb" in d:
                    tput, sst_read_per_tx, sst_write_per_tx, cpu_time_per_tx, utilized_cpus = metrics_data
                    row = [method, tx, selectivity, included_columns, join, dram_gib, target_gib, update_size, timestamp, os.path.join(d, exp, f), tput, sst_read_per_tx, sst_write_per_tx, cpu_time_per_tx, utilized_cpus]
                else:
                    tput, w_mib, r_mib, cycles_per_tx, utilized_cpus, cpu_time_per_tx, ssd_reads_per_tx, ssd_writes_per_tx = metrics_data
                    row = [method, tx, selectivity, included_columns, join, dram_gib, target_gib, update_size, timestamp, os.path.join(d, exp, f), tput, w_mib, r_mib, cycles_per_tx, utilized_cpus, cpu_time_per_tx, ssd_reads_per_tx, ssd_writes_per_tx]
            except ValueError as e:
                print("Error in:", os.path.join(d, exp, f), "with message:", e)
                continue
            rows.append(row)
    leanstore_df = pd.DataFrame(leanstore_rows, columns=leanstore_headers)
    leanstore_df.set_index(params, inplace=True)
    rocksdb_df = pd.DataFrame(rocksdb_rows, columns=rocksdb_headers)
    rocksdb_df.set_index(params, inplace=True)
    leanstore_df.to_csv("synthesis_leanstore.csv")
    rocksdb_df.to_csv("synthesis_rocksdb.csv")
    return leanstore_df, rocksdb_df
            
def synthesize(path):
    df = pd.read_csv(path)
    if len(df) < 120:
        raise ValueError("Insufficient data in: ", path, " with length: ", len(df))
    df = df.tail(len(df) // 2) # Take the last half of the data
    
    def safe_index_mean(col):
        nonlocal df
        try:
            return df[col].mean()
        except KeyError as e:
            print("KeyError with column:", col, "in file:", path, ". Message:", e)
            return np.nan
        
    tput = safe_index_mean("OLTP TX")
    if "rocksdb" not in path:
        w_mib = safe_index_mean("W MiB")
        r_mib = safe_index_mean("R MiB")
        cycles_per_tx = safe_index_mean("Cycles/TX")
        utilized_cpus = safe_index_mean("Utilized CPUs")
        cpu_time_per_tx = safe_index_mean("CPUTime/TX (ms)")
        ssd_reads_per_tx = safe_index_mean("SSDReads/TX")
        ssd_writes_per_tx = safe_index_mean("SSDWrites/TX")
        return tput, w_mib, r_mib, cycles_per_tx, utilized_cpus, cpu_time_per_tx, ssd_reads_per_tx, ssd_writes_per_tx
    else:
        sst_read_per_tx = safe_index_mean("SSTRead(ms)/TX")
        sst_write_per_tx = safe_index_mean("SSTWrite(ms)/TX")
        cpu_time_per_tx = safe_index_mean("CPUTime/TX (ms)")
        utilized_cpus = safe_index_mean("Utilized CPUs")
        return tput, sst_read_per_tx, sst_write_per_tx, cpu_time_per_tx, utilized_cpus

SIZE_INDEX_COLS = ["method", "storage", "target", "selectivity", "included_columns", "join"]

def collect_size_data():
    params = SIZE_INDEX_COLS
    metrics = ["core_size", "rest_size", "additional_size", "core_time", "rest_time", "additional_time"]
    headers = params + metrics
    rows = []
    for d in os.listdir("."):
        if not os.path.isdir(d):
            continue
        if "base" in d:
            method = "traditional indexes"
        elif "merged" in d:
            method = "merged index"
        elif "join" in d:
            method = "materialized join view"
        else:
            print("Ignoring directory: ", d)
            continue
        if "rocksdb" in d:
            storage = "lsm-forest"
        else:
            storage = "b-tree"
        size_f = os.path.join(d, "size.csv")
        size_outer_f = os.path.join(d, "size_outer.csv")
        rows_inner = process_size_file(size_f)
        rows_outer = process_size_file(size_outer_f)
        if rows_inner is not None:
            rows.extend([[method, storage] + r for r in rows_inner])
        if rows_outer is not None:
            rows.extend([[method, storage] + r for r in rows_outer])
            
    size_df = pd.DataFrame(rows, columns=headers)
    size_df.set_index(params, inplace=True)
    size_df.to_csv("synthesis_size.csv")
    return size_df

def process_size_file(path):
    if not os.path.exists(path):
        print(f"Missing file {path}")
        return None
    size_df = pd.read_csv(path)
    # Consolidate the data
    size_dir = defaultdict(dict)
    for i, row in size_df.iterrows():
        size_dir[row["config"]][row["table(s)"]] = (row["size"], row["time(ms)"])
    # Convert to rows
    rows = []
    for config, tables in size_dir.items():
        configs = config.split("|")
        if "outer" in path:
            join = "outer"
        else:
            join = "inner"
        _, target, selectivity, included_columns = configs[:4]
        additional_time = 0
        additional_size = 0
        for table, (s, t) in tables.items():
            if table == "core":
                rest_size = s
                rest_time = t
            elif table == "stock+orderline_secondary" or table == "merged_index" or table == "join_results":
                core_size = s
                core_time = t
            else:
                print("Additional table found: ", table)
                additional_size = s
                additional_time = t
        row = [int(target), int(selectivity), int(included_columns), join, float(core_size), float(rest_size), float(additional_size), int(core_time), int(rest_time), int(additional_time)]
        rows.append(row)
    return rows

def read_collected_data():
    leanstore_df = pd.read_csv("synthesis_leanstore.csv", index_col=TX_INDEX_COLS)
    rocksdb_df = pd.read_csv("synthesis_rocksdb.csv", index_col=TX_INDEX_COLS)
    size_df = pd.read_csv("synthesis_size.csv", index_col=SIZE_INDEX_COLS)
    return leanstore_df, rocksdb_df, size_df
    
if __name__ == "__main__":
    collect_tx_data()
    collect_size_data()