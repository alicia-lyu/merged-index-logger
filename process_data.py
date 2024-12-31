import os, re
import pandas as pd
from collections import defaultdict

DEFAULT_SELECTIVITY = 100
DEFAULT_COLUMNS = 1
DEFAULT_UPDATE_SIZE = 5

def collect_tx_data():
    leanstore_rows = []
    rocksdb_rows = []
    params = ["method", "tx", "selectivity", "included_columns", "join", "dram_gib", "target_gib", "update_size"]
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
                print("Ignoring experiment: ", exp)
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
                print("Error in: ", os.path.join(d, exp, f), " with message: ", e)
                continue
            rows.append(row)
    leanstore_df = pd.DataFrame(leanstore_rows, columns=leanstore_headers)
    leanstore_df.set_index(params, inplace=True)
    rocksdb_df = pd.DataFrame(rocksdb_rows, columns=rocksdb_headers)
    rocksdb_df.set_index(params, inplace=True)
    leanstore_df.to_csv("leanstore_tx_data.csv")
    rocksdb_df.to_csv("rocksdb_tx_data.csv")
    return leanstore_df, rocksdb_df
            
def synthesize(path):
    df = pd.read_csv(path)
    if len(df) < 120:
        raise ValueError("Insufficient data in: ", path, " with length: ", len(df))
    df = df.tail(len(df) // 2) # Take the last half of the data
    tput = df["OLTP TX"].mean()
    if "rocksdb" not in path:
        w_mib = df["W MiB"].mean()
        r_mib = df["R MiB"].mean()
        cycles_per_tx = df["Cycles/TX"].mean()
        utilized_cpus = df["Utilized CPUs"].mean()
        cpu_time_per_tx = df["CPUTime/TX (ms)"].mean()
        ssd_reads_per_tx = df["SSDReads/TX"].mean()
        ssd_writes_per_tx = df["SSDWrites/TX"].mean()
        return tput, w_mib, r_mib, cycles_per_tx, utilized_cpus, cpu_time_per_tx, ssd_reads_per_tx, ssd_writes_per_tx
    else:
        sst_read_per_tx = df["SSTRead(ms)/TX"].mean()
        sst_write_per_tx = df["SSTWrite(ms)/TX"].mean()
        cpu_time_per_tx = df["CPUTime/TX (ms)"].mean()
        utilized_cpus = df["Utilized CPUs"].mean()
        return tput, sst_read_per_tx, sst_write_per_tx, cpu_time_per_tx, utilized_cpus

# method,storage,dram,target,selectivity,included_columns,core_size,rest_size,additional_size,core_time,rest_time,additional_time
def collect_size_data():
    params = ["method", "storage", "target", "selectivity", "included_columns", "join"]
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
        time_f = os.path.join(d, "time.csv")
        if not os.path.exists(size_f) or not os.path.exists(time_f):
            print("Missing size or time file in: ", d)
            continue
        size_df = pd.read_csv(size_f)
        # Consolidate the data
        size_dir = defaultdict(dict)
        for i, row in size_df.iterrows():
            size_dir[row["config"]][row["table(s)"]] = (row["size"], row["time(ms)"])
        # Convert to rows
        for config, tables in size_dir.items():
            configs = config.split("|")
            if (len(configs) == 4):
                join = "inner"
            elif (len(configs) == 5):
                join = "outer"
            else:
                print("Unknown join type: ", config)
                continue
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
            row = [method, storage, int(target), int(selectivity), int(included_columns), join, float(core_size), float(rest_size), float(additional_size), int(core_time), int(rest_time), int(additional_time)]
            rows.append(row)
    size_df = pd.DataFrame(rows, columns=headers)
    size_df.set_index(params, inplace=True)
    size_df.to_csv("size_data.csv")
    return size_df
    
if __name__ == "__main__":
    collect_tx_data()
    collect_size_data()