from process_data import read_collected_data
from draw import safe_loc, get_storage_indexing_values, METHODS, STORAGES
import pandas as pd

leanstore_tx, rocksdb_tx, size_df = read_collected_data()

def get_cpu_utilization():
    headers = STORAGES
    rows = []
    for method in METHODS:
        row = []
        for storage in STORAGES:
            indexing_values = [get_storage_indexing_values(storage, method, tx) for tx in ["read-locality", "scan"]]
            all_cpu_data = [safe_loc(rocksdb_tx if storage == STORAGES[2] else leanstore_tx, i, "utilized_cpus") for i in indexing_values]
            all_cpu_utilization = [c/4 for c in all_cpu_data]
            row.append(f"{min(all_cpu_utilization) * 100:.2f}%--{max(all_cpu_utilization) * 100:.2f}%")
        rows.append(row)
    df = pd.DataFrame(rows, columns=headers, index=METHODS)
    fn = "charts/cpu_utilization.csv"
    df.to_csv(fn)
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
    csv2textab(fn1)
            
            
            