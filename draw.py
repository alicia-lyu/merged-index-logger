from process_data import collect_size_data, collect_tx_data, DEFAULT_COLUMNS, DEFAULT_SELECTIVITY, DEFAULT_UPDATE_SIZE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# Constants
COLORS = {
    "yellow": "#ffbe0b",
    "pink": "#ff006e",
    "blue": "#3a86ff",
    "orange": "#fb5607",
    "purple": "#8338ec",
    "green": "#00a86b"
}
DEFAULT_JOIN = "inner"
DEFAULT_DRAM = 1
DEFAULT_TARGET = 4
RATIO_VMIN = 0
RATIO_VMAX = 2
QUERY_CMAP = "RdYlGn"
SIZE_CMAP = "coolwarm"

# Data Collection
leanstore_tx, rocksdb_tx = collect_tx_data()
size_df = collect_size_data()

# Helper Functions
def get_storage_indexing_values(storage, method, tx):
    common_values = (method, tx, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN, DEFAULT_DRAM, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)
    if storage == "memory-resident b-tree":
        return (method, tx, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN, 16, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)
    return common_values

def get_col_sel_indexing_values(included_columns, selectivity, method, tx):
    join_type = "outer" if selectivity == "outer" else DEFAULT_JOIN
    return (method, tx, selectivity, included_columns, join_type, DEFAULT_DRAM, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)

def safe_loc(data, indexing_values, column):
    try:
        result = data.loc[indexing_values, column]
        if isinstance(result, np.float64):
            return result
        else:
            return result.iloc[-1]
    except KeyError as e:
        print(f"KeyError for indexing values {indexing_values}: {e}")
        return 1

def compute_heatmap(data, row_values, col_values, value_fn):
    heatmap = np.ones((len(row_values), len(col_values)))
    for i, col_val in enumerate(col_values):
        for j, row_val in enumerate(row_values):
            heatmap[j, i] = value_fn(row_val, col_val)
    return heatmap

def draw_heatmap(heatmap, ax, cmap, vmin, vmax, xticks, yticks, xlabel, ylabel):
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=30, ha="right")
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

def add_colorbar(fig, im, label):
    bar = fig.colorbar(im, ax=fig.axes, label=label)
    bar.formatter = mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    ticks = bar.get_ticks()
    if 1 not in ticks:
        ticks = np.concatenate(([1], ticks))
        ticks.sort()
    bar.set_ticks(ticks)
    bar.ax.axhline(y=1, color="black", linestyle="--", linewidth=1)

def draw_bars(fig, ax, X, Y, ylabel):
    fig.set_size_inches(2 + len(X) * 0.6, 4)
    rotation = len(max(X, key=len)) > 3
    
    ax.bar(X, Y, color=COLORS["green"])
    ax.set_xticks(range(len(X)))
    ax.set_xticklabels(X, rotation=30 if rotation else 0, ha="right" if rotation else "center")
    y_ticks = ax.get_yticks()
    if 1 not in y_ticks:
        y_ticks = np.concatenate(([1], y_ticks))
        y_ticks.sort()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y * 100):d}%" for y in y_ticks])
    ax.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel(ylabel)

# Plot Functions
# X: in-memory b-tree, disk-based b-tree, LSM
# Y: Speed up in maintenance MergedIndex/BaseTables (bars) 
def maintenance_match():
    X = ["memory-resident\nb-tree", "disk-resident\nb-tree", "lsm-forest"]
    Y = [
        safe_loc(leanstore_tx, get_storage_indexing_values(storage, "merged index", "write"), "tput") /
        safe_loc(leanstore_tx, get_storage_indexing_values(storage, "traditional indexes", "write"), "tput")
        for storage in ["memory-resident b-tree", "disk-resident b-tree", "lsm-forest"]
    ]
    fig, ax = plt.subplots()
    draw_bars(fig, ax, X, Y, ylabel="Ratio of Transaction Throughput\nMerged Index / Traditional Indexes")
    fig.tight_layout()
    fig.savefig("maintenance_match.png", dpi=300)

def query_heatmaps(baseline):
    row_values = [5, 19, 50, 100, "outer"]
    col_values = [0, 2, 1]
    heatmap1 = compute_heatmap(
        leanstore_tx, row_values, col_values,
        lambda sel, col: safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, "merged index", "read-locality"), "tput") /
        safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, baseline, "read-locality"), "tput")
    )
    heatmap2 = compute_heatmap(
        leanstore_tx, row_values, col_values,
        lambda sel, col: safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, "merged index", "scan"), "tput") /
        safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, baseline, "scan"), "tput")
    )
    storage_values = ["memory-resident b-tree", "disk-resident b-tree", "lsm-forest"]
    tx_values = ["read-locality", "scan"]
    heatmap3 = compute_heatmap(
        leanstore_tx, storage_values, tx_values,
        lambda storage, tx: safe_loc(leanstore_tx, get_storage_indexing_values(storage, "merged index", tx), "tput") /
        safe_loc(leanstore_tx, get_storage_indexing_values(storage, baseline, tx), "tput")
    )
    return heatmap1, heatmap2, heatmap3

def draw_query_heatmap(baseline):
    heatmap1, heatmap2, heatmap3 = query_heatmaps(baseline)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    xticks = ["none", "selected", "all"]
    yticks = ["InnerJoin,\nSO=5%", "InnerJoin,\nSO=19%", "InnerJoin,\nSO=50%", "InnerJoin,\nSO=100%", "OuterJoin"]

    im1 = draw_heatmap(heatmap1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, xticks, yticks, "Included Columns", "Join Type and Join Selectivity")
    im2 = draw_heatmap(heatmap2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, xticks, yticks, "Included Columns", "Join Type and Join Selectivity")
    tx_xticks = ["Point Lookup", "Range Scan"]
    storage_yticks = ["Memory-resident\nb-tree", "Disk-resident\nb-tree", "LSM"]
    im3 = draw_heatmap(heatmap3, ax3, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, tx_xticks, storage_yticks, "Query Type", "Storage Type")

    ax1.set_title("Point Lookup")
    ax2.set_title("Range Scan")
    add_colorbar(fig, im3, f"Ratio of Transaction Throughput\nMerged Index / {baseline.capitalize()}")
    fig.savefig(f"query_vs_{baseline}.png", dpi=300)

def get_heatmap_figsize(row_len, col_len):
    return (col_len * 0.6 + 3, row_len * 0.6 + 2)

# Fig 3.1 Space overhead heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: b-tree, lsm-forest 
# - Z: Space overhead MergedIndex/BaseTables
def space_overhead():
    row_values = ["b-tree", "lsm-forest"]
    col_values = [0, 2, 1]
    heatmap = compute_heatmap(
        size_df, row_values, col_values,
        lambda storage, col: safe_loc(size_df, tuple(["merged index", storage, DEFAULT_TARGET, DEFAULT_SELECTIVITY, col, DEFAULT_JOIN]), "core_size") /
        safe_loc(size_df, tuple(["traditional indexes", storage, DEFAULT_TARGET, DEFAULT_SELECTIVITY, col, DEFAULT_JOIN]), "core_size")
    )
    fig, ax = plt.subplots(figsize=get_heatmap_figsize(len(row_values), len(col_values)), layout="constrained")
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX, ["none", "selected", "all"], ["b-tree", "lsm-forest"], "Included Columns", "Storage Type")
    add_colorbar(fig, im, "Size Ratio of Storage Structure\nMerged Index / Traditional Indexes")
    fig.savefig("space_overhead.png", dpi=300)

# Fig 3.2: Compression effect heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, OuterJoin
# - Z: Compression effect MergedIndex/MaterializedJoin
def compression_effect():
    row_values = [5, 19, 50, 100, "outer"]
    col_values = [0, 2, 1]
    heatmap = compute_heatmap(
        size_df, row_values, col_values,
        lambda sel, col: safe_loc(size_df, tuple(["merged index", "b-tree", DEFAULT_TARGET, sel if sel != "outer" else DEFAULT_SELECTIVITY, col, DEFAULT_JOIN if sel != "outer" else "outer"]), "core_size") /
        safe_loc(size_df, tuple(["materialized join view", "b-tree", DEFAULT_TARGET, sel if sel != "outer" else DEFAULT_SELECTIVITY, col, DEFAULT_JOIN if sel != "outer" else "outer"]), "core_size")
    )
    fig, ax = plt.subplots(figsize=get_heatmap_figsize(len(row_values), len(col_values)), layout="constrained")
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX, ["none", "selected", "all"], ["InnerJoin,\nSO=5%", "InnerJoin,\nSO=19%", "InnerJoin,\nSO=50%", "InnerJoin,\nSO=100%", "OuterJoin"], "Included Columns", "Join Type and Join Selectivity")
    add_colorbar(fig, im, "Size Ratio of Storage Structure\nMerged Index / Materialized Join View")
    fig.savefig("compression_effect.png", dpi=300)

# Execution
if __name__ == "__main__":
    maintenance_match()
    draw_query_heatmap("traditional indexes")
    draw_query_heatmap("materialized join view")
    space_overhead()
    compression_effect()