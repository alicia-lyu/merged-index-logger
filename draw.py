from process_data import collect_size_data, collect_tx_data, DEFAULT_COLUMNS, DEFAULT_SELECTIVITY, DEFAULT_UPDATE_SIZE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

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

# params = ["method", "tx", "selectivity", "included_columns", "join", "dram_gib", "target_gib", "update_size"]
leanstore_tx, rocksdb_tx = collect_tx_data()
# params = ["method", "storage", "dram", "target", "selectivity", "included_columns", "join"]
size_df = collect_size_data()

def get_storage_indexing_values(storage, method, tx):
    if storage == "memory-resident b-tree":
        return (method, tx, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN, 16, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)
    if storage == "disk-resident b-tree" or storage == "lsm-forest":
        return (method, tx, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN, DEFAULT_DRAM, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)

# X: in-memory b-tree, disk-based b-tree, LSM
# Y: Speed up in maintenance MergedIndex/BaseTables (bars) 
def maintenance_match():
    X = ["memory-resident\nb-tree", "disk-resident\nb-tree", "lsm-forest"]
    Y = [
        leanstore_tx.loc[get_storage_indexing_values("memory-resident b-tree", "merged index", "write"), "tput"].squeeze() / leanstore_tx.loc[get_storage_indexing_values("memory-resident b-tree", "traditional indexes", "write"), "tput"].squeeze(),
        leanstore_tx.loc[get_storage_indexing_values("disk-resident b-tree", "merged index", "write"), "tput"].squeeze() / leanstore_tx.loc[get_storage_indexing_values("disk-resident b-tree", "traditional indexes", "write"), "tput"].squeeze(),
        rocksdb_tx.loc[get_storage_indexing_values("lsm-forest", "merged index", "write"), "tput"].squeeze() / rocksdb_tx.loc[get_storage_indexing_values("lsm-forest", "traditional indexes", "write"), "tput"].squeeze()
    ]
    fig, ax = plt.subplots()
    draw_bars(fig, ax, X, Y)
    ax.set_ylabel("Ratio of Transaction Throughput\nMerged Index / Traditional Indexes")
    fig.tight_layout()
    fig.savefig("maintenance_match.png", dpi=300)

def get_col_sel_indexing_values(included_columns, selectivity, method, tx):
    if selectivity == "outer":
        return (method, tx, DEFAULT_SELECTIVITY, included_columns, "outer", DEFAULT_DRAM, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)
    else:
        return (method, tx, selectivity, included_columns, DEFAULT_JOIN, DEFAULT_DRAM, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)

def query_heatmaps(baseline):
    heatmap1 = np.ones((5, 3))
    heatmap2 = np.ones((5, 3))
    for i, included_columns in enumerate([0, 2, 1]):
        for j, selectivity in enumerate([5, 19, 50, 100, "outer"]):
            try:
                heatmap1[j, i] = leanstore_tx.loc[get_col_sel_indexing_values(included_columns, selectivity, "merged index", "read-locality"), "tput"].squeeze() / leanstore_tx.loc[get_col_sel_indexing_values(included_columns, selectivity, baseline, "read-locality"), "tput"].squeeze()
            except KeyError as e:
                print(e)
                heatmap1[j, i] = 1
            try:
                heatmap2[j, i] = leanstore_tx.loc[get_col_sel_indexing_values(included_columns, selectivity, "merged index", "scan"), "tput"].squeeze() / leanstore_tx.loc[get_col_sel_indexing_values(included_columns, selectivity, baseline, "scan"), "tput"].squeeze()
            except KeyError as e:
                print(e)
                heatmap2[j, i] = 1
    
    heatmap3 = np.ones((3, 2))
    for i, tx in enumerate(["read-locality", "scan"]):
        for j, storage in enumerate(["memory-resident b-tree", "disk-resident b-tree", "lsm-forest"]):
            try:
                heatmap3[j, i] = leanstore_tx.loc[get_storage_indexing_values(storage, "merged index", tx), "tput"].squeeze() / leanstore_tx.loc[get_storage_indexing_values(storage, baseline, tx), "tput"].squeeze()
            except KeyError as e:
                print(e)
                heatmap3[j, i] = 1
    print(heatmap1, heatmap2, heatmap3)
    return heatmap1, heatmap2, heatmap3

# 3 subplots, all heatmap green-red
# Fig 2.1 heat map green-red
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, OuterJoin
# Z: Speed up of point lookup MergedIndex/BaseTables
# Fig 2.2 heat map green-red
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, OuterJoin
# Z: Speed up of range scan MergedIndex/BaseTables
# Fig 2.3 heat map green-red
# - X: Point lookup, range scan
# - Y: in-memory b-tree, disk-based b-tree, LSM
# Z: Speed up of MergedIndex/BaseTables
def draw_query_heatmap(baseline):
    heatmap1, heatmap2, heatmap3 = query_heatmaps(baseline)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout="constrained", figsize=(12, 4), gridspec_kw={"width_ratios": [len(heatmap1[0]), len(heatmap2[0]), len(heatmap3[0])]})

    v_min = min(heatmap1.min(), heatmap2.min(), heatmap3.min())
    v_max = max(heatmap1.max(), heatmap2.max(), heatmap3.max())
    diff_max = max(abs(v_min - 1), abs(v_max - 1))
    v_min = 1 - diff_max
    v_max = 1 + diff_max
    im1 = ax1.imshow(heatmap1, cmap="RdYlGn", vmin=v_min, vmax=v_max)
    im2 = ax2.imshow(heatmap2, cmap="RdYlGn", vmin=v_min, vmax=v_max)

    # Set x and y ticks for ax1 and ax2
    for ax in [ax1, ax2]:
        ax.set_xticks(range(3))
        ax.set_xticklabels(["none", "selected", "all"], rotation=30, ha="right")
        ax.set_xlabel("Included Columns")
        ax.set_yticks(range(5))
        ax.set_yticklabels(["InnerJoin,\nSO=5%", "InnerJoin,\nSO=19%", "InnerJoin,\nSO=50%", "InnerJoin,\nSO=100%", "OuterJoin"])
    
    # Set x and y ticks for ax3
    im3 = ax3.imshow(heatmap3, cmap="RdYlGn", vmin=v_min, vmax=v_max)
    ax3.set_xticks(range(2))
    ax3.set_xticklabels(["Point Lookup", "Range Scan"], rotation=30, ha="right")
    ax3.set_xlabel("Query Type")
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(["Memory-resident\nb-tree", "Disk-resident\nb-tree", "lsm-forest"])
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)

    # Add colorbar
    bar = fig.colorbar(im3, ax=[ax1, ax2, ax3], label=f"Ratio of Transaction Throughput\nMerged Index / {baseline.capitalize()}")
    ticks = bar.get_ticks()
    if 1 not in ticks:
        ticks = np.concatenate(([1], ticks))
        ticks.sort()
    bar.formatter = mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    bar.update_ticks()
    cbar_ax = bar.ax  # Access the colorbar's axis
    cbar_ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

    # Save the figure
    fig.savefig(f"query_vs_{baseline}.png", dpi=300)

# Fig 3.1 Space overhead heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: b-tree, lsm-forest 
# - Z: Space overhead MergedIndex/BaseTables
def space_overhead():
    pass

# Fig 3.2: Compression effect heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, OuterJoin
# - Z: Compression effect MergedIndex/MaterializedJoin
def compression_effect():
    pass

# Fig 3.3: Full DB space
# - X: Three methods
# - Y: Space requirements (3 parts)
def full_db_space():
    pass

def draw_bars(fig, ax, X, Y):
    # fig size adaptive to len(X):
    fig.set_size_inches(2 + len(X) * 0.7, 4)
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
    
if __name__ == "__main__":
    maintenance_match()
    draw_query_heatmap("traditional indexes")
    draw_query_heatmap("materialized join view")