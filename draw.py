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

RATIO_VMIN = 0
RATIO_VMAX = 2

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
    print(heatmap1)
    print(heatmap2)
    print(heatmap3)
    max_val = max(heatmap1[np.isfinite(heatmap1)].max(), heatmap2[np.isfinite(heatmap2)].max(), heatmap3[np.isfinite(heatmap3)].max(), 2)
    heatmap1[np.isinf(heatmap1)] = max_val
    heatmap2[np.isinf(heatmap2)] = max_val
    heatmap3[np.isinf(heatmap3)] = max_val
    return heatmap1, heatmap2, heatmap3

QUERY_CMAP = "PiYG"

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
    print(f"Drawing query heatmap for {baseline}")
    heatmap1, heatmap2, heatmap3 = query_heatmaps(baseline)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout="constrained", figsize=(12, 4), gridspec_kw={"width_ratios": [len(heatmap1[0]), len(heatmap2[0]), len(heatmap3[0])]})
    
    im1 = draw_heatmap(heatmap1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX)
    im2 = draw_heatmap(heatmap2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX)

    # Set x and y ticks for ax1 and ax2
    for ax in [ax1, ax2]:
        ax.set_xticklabels(["none", "selected", "all"], rotation=30, ha="right")
        ax.set_xlabel("Included Columns")
        ax.set_yticklabels(["InnerJoin,\nSO=5%", "InnerJoin,\nSO=19%", "InnerJoin,\nSO=50%", "InnerJoin,\nSO=100%", "OuterJoin"])
    ax1.set_title("Point Lookup")
    ax2.set_title("Range Scan")
    
    # Set x and y ticks for ax3
    im3 = draw_heatmap(heatmap3, ax3, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX)
    ax3.set_xticklabels(["Point Lookup", "Range Scan"], rotation=30, ha="right")
    ax3.set_xlabel("Query Type")
    ax3.set_yticklabels(["Memory-resident\nb-tree", "Disk-resident\nb-tree", "lsm-forest"])

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

def draw_heatmap(heatmap, ax, cmap, vmin=None, vmax=None):
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
    ax.set_xticks(range(len(heatmap[0])))
    ax.set_yticks(range(len(heatmap)))
    return im

SIZE_CMAP = "coolwarm"

# Fig 3.1 Space overhead heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: b-tree, lsm-forest 
# - Z: Space overhead MergedIndex/BaseTables
def space_overhead():
    # params = ["method", "storage", "target", "selectivity", "included_columns", "join"]
    heatmap = np.ones((2, 3))
    for i, included_columns in enumerate([0, 2, 1]):
        for j, storage in enumerate(["b-tree", "lsm-forest"]):
            shared_indexing_values = [storage, DEFAULT_TARGET, DEFAULT_SELECTIVITY, included_columns, DEFAULT_JOIN]
            try:
                indexing_values1 = tuple(['merged index'] + shared_indexing_values)
                indexing_values2 = tuple(['traditional indexes'] + shared_indexing_values)
                print(indexing_values1)
                print(indexing_values2)
                heatmap[j, i] = size_df.loc[indexing_values1, "core_size"].mean() / size_df.loc[indexing_values2, "core_size"].mean()
            except KeyError as e:
                print(e, indexing_values1, indexing_values2)
                heatmap[j, i] = 1
    fig, ax = plt.subplots()
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX)
    ax.set_xticklabels(["none", "selected", "all"], rotation=30, ha="right")
    ax.set_xlabel("Included Columns")
    ax.set_yticklabels(["b-tree", "lsm-forest"])
    
    bar = fig.colorbar(im, ax=ax, label="Size Ratio of Storage Structure\nMerged Index / Traditional Indexes")
    ticks = bar.get_ticks()
    if 1 not in ticks:
        ticks = np.concatenate(([1], ticks))
        ticks.sort()
    bar.formatter = mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    bar.update_ticks()
    cbar_ax = bar.ax  # Access the colorbar's axis
    cbar_ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    fig.tight_layout()
    fig.savefig("space_overhead.png", dpi=300)

# Fig 3.2: Compression effect heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, OuterJoin
# - Z: Compression effect MergedIndex/MaterializedJoin
def compression_effect():
    heatmap = np.ones((5, 3))
    for i, included_columns in enumerate([0, 2, 1]):
        for j, selectivity in enumerate([5, 19, 50, 100, "outer"]):
            shared_indexing_values = ['b-tree', DEFAULT_TARGET, selectivity if selectivity != "outer" else DEFAULT_SELECTIVITY, included_columns, DEFAULT_JOIN if selectivity != "outer" else "outer"]
            indexing_values1 = tuple(['merged index'] + shared_indexing_values)
            indexing_values2 = tuple(['materialized join view'] + shared_indexing_values)
            try:
                heatmap[j, i] = size_df.loc[indexing_values1, "core_size"].mean() / size_df.loc[indexing_values2, "core_size"].mean()
            except KeyError as e:
                print(e, indexing_values1, indexing_values2)
                heatmap[j, i] = 1
    fig, ax = plt.subplots()
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX)
    ax.set_xticklabels(["none", "selected", "all"], rotation=30, ha="right")
    ax.set_xlabel("Included Columns")
    ax.set_yticklabels(["InnerJoin,\nSO=5%", "InnerJoin,\nSO=19%", "InnerJoin,\nSO=50%", "InnerJoin,\nSO=100%", "OuterJoin"])
    
    bar = fig.colorbar(im, ax=ax, label="Size Ratio of Storage Structure\nMerged Index / Materialized Join View")
    ticks = bar.get_ticks()
    if 1 not in ticks:
        ticks = np.concatenate(([1], ticks))
        ticks.sort()
    bar.formatter = mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    bar.update_ticks()
    cbar_ax = bar.ax  # Access the colorbar's axis
    cbar_ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    fig.tight_layout()
    fig.savefig("compression_effect.png", dpi=300)

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
    space_overhead()
    compression_effect()