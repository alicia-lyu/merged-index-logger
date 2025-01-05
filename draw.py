from process_data import read_collected_data, DEFAULT_SELECTIVITY, DEFAULT_UPDATE_SIZE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import cmasher as cmr

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = 600
plt.rcParams["font.stretch"] = 'extra-condensed'
plt.rcParams["axes.labelweight"] = 600
plt.rcParams["axes.titleweight"] = 600


# Constants
COLORS = {
    "yellow": "#ffbe0b",
    "pink": "#ff006e",
    "blue": "#3a86ff",
    "orange": "#fb5607",
    "purple": "#8338ec",
    "green": "#00a86b"
}
METHODS = ["traditional indexes", "merged index", "materialized join view"]
STORAGES = ["memory-resident b-tree", "disk-resident b-tree", "lsm-forest"]
TX_TYPES = ["write", "read-locality", "scan"]
DEFAULT_JOIN = "inner"
DEFAULT_DRAM = 1
DEFAULT_TARGET = 4
RATIO_VMIN = 0
RATIO_VMAX = 2
QUERY_CMAP = cmr.guppy
SIZE_CMAP = cmr.prinsenvlag_r
DEFAULT_COLUMNS = 1 # 2

# Data Collection
leanstore_tx, rocksdb_tx, size_df = read_collected_data()

# Helper Functions
def get_storage_indexing_values(storage, method, tx):
    common_values = (method, tx, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN, DEFAULT_DRAM, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)
    if storage == STORAGES[0]:
        return (method, tx, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN, 16, DEFAULT_TARGET, DEFAULT_UPDATE_SIZE)
    return common_values

def get_col_sel_indexing_values(included_columns, selectivity, method, tx):
    join_type = "outer" if selectivity == "outer" else DEFAULT_JOIN
    selectivity = DEFAULT_SELECTIVITY if selectivity == "outer" else selectivity
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

def compute_heatmap(row_values, col_values, value_fn):
    heatmap = np.ones((len(row_values), len(col_values)))
    for i, col_val in enumerate(col_values):
        for j, row_val in enumerate(row_values):
            val = value_fn(row_val, col_val)
            if val is np.nan or val is None:
                val = 1
            elif val == np.inf:
                val = RATIO_VMAX
            elif val == -np.inf:
                val = RATIO_VMIN
            heatmap[j, i] = val
    print(heatmap)
    return heatmap

def get_heatmap_figsize(y_len, x_len):
    return (x_len * 0.6 + 2.5, y_len * 0.6 + 2)

def draw_heatmap(heatmap, ax, cmap, vmin, vmax, xticks, yticks, xlabel, ylabel):
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(len(yticks)):
        ax.axhline(i - 0.5, color="grey", linewidth=0.5)
    for i in range(len(xticks)):
        ax.axvline(i - 0.5, color="grey", linewidth=0.5)
    # if heatmap <= 8 cells, add text to each cell
    # if heatmap.size <= 8:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                # choose color between white and black depending on the background color
                c = cmap(heatmap[i, j])
                luminance = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                if luminance > 0.5:
                    text_color = "white"
                else:
                    text_color = "black"
                ax.text(j, i, f"{round(heatmap[i, j], 2):.2f}x", ha="center", va="center", color=text_color, fontweight="normal")
    
    ax.set_xticks(range(len(xticks)))
    max_tick_label = max(xticks, key=len)
    rotation = 0 if (len(max_tick_label) < 5 or "\n" in max_tick_label) else 30
    ax.set_xticklabels(xticks, rotation=rotation, ha="right" if rotation else "center")
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

def add_colorbar(fig, im, label, orientation=None, location=None, ax=None):
    if orientation is None:
        orientation = "vertical" if len(fig.axes) > 1 else "horizontal"
    if location is None:
        location = "right" if orientation == "vertical" else "top"
    if ax is None:
        ax = fig.axes
    bar = fig.colorbar(im, ax=ax, label=label, orientation=orientation, location=location)
    ticks = bar.get_ticks()
    if 1 not in ticks:
        ticks = np.concatenate(([1], ticks))
        ticks.sort()
    last_tick = ticks[-1]
    def format_tick(x, _):
        s = f"{x * 100:.0f}%"
        if x == last_tick and x != 1:
            s += "+"
        return s
    bar.formatter = mticker.FuncFormatter(format_tick)
    bar.set_ticks(ticks)
    bar_line = bar.ax.axhline(y=1, color="black", linestyle="--", linewidth=1)
    return bar, bar_line

def draw_bars(ax, X, Y, ylabel):
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
# Y: v.s. traditional indexes, v.s. materialized join view
# Z: Speed up in maintenance MergedIndex:BaseTables (bars) 
# def maintenance():
#     X = [METHODS[0], METHODS[2]]
#     heatmap = compute_heatmap(
#         X, STORAGES,
#         lambda method, storage: safe_loc(leanstore_tx if "lsm" not in storage else rocksdb_tx, get_storage_indexing_values(storage, METHODS[1], "write"), "tput") / safe_loc(leanstore_tx if "lsm" not in storage else rocksdb_tx, get_storage_indexing_values(storage, method, "write"), "tput")
#     )
#     fig, ax = plt.subplots(figsize=get_heatmap_figsize(len(X), len(STORAGES)), constrained_layout=True)
#     im = draw_heatmap(heatmap, ax, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, [s.replace(" ", "\n").replace("-", "-\n", 1) for s in STORAGES], [x.capitalize().replace(" ", "\n") for x in X], None, "Merged Index vs.")
#     # reset rotation of x labels
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
#     add_colorbar(fig, im, "Ratio of Update Throughput\nMerged Index : Either Traditional Option")
#     fig.savefig("charts/maintenance.png", dpi=300)
    
# def maintenance_columns():
#     X = [METHODS[0], METHODS[2]]
#     Y = ["none", "selected", "all"]
#     heatmap = compute_heatmap(
#         X, [0, 2, 1],
#         lambda method, column: safe_loc(leanstore_tx, get_col_sel_indexing_values(column, DEFAULT_SELECTIVITY, METHODS[1], "write"), "tput") / safe_loc(leanstore_tx, get_col_sel_indexing_values(column, DEFAULT_SELECTIVITY, method, "write"), "tput")
#     )
#     fig, ax = plt.subplots(figsize=get_heatmap_figsize(len(X), len(Y)), constrained_layout=True)
#     im = draw_heatmap(heatmap, ax, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, Y, [x.capitalize().replace(" ", "\n") for x in X], "Included Columns", "Merged Index vs.")
#     add_colorbar(fig, im, "Ratio of Update Throughput\nMerged Index : Either Traditional Option")
#     fig.savefig("charts/maintenance_columns.png", dpi=300)

def maintenance():
    X = [METHODS[0], METHODS[2]]
    Y1 = STORAGES
    Y2 = ["none", "selected", "all"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), layout="constrained", sharey=True)
    heatmap1 = compute_heatmap(
        X, STORAGES,
        lambda method, storage: safe_loc(leanstore_tx if "lsm" not in storage else rocksdb_tx, get_storage_indexing_values(storage, METHODS[1], "write"), "tput") / safe_loc(leanstore_tx if "lsm" not in storage else rocksdb_tx, get_storage_indexing_values(storage, method, "write"), "tput")
    )
    heatmap2 = compute_heatmap(
        X, [0, 2, 1],
        lambda method, column: safe_loc(leanstore_tx, get_col_sel_indexing_values(column, DEFAULT_SELECTIVITY, METHODS[1], "write"), "tput") / safe_loc(leanstore_tx, get_col_sel_indexing_values(column, DEFAULT_SELECTIVITY, method, "write"), "tput")
    )
    im1 = draw_heatmap(heatmap1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, [s.replace(" ", "\n").replace("-", "-\n", 1) for s in STORAGES], [x.capitalize().replace(" ", "\n") for x in X], None, "Merged Index vs.")
    im2 = draw_heatmap(heatmap2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, ["none", "selected", "all"], [x.capitalize().replace(" ", "\n") for x in X], "Included Columns", None)
    b, _ = add_colorbar(fig, im1, "Ratio of Update Throughput\nMerged Index : Either Traditional Option")
    # bar label font size smaller
    b.ax.yaxis.label.set_fontsize(8)
    fig.savefig("charts/maintenance.png", dpi=300)

def query_heatmaps(baseline):
    row_values = [5, 19, 50, 100, "outer"]
    col_values = [0, 2, 1]
    heatmap1 = compute_heatmap(
        row_values, col_values,
        lambda sel, col: safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, METHODS[1], "read-locality"), "tput") /
        safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, baseline, "read-locality"), "tput")
    )
    heatmap2 = compute_heatmap(
        row_values, col_values,
        lambda sel, col: safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, METHODS[1], "scan"), "tput") /
        safe_loc(leanstore_tx, get_col_sel_indexing_values(col, sel, baseline, "scan"), "tput")
    )
    tx_values = ["read-locality", "scan"]
    heatmap3 = compute_heatmap(
        STORAGES, tx_values,
        lambda storage, tx: safe_loc(leanstore_tx, get_storage_indexing_values(storage, METHODS[1], tx), "tput") /
        safe_loc(leanstore_tx, get_storage_indexing_values(storage, baseline, tx), "tput")
    )
    return heatmap1, heatmap2, heatmap3

def draw_query_heatmap(baseline):
    heatmap1, heatmap2, heatmap3 = query_heatmaps(baseline)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
    xticks = ["none", "selected", "all"]
    yticks = ["Inner Join,\nSO=5%", "Inner Join,\nSO=19%", "Inner Join,\nSO=50%", "Inner Join,\nSO=100%", "Outer Join"]

    im1 = draw_heatmap(heatmap1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, xticks, yticks, "Included Columns", "Join Type and Join Selectivity")
    im2 = draw_heatmap(heatmap2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, xticks, yticks, "Included Columns", None)
    tx_xticks = ["Point Lookup", "Range Scan"]
    storage_yticks = ["Memory-resident\nb-tree", "Disk-resident\nb-tree", "LSM"]
    im3 = draw_heatmap(heatmap3, ax3, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, tx_xticks, storage_yticks, None, None)

    ax1.set_title("Point Lookup")
    ax2.set_title("Range Scan")
    add_colorbar(fig, im3, f"Ratio of Query Throughput\nMerged Index : {baseline.capitalize()}")
    fig.savefig(f"charts/query_{'speedup' if baseline == 'traditional indexes' else 'match'}.png", dpi=300)

def indexing_size_by_storage_col(storage, col, method):
    return (method, storage, DEFAULT_TARGET, DEFAULT_SELECTIVITY, col, DEFAULT_JOIN)

def space():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    im1 = space_overhead(ax1)
    im2 = compression_effect(ax2)
    add_colorbar(fig, im1, "Size Ratio\nMerged Index : Either Traditional Option")
    fig.savefig("charts/space.png", dpi=300)

# Fig 3.1 Space overhead heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: b-tree, lsm-forest 
# - Z: Space overhead MergedIndex:BaseTables
def space_overhead(ax):
    row_values = ["b-tree", "lsm-forest"]
    col_values = [0, 2, 1]
    heatmap = compute_heatmap(
        row_values, col_values,
        lambda storage, col: 
            safe_loc(size_df, indexing_size_by_storage_col(storage, col, METHODS[1]), "core_size") /
            safe_loc(size_df, indexing_size_by_storage_col(storage, col, METHODS[0]), "core_size")
    )
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX, ["none", "selected", "all"], ["b-tree", "lsm-forest"], "Included Columns", None)
    ax.set_title("Merged Index vs.\nTraditional Indexes")
    return im

def indexing_size_by_sel_col(sel, col, method):
    join_type = "outer" if sel == "outer" else DEFAULT_JOIN
    sel = DEFAULT_SELECTIVITY if sel == "outer" else sel
    return (method, "b-tree", DEFAULT_TARGET, sel, col, join_type)

# Fig 3.2: Compression effect heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, Outer Join
# - Z: Compression effect MergedIndex:MaterializedJoin
def compression_effect(ax):
    row_values = [5, 19, 50, 100, "outer"]
    col_values = [0, 2, 1]
    heatmap = compute_heatmap(
        row_values, col_values,
        lambda sel, col: 
            safe_loc(size_df, indexing_size_by_sel_col(sel, col, METHODS[1]), "core_size") / 
            safe_loc(size_df, indexing_size_by_sel_col(sel, col, METHODS[2]), "core_size")
    )
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX, ["none", "selected", "all"], ["Inner Join,\nSO=5%", "Inner Join,\nSO=19%", "Inner Join,\nSO=50%", "Inner Join,\nSO=100%", "Outer Join"], None, "Join Type and Join Selectivity")
    ax.set_title("Merged Index vs.\nMaterialized Join View")
    return im

# Fig 3.3: Size overview
# - X: Traditional Indexes, Merged Index, Materialized Join View
# - Y: Core, Rest, Additional, stacked bar    
def size_overview(storage):
    X = [m.capitalize() for m in METHODS]
    get_index_values = lambda method: (method, storage, DEFAULT_TARGET, DEFAULT_SELECTIVITY, DEFAULT_COLUMNS, DEFAULT_JOIN)
    
    Y1 = [
        safe_loc(size_df, get_index_values(METHODS[0]), "rest_size"),
        safe_loc(size_df, get_index_values(METHODS[1]), "rest_size"),
        safe_loc(size_df, get_index_values(METHODS[2]), "rest_size")
    ]
    
    Y2 = [
        safe_loc(size_df, get_index_values(METHODS[0]), "additional_size"),
        safe_loc(size_df, get_index_values(METHODS[1]), "additional_size"),
        safe_loc(size_df, get_index_values(METHODS[2]), "additional_size")
    ]
    
    Y3 = [
        safe_loc(size_df, get_index_values(METHODS[0]), "core_size"),
        safe_loc(size_df, get_index_values(METHODS[1]), "core_size"),
        safe_loc(size_df, get_index_values(METHODS[2]), "core_size")
    ]
    
    fig, ax = plt.subplots(figsize=(len(X) * 0.6 + 2, 4), layout="constrained")
    ax.bar(X, Y1, color=COLORS["yellow"], label="Rest of the tables", hatch="////")
    ax.bar(X, Y2, color=COLORS["pink"], label="Additional storage", bottom=Y1, hatch="\\\\\\\\")
    ax.bar(X, Y3, color=COLORS["blue"], label="Core view or\nstorage structure(s)", bottom=[Y1[i] + Y2[i] for i in range(len(X))])
    ax.set_ylabel("Size (GiB)")
    ax.set_xticks(range(len(X)))
    ax.set_xticklabels([x.replace(" ", "\n") for x in X])
    ax.legend(loc="upper left")
    fig.savefig(f"charts/size_{storage}.png", dpi=300)

# X: three TX
# Y: v.s. traditional indexes, v.s. materialized join view
# Z: Speed up in TX by MergedIndex (heatmap)
def common_case_heatmap(storage):
    X = [METHODS[0], METHODS[2]]
    Y = TX_TYPES
    heatmap = compute_heatmap(
        X, TX_TYPES,
        lambda method, tx: safe_loc(leanstore_tx, get_storage_indexing_values(storage, METHODS[1], tx), "tput") / safe_loc(leanstore_tx, get_storage_indexing_values(storage, method, tx), "tput")
    )
    return heatmap
    
def common_case():
    h1 = common_case_heatmap("b-tree")
    h2 = common_case_heatmap("lsm-forest")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    X = ['Maintenance\nagainst\nupdates', 'Join query:\npoint lookup', 'Join query:\nrange scan']
    Y = [METHODS[0], METHODS[2]]
    im1 = draw_heatmap(h1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, X, [y.capitalize().replace(" ", "\n") for y in Y], None, "Merged Index vs.")
    im2 = draw_heatmap(h2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, X, [y.capitalize().replace(" ", "\n") for y in Y], None, None)
    ax1.set_title("b-trees")
    ax2.set_title("lsm-forests")
    add_colorbar(fig, im1, "Ratio of Throughput\nMerged Index : Either Traditional Option")
    fig.savefig("charts/common_case.png", dpi=300)

# Execution
if __name__ == "__main__":
    common_case()
    maintenance()
    # maintenance_columns()
    draw_query_heatmap(METHODS[0])
    draw_query_heatmap(METHODS[2])
    space()
    size_overview("b-tree")
    size_overview("lsm-forest")