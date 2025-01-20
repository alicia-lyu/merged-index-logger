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

COLUMN_LABELS = ["keys", "covering", "all"]

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
            v = result
        else:
            v = result.iloc[-1]
        if np.isnan(v):
            print(f"NaN value for indexing with {indexing_values} on column {column}")
        return v
    except KeyError as e:
        print(f"KeyError for indexing with {indexing_values} on column {column}: {e}")
        return np.nan
    
def index_df_storage(storage, method, tx, col="tput"):
    return safe_loc(leanstore_tx if "lsm" not in storage else rocksdb_tx, get_storage_indexing_values(storage, method, tx), col)

def compare_storage(storage, method, tx, col="tput"):
    numerator = index_df_storage(storage, METHODS[1], tx, col)
    denominator = index_df_storage(storage, method, tx, col)
    if denominator == 0:
        return np.inf
    elif np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    else:
        return numerator / denominator
    
def index_df_col_sel(included_columns, selectivity, method, tx, col="tput"):
    return safe_loc(leanstore_tx, get_col_sel_indexing_values(included_columns, selectivity, method, tx), col)

def compare_col_sel(included_columns, selectivity, method, tx, col="tput"):
    numerator = index_df_col_sel(included_columns, selectivity, METHODS[1], tx, col)
    denominator = index_df_col_sel(included_columns, selectivity, method, tx, col)
    if denominator == 0:
        return np.inf
    elif np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    else:
        return numerator / denominator

def compute_heatmap(row_values, col_values, value_fn):
    heatmap = np.ones((len(row_values), len(col_values)))
    for i, col_val in enumerate(col_values):
        for j, row_val in enumerate(row_values):
            val = value_fn(row_val, col_val)
            heatmap[j, i] = val
    return heatmap

def get_heatmap_figsize(y_len, x_len):
    return (x_len * 0.6 + 2.5, y_len * 0.6 + 2)

def get_text_func(vmax):
    if vmax > 1:
        text_func = lambda t: f"{round(t, 2):.2f}x"
    else:
        text_func = lambda t: f"{round(t * 100, 2):.2f}%"
    return text_func

def draw_heatmap(heatmap, ax, cmap, vmin, vmax, xticks, yticks, xlabel, ylabel):
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(len(yticks)):
        ax.axhline(i - 0.5, color="grey", linewidth=0.5)
    for i in range(len(xticks)):
        ax.axvline(i - 0.5, color="grey", linewidth=0.5)
    # if heatmap <= 8 cells, add text to each cell
    text_func = get_text_func(vmax)
    if heatmap.size <= 9:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                v = heatmap[i, j]
                if np.isnan(v):
                    print(f"NaN value for cell ({i}, {j})")
                    continue
                # choose color between white and black depending on the background color
                c = cmap(v)
                luminance = 0.6 * c[0] + 0.2 * c[1] + 0.3 * c[2]
                if luminance < 0.5:
                    text_color = "white"
                else:
                    text_color = "black"
                ax.text(j, i, text_func(heatmap[i, j]), ha="center", va="center", color=text_color, fontweight="normal")
    
    ax.set_xticks(range(len(xticks)))
    max_tick_label = max(xticks, key=len)
    # ax_length x axis
    x_inches = ax.get_position().x1 - ax.get_position().x0
    rotation = 0 if (len(max_tick_label) < x_inches * 40 or "\n" in max_tick_label) else 30
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
    if orientation == "vertical":
        # label as fig title
        fig.suptitle(label, fontweight="bold")
        label = None
    bar = fig.colorbar(im, ax=ax, label=label, orientation=orientation, location=location)
    ticks = bar.get_ticks()
    if 1 not in ticks:
        ticks = np.concatenate(([1], ticks))
        ticks.sort()
    last_tick = ticks[-1]
    text_func = get_text_func(last_tick)
    def format_tick(x, _):
        s = text_func(x)
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
    ax.set_ylabel(ylabel)

def maintenance():
    X = [METHODS[0], METHODS[2]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), layout="constrained", sharey=True)
    heatmap1 = compute_heatmap(
        X, STORAGES,
        lambda method, storage: compare_storage(storage, method, "write")
    )
    heatmap2 = compute_heatmap(
        X, [0, 2, 1],
        lambda method, column: compare_col_sel(column, DEFAULT_SELECTIVITY, method, "write")
    )
    im1 = draw_heatmap(heatmap1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, [s.replace(" ", "\n").replace("-", "-\n", 1) for s in STORAGES], [x.capitalize().replace(" ", "\n") for x in X], None, "Merged Index vs.")
    im2 = draw_heatmap(heatmap2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, COLUMN_LABELS, [x.capitalize().replace(" ", "\n") for x in X], "Included Columns", None)
    ax1.set_aspect(0.8)
    ax2.set_aspect(0.8)
    b, _ = add_colorbar(fig, im1, "Ratio of Update Throughput\nMerged Index : Either Traditional Option")
    fig.savefig("charts/maintenance.png", dpi=300)

def query_heatmaps(baseline):
    row_values = [5, 19, 50, 100, "outer"]
    col_values = [0, 2, 1]
    heatmap1 = compute_heatmap(
        row_values, col_values,
        lambda sel, col: compare_col_sel(col, sel, baseline, "read-locality")
    )
    heatmap2 = compute_heatmap(
        row_values, col_values,
        lambda sel, col: compare_col_sel(col, sel, baseline, "scan")
    )
    tx_values = ["read-locality", "scan"]
    heatmap3 = compute_heatmap(
        STORAGES, tx_values,
        lambda storage, tx: compare_storage(storage, baseline, tx)
    )
    return heatmap1, heatmap2, heatmap3

def draw_query_heatmap(baseline):
    heatmap1, heatmap2, heatmap3 = query_heatmaps(baseline)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
    xticks = COLUMN_LABELS
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

def compare_size_by_storage_col(storage, col, method, size_col="core_size"):
    numerator = safe_loc(size_df, indexing_size_by_storage_col(storage, col, METHODS[1]), size_col)
    denominator = safe_loc(size_df, indexing_size_by_storage_col(storage, col, method), size_col)
    if denominator == 0:
        return np.inf
    elif np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    else:
        return numerator / denominator

def space():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    im1 = space_overhead(ax1)
    im2 = compression_effect(ax2)
    ax2.set_aspect(0.5)
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
            compare_size_by_storage_col(storage, col, METHODS[0], "core_size")
    )
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX, COLUMN_LABELS, ["b-tree", "lsm-forest"], "Included Columns", None)
    ax.set_title("Merged Index vs.\nTraditional Indexes")
    return im

def indexing_size_by_sel_col(sel, col, method, size_col="core_size"):
    join_type = "outer" if sel == "outer" else DEFAULT_JOIN
    sel = DEFAULT_SELECTIVITY if sel == "outer" else sel
    index_columns = (method, "b-tree", DEFAULT_TARGET, sel, col, join_type)
    return safe_loc(size_df, index_columns, size_col)

def compare_size_by_sel_col(sel, col, method, size_col="core_size"):
    numerator = indexing_size_by_sel_col(sel, col, METHODS[1], size_col)
    denominator = indexing_size_by_sel_col(sel, col, method, size_col)
    if denominator == 0:
        return np.inf
    elif np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    else:
        return numerator / denominator

# Fig 3.2: Compression effect heat map blue-orange
# - X: included columns (none, selected, all)
# - Y: InnerJoin5, InnerJoin19, InnerJoin50, InnerJoin100, Outer Join
# - Z: Compression effect MergedIndex:MaterializedJoin
def compression_effect(ax):
    row_values = [5, 19, 50, 100, "outer"]
    col_values = [0, 2, 1]
    heatmap = compute_heatmap(
        row_values, col_values,
        lambda sel, col: compare_size_by_sel_col(sel, col, METHODS[2], "core_size")
    )
    im = draw_heatmap(heatmap, ax, SIZE_CMAP, RATIO_VMIN, RATIO_VMAX, COLUMN_LABELS, ["Inner Join,\nSO=5%", "Inner Join,\nSO=19%", "Inner Join,\nSO=50%", "Inner Join,\nSO=100%", "Outer Join"], None, "Join Type and Join Selectivity")
    ax.set_title("Merged Index vs.\nMaterialized Join View")
    return im

# Fig 3.3: Size overview
# - X: Traditional Indexes, Merged Index, Materialized Join View
# - Y: Core, Rest, Additional, stacked bar    
def size_overview(storage):
    X = [m.capitalize() for m in METHODS]
    get_index_values = lambda method: (method, storage, DEFAULT_TARGET, DEFAULT_SELECTIVITY, 2, DEFAULT_JOIN) # use selected columns
    
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
    
    fig, ax = plt.subplots(figsize=(len(X) * 0.6 + 3, 3), layout="constrained")
    ax.bar(X, Y1, color=COLORS["yellow"], label="Rest of the tables", hatch="////")
    ax.bar(X, Y2, color=COLORS["pink"], label="Additional storage", bottom=Y1, hatch="\\\\\\\\")
    ax.bar(X, Y3, color=COLORS["blue"], label="Core storage structure(s)", bottom=[Y1[i] + Y2[i] for i in range(len(X))])
    ax.set_ylabel("Size (GiB)")
    ax.set_xticks(range(len(X)))
    ax.set_xticklabels([x.replace(" ", "\n") for x in X])
    ax.legend(loc="upper left")
    fig.savefig(f"charts/size_{storage}.png", dpi=300)

def core_time():
    X = [m.capitalize() for m in METHODS]
    get_index_values = lambda method: (method, "lsm-forest", DEFAULT_TARGET, DEFAULT_SELECTIVITY, 2, DEFAULT_JOIN, DEFAULT_DRAM)
    Y1 = [
        safe_loc(size_df, get_index_values(METHODS[0]), "rest_time"),
        safe_loc(size_df, get_index_values(METHODS[1]), "rest_time"),
        safe_loc(size_df, get_index_values(METHODS[2]), "rest_time")
    ]
    Y1 = [ y / 1000 for y in Y1 ]
    Y2 = [
        safe_loc(size_df, get_index_values(METHODS[0]), "additional_time"),
        safe_loc(size_df, get_index_values(METHODS[1]), "additional_time"),
        safe_loc(size_df, get_index_values(METHODS[2]), "additional_time")
    ]
    Y2 = [ y / 1000 for y in Y2 ]
    Y3 = [
        safe_loc(size_df, get_index_values(METHODS[0]), "core_time"),
        safe_loc(size_df, get_index_values(METHODS[1]), "core_time"),
        safe_loc(size_df, get_index_values(METHODS[2]), "core_time")
    ]
    Y3 = [ y / 1000 for y in Y3 ]
    fig, ax = plt.subplots(figsize=(len(X) * 0.6 + 3, 3), layout="constrained")
    ax.bar(X, Y1, color=COLORS["yellow"], label="Rest of the tables", hatch="////")
    ax.bar(X, Y2, color=COLORS["pink"], label="Additional storage", bottom=Y1, hatch="\\\\\\\\")
    ax.bar(X, Y3, color=COLORS["blue"], label="Core storage structure(s)", bottom=[Y1[i] + Y2[i] for i in range(len(X))])
    ax.set_ylabel("Time (s)")
    ax.set_xticks(range(len(X)), [x.replace(" ", "\n") for x in X])
    ax.legend(loc="upper left")
    fig.savefig("charts/time.png", dpi=300)
    
# X: three TX
# Y: v.s. traditional indexes, v.s. materialized join view
# Z: Speed up in TX by MergedIndex (heatmap)
def common_case_heatmap(storage):
    X = [METHODS[0], METHODS[2]]
    Y = TX_TYPES
    heatmap = compute_heatmap(
        X, TX_TYPES,
        lambda method, tx: compare_storage(storage, method, tx)
    )
    return heatmap
    
def common_case():
    h1 = common_case_heatmap("b-tree")
    h2 = common_case_heatmap("lsm-forest")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), layout="constrained", sharey=True)
    X = ['Maintenance\nagainst\nupdates', 'Join query:\npoint lookup', 'Join query:\nrange scan']
    Y = [METHODS[0], METHODS[2]]
    im1 = draw_heatmap(h1, ax1, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, X, [y.capitalize().replace(" ", "\n") for y in Y], None, "Merged Index vs.")
    im2 = draw_heatmap(h2, ax2, QUERY_CMAP, RATIO_VMIN, RATIO_VMAX, X, [y.capitalize().replace(" ", "\n") for y in Y], None, None)
    ax1.set_title("b-trees")
    ax2.set_title("lsm-forests")
    ax1.set_aspect(0.8)
    ax2.set_aspect(0.8)
    add_colorbar(fig, im1, "Ratio of Throughput\nMerged Index : Either Traditional Option")
    fig.savefig("charts/common_case.png", dpi=300)

def cpu_utilization():
    X = STORAGES
    Y = METHODS
    heatmap_read = compute_heatmap(
        Y, X,
        lambda method, storage: index_df_storage(storage, method, "read-locality", "utilized_cpus") / 4
    )
    heatmap_scan = compute_heatmap(
        Y, X,
        lambda method, storage: index_df_storage(storage, method, "scan", "utilized_cpus") / 4
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.8), layout="constrained", sharey=True)
    # object version of OrRd colormap
    cmap = plt.get_cmap("OrRd")
    im1 = draw_heatmap(heatmap_read, ax1, cmap, 0, 1, [x.replace(" ", "\n") for x in X], [y.capitalize().replace(" ", "\n") for y in Y], None, None)
    im2 = draw_heatmap(heatmap_scan, ax2, cmap, 0, 1, [x.replace(" ", "\n") for x in X], [y.capitalize().replace(" ", "\n") for y in Y], None, None)
    add_colorbar(fig, im1, "CPU Utilization\nof Join Queries")
    # change aspect ratio of heatmap
    ax1.set_aspect(0.43)
    ax2.set_aspect(0.43)
    ax1.set_title("Point Lookup")
    ax2.set_title("Range Scan")
    fig.savefig("charts/cpu_utilization.png", dpi=300)

# Execution
if __name__ == "__main__":
    cpu_utilization()
    core_time()
    common_case()
    maintenance()
    draw_query_heatmap(METHODS[0])
    draw_query_heatmap(METHODS[2])
    space()
    size_overview("b-tree")
    size_overview("lsm-forest")