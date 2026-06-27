"""
Convert Modified Swiss Dwellings (MSD) dataset to HouseDiffusion NPZ format.

The MSD dataset has two structures; this script handles both:

  ── V1 (pickle-based, well documented) ─────────────────────────────────────
  modified-swiss-dwellings-v1-train/
      graph_out/   <id>.pickle   NetworkX graph, nodes have:
                                   roomtype  : str  e.g. 'Bedroom'
                                   geometry  : shapely.Polygon
                                   centroid  : tuple (x, y)
                               edges have:
                                   connectivity: 'door'|'entrance'|'passage'
      full_out/    <id>.npy     segmentation image [512,512,3]

  ── V2 (CSV-based) ──────────────────────────────────────────────────────────
  mds_V2_5.372k.csv             Pandas dataframe, one row per room, columns:
                                   fp_id, room_id, room_type, geometry (WKT),
                                   centroid_x, centroid_y, ...

Usage:
    # V1 pickle format
    python msd_to_housediffusion.py \\
        --mode pickle \\
        --graph_dir path/to/graph_out \\
        --target_set 8 \\
        --out_dir processed_rplan

    # V2 CSV format
    python msd_to_housediffusion.py \\
        --mode csv \\
        --csv_path path/to/mds_V2_5.372k.csv \\
        --target_set 8 \\
        --out_dir processed_rplan
"""

import os
import pickle
import argparse
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from tqdm import tqdm

# ── Room type mapping: MSD → RPLAN ID ─────────────────────────────────────
# RPLAN: 1=living, 2=master bed, 3=kitchen, 4=bathroom, 5=dining,
#        6=child room, 7=study, 8=second room, 10=guest, 11=balcony,
#        12=entrance/corridor, 13=storage
MSD_TO_RPLAN = {
    'Livingroom':   1,
    'Bedroom':      2,   # first bedroom → master (2), extras → 6, 8, 7
    'Kitchen':      3,
    'Bathroom':     4,
    'Dining':       5,
    'Corridor':     12,
    'Balcony':      11,
    'Storeroom':    13,
    'Stairs':       10,
    'Structure':    None,  # skip structural elements
    'Door':         None,
    'Entrance Door':None,
    'Window':       None,
}

# For multiple bedrooms in the same floor plan use different IDs
BEDROOM_IDS = [2, 6, 8, 7]  # master, child, second, study

MAX_POINTS  = 100
MAX_CORNERS = 12   # max polygon corners per room (simplify if more)
PAD_GRAPH   = 200  # fixed graph length

def one_hot(x, z):
    v = np.zeros(z)
    if 0 <= x < z:
        v[x] = 1.0
    return v


def simplify_polygon(coords, max_corners):
    """Keep at most max_corners from a polygon's exterior coordinates."""
    if len(coords) <= max_corners:
        return np.array(coords)
    # Evenly subsample
    idx = np.round(np.linspace(0, len(coords) - 1, max_corners)).astype(int)
    return np.array(coords)[idx]


def normalize_coords(all_points):
    """
    Normalize all polygon points to [-1, 1] based on the bounding box
    of the entire floor plan.
    """
    pts = np.vstack(all_points)
    mn, mx = pts.min(0), pts.max(0)
    span = mx - mn
    span[span == 0] = 1.0   # avoid division by zero
    # center → [0,1] → [-1,1]
    normalized = []
    for p in all_points:
        p = np.array(p)
        p = (p - mn) / span       # [0, 1]
        p = p * 2 - 1             # [-1, 1]
        normalized.append(p)
    return normalized


def build_house(rooms, edges):
    """
    Convert a list of (room_type_id, polygon_corners) and edge list
    into the 100×94 house_layouts array + masks.

    rooms : [(rtype_id, np.array[nc, 2]), ...]
    edges : [(i, j), ...]   room indices connected by a door/passage
    """
    house_rows = []
    corner_bounds = []
    num_points = 0

    for room_idx, (rtype_id, corners) in enumerate(rooms):
        nc = len(corners)
        rtype_vec  = np.tile(one_hot(rtype_id, 25), (nc, 1))
        room_vec   = np.tile(one_hot(room_idx + 1, 32), (nc, 1))
        corner_vec = np.array([one_hot(c, 32) for c in range(nc)])
        pad_col    = np.ones((nc, 1))
        conn_col   = np.array(
            [[c, (c + 1) % nc] for c in range(nc)], dtype=float
        ) + num_points

        row = np.concatenate(
            [corners, rtype_vec, corner_vec, room_vec, pad_col, conn_col],
            axis=1
        )  # (nc, 94)
        house_rows.append(row)
        corner_bounds.append((num_points, num_points + nc))
        num_points += nc

    layout = np.concatenate(house_rows, axis=0)  # (total_points, 94)
    assert layout.shape[1] == 94

    # Padding
    padding = np.zeros((MAX_POINTS - len(layout), 94))
    gen_mask = np.ones((MAX_POINTS, MAX_POINTS))
    gen_mask[:len(layout), :len(layout)] = 0
    layout = np.concatenate([layout, padding], axis=0)

    # Build padded graph
    edges_arr = np.array([[i, 1, j] for i, j in edges], dtype=float) \
        if edges else np.zeros((0, 3))
    graph = np.zeros((PAD_GRAPH, 3))
    graph[:len(edges_arr)] = edges_arr

    # Attention masks
    door_mask = np.ones((MAX_POINTS, MAX_POINTS))
    self_mask = np.ones((MAX_POINTS, MAX_POINTS))
    edge_set = set(map(tuple, edges)) | set((j, i) for i, j in edges)
    for i, (s_i, e_i) in enumerate(corner_bounds):
        for j, (s_j, e_j) in enumerate(corner_bounds):
            if i == j:
                self_mask[s_i:e_i, s_j:e_j] = 0
            elif (i, j) in edge_set:
                door_mask[s_i:e_i, s_j:e_j] = 0

    return layout, graph, door_mask, self_mask, gen_mask


def process_nx_graph(G):
    """
    Extract rooms + edges from a NetworkX graph_out pickle.
    Returns (rooms, edges) or None if the floor plan should be skipped.
    """
    rooms = []
    bedroom_count = 0

    for node_id, data in G.nodes(data=True):
        rtype_str = data.get('roomtype', '')
        rtype_id  = MSD_TO_RPLAN.get(rtype_str)
        if rtype_id is None:
            continue  # skip structural, doors, windows

        geom = data.get('geometry')
        if geom is None or not isinstance(geom, Polygon) or geom.is_empty:
            continue

        # For multiple bedrooms, cycle through different RPLAN bedroom IDs
        if rtype_str == 'Bedroom':
            rtype_id = BEDROOM_IDS[min(bedroom_count, len(BEDROOM_IDS) - 1)]
            bedroom_count += 1

        # Extract exterior ring (drop closing duplicate point)
        coords = list(geom.exterior.coords)[:-1]
        if len(coords) < 3:
            continue

        corners = simplify_polygon(coords, MAX_CORNERS)
        rooms.append((node_id, rtype_id, corners))

    if len(rooms) < 2:
        return None

    # Re-index rooms 0..N and remap edges
    node_id_to_idx = {nid: i for i, (nid, _, _) in enumerate(rooms)}
    rooms_out = [(rtype_id, corners) for _, rtype_id, corners in rooms]

    edges = []
    for u, v, edata in G.edges(data=True):
        if u in node_id_to_idx and v in node_id_to_idx:
            edges.append((node_id_to_idx[u], node_id_to_idx[v]))

    # Normalize coordinates
    all_pts = [c for _, c in rooms_out]
    all_pts_norm = normalize_coords(all_pts)
    rooms_out = [(rt, p) for (rt, _), p in zip(rooms_out, all_pts_norm)]

    # Check total corner count fits in MAX_POINTS
    total = sum(len(c) for _, c in rooms_out)
    if total >= MAX_POINTS:
        return None

    return rooms_out, edges, len(rooms_out)


# ── Pickle-based converter ─────────────────────────────────────────────────

def convert_pickle(graph_dir, target_set, out_dir, train_ratio=0.85):
    files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pickle')])
    print(f"Found {len(files)} pickle files in {graph_dir}")

    all_data = []
    for fname in tqdm(files, desc="Parsing pickle files"):
        fpath = os.path.join(graph_dir, fname)
        with open(fpath, 'rb') as f:
            G = pickle.load(f)
        result = process_nx_graph(G)
        if result is None:
            continue
        rooms, edges, n_rooms = result
        all_data.append((rooms, edges, n_rooms))

    print(f"Valid floor plans: {len(all_data)}")
    _save_splits(all_data, target_set, out_dir, train_ratio)


# ── CSV-based converter ───────────────────────────────────────────────────

def convert_csv(csv_path, target_set, out_dir, train_ratio=0.85):
    import pandas as pd
    from shapely import wkt as shapely_wkt

    df = pd.read_csv(csv_path)
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Auto-detect column names (common variations)
    # Prefer unit_id (individual apartment) over floor_id (whole building floor)
    # Iterate the priority list, not the dataframe columns
    fp_col = next((c for c in ['unit_id', 'fp_id', 'floorplan_id', 'id', 'ID', 'floor_id']
                   if c in df.columns), None)
    type_col = next((c for c in ['roomtype', 'room_type', 'type', 'subtype', 'label']
                     if c in df.columns), None)
    geom_col = next((c for c in ['geom', 'geometry', 'polygon', 'shape']
                     if c in df.columns), None)

    if fp_col is None or type_col is None or geom_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Found: {list(df.columns)}\n"
            "Please set fp_col, type_col, geom_col manually in the script."
        )

    print(f"Using columns: fp_id='{fp_col}', type='{type_col}', geom='{geom_col}'")

    all_data = []
    for fp_id, group in tqdm(df.groupby(fp_col), desc="Parsing floor plans"):
        bedroom_count = 0
        rooms = []

        for _, row in group.iterrows():
            rtype_str = str(row[type_col])
            rtype_id  = MSD_TO_RPLAN.get(rtype_str)
            if rtype_id is None:
                continue

            geom_raw = row[geom_col]
            try:
                if isinstance(geom_raw, str):
                    geom = shapely_wkt.loads(geom_raw)
                else:
                    geom = geom_raw
            except Exception:
                continue

            if not isinstance(geom, Polygon) or geom.is_empty:
                continue

            if rtype_str == 'Bedroom':
                rtype_id = BEDROOM_IDS[min(bedroom_count, len(BEDROOM_IDS) - 1)]
                bedroom_count += 1

            coords = list(geom.exterior.coords)[:-1]
            if len(coords) < 3:
                continue

            corners = simplify_polygon(coords, MAX_CORNERS)
            rooms.append((rtype_id, corners))

        if len(rooms) < 2:
            continue

        # Normalize
        all_pts = [c for _, c in rooms]
        all_pts_norm = normalize_coords(all_pts)
        rooms_norm = [(rt, p) for (rt, _), p in zip(rooms, all_pts_norm)]

        total = sum(len(c) for _, c in rooms_norm)
        if total >= MAX_POINTS:
            continue

        # Build simple connectivity: first room (living/corridor) → all others
        edges = [(0, j) for j in range(1, len(rooms_norm))]
        all_data.append((rooms_norm, edges, len(rooms_norm)))

    print(f"Valid floor plans: {len(all_data)}")
    _save_splits(all_data, target_set, out_dir, train_ratio)


# ── Shared saving logic ────────────────────────────────────────────────────

def _save_splits(all_data, target_set, out_dir, train_ratio):
    os.makedirs(out_dir, exist_ok=True)

    target_fps = [d for d in all_data if d[2] == target_set]
    other_fps  = [d for d in all_data if d[2] != target_set]
    print(f"  Floor plans with {target_set} rooms (eval): {len(target_fps)}")
    print(f"  Other floor plans (train):                  {len(other_fps)}")

    if len(target_fps) == 0:
        print(f"WARNING: no floor plans found with exactly {target_set} rooms.")
        print(f"Available room counts: "
              f"{sorted(set(d[2] for d in all_data))}")
        return

    # Build cnumber_dist from OTHER floor plans (used during eval sampling)
    cnumber_dist = defaultdict(list)
    for rooms, _, _ in other_fps:
        for rtype_id, corners in rooms:
            cnumber_dist[rtype_id].append(len(corners))
    # Fallback: ensure each known type has at least one entry
    for rt in MSD_TO_RPLAN.values():
        if rt and rt not in cnumber_dist:
            cnumber_dist[rt] = [4]

    # ── Train split ─────────────────────────────────────────────────────────
    train_houses, train_graphs, train_dmasks, train_smasks, train_gmasks = \
        [], [], [], [], []
    for rooms, edges, _ in tqdm(other_fps, desc="Building train split"):
        layout, graph, dm, sm, gm = build_house(rooms, edges)
        train_houses.append(layout)
        train_graphs.append(graph)
        train_dmasks.append(dm)
        train_smasks.append(sm)
        train_gmasks.append(gm)

    np.savez_compressed(
        f"{out_dir}/rplan_train_{target_set}",
        graphs=np.array(train_graphs),
        houses=np.array(train_houses),
        door_masks=np.array(train_dmasks),
        self_masks=np.array(train_smasks),
        gen_masks=np.array(train_gmasks),
    )
    np.savez_compressed(
        f"{out_dir}/rplan_train_{target_set}_cndist",
        cnumber_dist=cnumber_dist,
    )
    print(f"Saved train: {len(train_houses)} samples")

    # ── Eval split ──────────────────────────────────────────────────────────
    eval_houses, eval_graphs, eval_dmasks, eval_smasks, eval_gmasks = \
        [], [], [], [], []
    syn_houses, syn_graphs, syn_dmasks, syn_smasks, syn_gmasks = \
        [], [], [], [], []

    for rooms, edges, _ in tqdm(target_fps, desc="Building eval split"):
        layout_gt, graph, dm, sm, gm = build_house(rooms, edges)

        # Synthetic: zero out coordinates (model will generate them)
        rooms_zero = [(rt, np.zeros_like(c)) for rt, c in rooms]
        layout_syn, graph_syn, dm_syn, sm_syn, gm_syn = build_house(rooms_zero, edges)

        eval_houses.append(layout_gt);  eval_graphs.append(graph)
        eval_dmasks.append(dm);         eval_smasks.append(sm)
        eval_gmasks.append(gm)

        syn_houses.append(layout_syn);  syn_graphs.append(graph_syn)
        syn_dmasks.append(dm_syn);      syn_smasks.append(sm_syn)
        syn_gmasks.append(gm_syn)

    np.savez_compressed(
        f"{out_dir}/rplan_eval_{target_set}",
        graphs=np.array(eval_graphs),
        houses=np.array(eval_houses),
        door_masks=np.array(eval_dmasks),
        self_masks=np.array(eval_smasks),
        gen_masks=np.array(eval_gmasks),
    )
    np.savez_compressed(
        f"{out_dir}/rplan_eval_{target_set}_syn",
        graphs=np.array(syn_graphs),
        houses=np.array(syn_houses),
        door_masks=np.array(syn_dmasks),
        self_masks=np.array(syn_smasks),
        gen_masks=np.array(syn_gmasks),
    )
    print(f"Saved eval:  {len(eval_houses)} samples")
    print(f"\nDone → {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) // 1024
        print(f"  {f}  ({size_kb} KB)")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert MSD dataset to HouseDiffusion NPZ format')
    parser.add_argument('--mode', choices=['pickle', 'csv'], default='pickle',
                        help='Dataset format')
    parser.add_argument('--graph_dir', default='modified-swiss-dwellings-v1-train/graph_out',
                        help='[pickle mode] Path to graph_out/ folder')
    parser.add_argument('--csv_path', default='mds_V2_5.372k.csv',
                        help='[csv mode] Path to the MSD CSV file')
    parser.add_argument('--target_set', type=int, default=8,
                        help='Number of rooms for the eval set')
    parser.add_argument('--out_dir', default='processed_rplan',
                        help='Output directory for NPZ files')
    parser.add_argument('--train_ratio', type=float, default=0.85)
    args = parser.parse_args()

    if args.mode == 'pickle':
        convert_pickle(args.graph_dir, args.target_set, args.out_dir,
                       args.train_ratio)
    else:
        convert_csv(args.csv_path, args.target_set, args.out_dir,
                    args.train_ratio)
