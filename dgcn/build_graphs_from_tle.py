import math
import numpy as np
import pandas as pd
import torch

from skyfield.api import load, EarthSatellite, wgs84
from torch_geometric.data import Data

def load_tle_3line(tle_path: str):
    """TLE data standard in three-line format: name + line1 + line2.
    Return: A dictionary mapping names to EarthSatellite objects."""
    ts = load.timescale()

    sats = {}
    with open(tle_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]
        try:
            sat = EarthSatellite(l1, l2, name, ts)
            sats[name] = sat
        except Exception as e:
            print(f"[WARN] Failed parsing TLE for {name}: {e}")
        i += 3

    if len(sats) == 0:
        raise RuntimeError("No TLE satellites loaded. Check TLE file format.")

    return sats, ts


def knn_edge_index(xyz_km: np.ndarray, k: int = 6):
    """xyz_km: (N,3)
    Perform k-Nearest Neighbors (kNN) undirected graph construction using Euclidean distance as the metric.
    Generate the adjacency list in edge_index format (COO format) with dimensions (2,E)"""
    N = xyz_km.shape[0]
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # pairwise dist^2
    diff = xyz_km[:, None, :] - xyz_km[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)  # (N,N)
    np.fill_diagonal(dist2, np.inf)

    edges = []
    kk = min(k, N - 1)
    for i in range(N):
        nn = np.argpartition(dist2[i], kk)[:kk]  # k nearest
        for j in nn:
            edges.append((i, j))
            edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2,E)
    return edge_index

def normalize_node_features(elev_deg, az_deg, dist_km, rr_kmps, min_elev_deg=10.0):
    # 1) elev: 10~90 -> 0~1
    elev_n = (elev_deg - min_elev_deg) / (90.0 - min_elev_deg)
    elev_n = np.clip(elev_n, 0.0, 1.0)

    # 2) az: 0~360 -> -1~1
    az_rad = np.deg2rad(az_deg)
    az_s = np.sin(az_rad).astype(np.float32)
    az_c = np.cos(az_rad).astype(np.float32)

    # 3) dist: log
    dist_n = np.log1p(dist_km) / np.log1p(3000.0)
    dist_n = np.clip(dist_n, 0.0, 1.0)

    # 4) rr: -1~1
    rr_n = np.tanh(rr_kmps / 10.0)

    return elev_n.astype(np.float32), az_s.astype(np.float32), az_c.astype(np.float32), dist_n.astype(np.float32), rr_n.astype(np.float32)

import numpy as np
import pandas as pd

def resample_service_csv_to_1s(df: pd.DataFrame, freq: str = "1S", max_ffill_gap_s: int = 2):
    """
    Resample the service CSV onto a strict 1-second grid.
      - Create a complete 1-second timeline between min and max timestamps.
      - For missing seconds:
          * If the missing run length <= max_ffill_gap_s, forward-fill Connected_Satellite (small gap).
          * If the missing run length  > max_ffill_gap_s, keep as missing (big gap).
      - Return:
          df_1s: resampled dataframe with 'Timestamp' as a normal column
          service_valid: boolean array indicating whether serving satellite is valid (observed or small-gap filled)
          observed_mask: boolean array indicating whether this second exists in the original CSV
    """
    if "Timestamp" not in df.columns or "Connected_Satellite" not in df.columns:
        raise ValueError("df must contain columns: Timestamp, Connected_Satellite")

    # Parse and sort timestamps (UTC)
    t = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    if t.isna().any():
        bad = df[t.isna()].head(5)
        raise ValueError(f"Timestamp parse failed for some rows, examples:\n{bad}")

    df = df.copy()
    df["_ts"] = t
    df = df.sort_values("_ts").drop_duplicates(subset=["_ts"], keep="last")
    df = df.set_index("_ts")
    df = df.drop(columns=["Timestamp"], errors="ignore")

    # Build a strict 1-second timeline
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")

    # Record which seconds were observed in the original CSV
    observed_mask = full_index.isin(df.index)

    # Reindex to full 1-second grid
    df_1s = df.reindex(full_index)

    # Identify missing runs in Connected_Satellite
    is_missing = df_1s["Connected_Satellite"].isna()
    # Run id for consecutive segments
    run_id = is_missing.ne(is_missing.shift()).cumsum()
    # Run length for each row
    run_len = is_missing.groupby(run_id).transform("size")

    # Small-gap positions are missing rows whose run length <= max_ffill_gap_s
    small_gap_pos = is_missing & (run_len <= max_ffill_gap_s)

    # Forward-fill candidate
    sat_ffill = df_1s["Connected_Satellite"].ffill()

    # Only fill small gaps; keep big gaps as NaN
    df_1s.loc[small_gap_pos, "Connected_Satellite"] = sat_ffill.loc[small_gap_pos]

    # service_valid: True if we have a serving satellite (observed or small-gap filled)
    service_valid = df_1s["Connected_Satellite"].notna().to_numpy()

    # Put Timestamp back as a column
    df_1s = df_1s.reset_index().rename(columns={"index": "Timestamp"})

    return df_1s, service_valid, observed_mask

def build_graphs(
    service_csv: str,
    tle_path: str,
    ut_lat_deg: float,
    ut_lon_deg: float,
    ut_alt_m: float,
    out_pt: str,
    top_k: int = 32,
    min_elev_deg: float = 10.0,
    knn_k: int = 6,
    max_ffill_gap_s: int = 2,   # small gap threshold
):
    df_raw = pd.read_csv(service_csv)

    # Resample to a strict 1-second grid with controlled filling
    df, service_valid, observed_mask = resample_service_csv_to_1s(
        df_raw, freq="1s", max_ffill_gap_s=max_ffill_gap_s
    )

    # UTC time
    t_series = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    if t_series.isna().any():
        bad = df[t_series.isna()].head(5)
        raise ValueError(f"Timestamp parse failed for some rows, examples:\n{bad}")

    # read TLE
    sats, ts = load_tle_3line(tle_path)

    # UT location
    ut = wgs84.latlon(ut_lat_deg, ut_lon_deg, elevation_m=ut_alt_m)

    graphs = []
    sat_name_list = list(sats.keys())

    for idx, t_utc in enumerate(t_series):
        t_unix = float(t_utc.timestamp())
        t = ts.from_datetime(t_utc.to_pydatetime())

        # Serving satellite name may be missing for big gaps
        serv_name = df.loc[idx, "Connected_Satellite"]
        serv_name = str(serv_name) if pd.notna(serv_name) else None

        # Iterate through all satellites to calculate topocentric coordinates.
        elevs, azims, dists_km, rr_kmps, xyz_km, names = [], [], [], [], [], []

        for name in sat_name_list:
            sat = sats[name]
            try:
                topoc = (sat - ut).at(t)
                alt, az, dist = topoc.altaz()
                elev_deg = float(alt.degrees)
                if elev_deg < min_elev_deg:
                    continue

                dist_km = float(dist.km)
                v = topoc.velocity.km_per_s
                r = topoc.position.km
                r_norm = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) + 1e-9
                rr = float((r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / r_norm)

                elevs.append(elev_deg)
                azims.append(float(az.degrees))
                dists_km.append(dist_km)
                rr_kmps.append(rr)
                xyz_km.append([float(r[0]), float(r[1]), float(r[2])])
                names.append(name)
            except Exception:
                continue

        if len(names) == 0:
            data = Data(
                x=torch.zeros((0, 7), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                service_idx=torch.tensor(-1, dtype=torch.long),
                sat_names=[],
                t_index=idx,
            )
            data.t_unix = torch.tensor(t_unix, dtype=torch.float64)

            # NEW: attach validity flags
            data.service_valid = torch.tensor(int(service_valid[idx]), dtype=torch.long)
            data.observed = torch.tensor(int(observed_mask[idx]), dtype=torch.long)

            graphs.append(data)
            continue

        elevs = np.asarray(elevs, dtype=np.float32)
        azims = np.asarray(azims, dtype=np.float32)
        dists_km = np.asarray(dists_km, dtype=np.float32)
        rr_kmps = np.asarray(rr_kmps, dtype=np.float32)
        xyz_km = np.asarray(xyz_km, dtype=np.float32)

        order = np.argsort(-elevs)
        order = order[:min(top_k, len(order))]

        names_k = [names[i] for i in order]
        elev_k = elevs[order]
        az_k = azims[order]
        dist_k = dists_km[order]
        rr_k = rr_kmps[order]
        xyz_k = xyz_km[order]

        # Only force-including the serving satellite when it is valid
        if serv_name is not None and (serv_name in sats) and (serv_name not in names_k):
            try:
                sat = sats[serv_name]
                topoc = (sat - ut).at(t)
                alt, az, dist = topoc.altaz()
                elev_deg = float(alt.degrees)
                dist_km = float(dist.km)
                v = topoc.velocity.km_per_s
                r = topoc.position.km
                r_norm = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) + 1e-9
                rr = float((r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / r_norm)

                names_k.append(serv_name)
                elev_k = np.concatenate([elev_k, np.array([elev_deg], dtype=np.float32)])
                az_k = np.concatenate([az_k, np.array([float(az.degrees)], dtype=np.float32)])
                dist_k = np.concatenate([dist_k, np.array([dist_km], dtype=np.float32)])
                rr_k = np.concatenate([rr_k, np.array([rr], dtype=np.float32)])
                xyz_k = np.vstack([xyz_k, np.array([[float(r[0]), float(r[1]), float(r[2])]], dtype=np.float32)])
            except Exception:
                pass

        # service_idx is -1 when serving satellite is missing (big gap)
        service_idx = names_k.index(serv_name) if (serv_name is not None and serv_name in names_k) else -1

        is_service = np.zeros((len(names_k),), dtype=np.float32)
        if service_idx >= 0:
            is_service[service_idx] = 1.0

        ones = np.ones((len(names_k),), dtype=np.float32)
        elev_n, az_s, az_c, dist_n, rr_n = normalize_node_features(elev_k, az_k, dist_k, rr_k, min_elev_deg=min_elev_deg)
        x = np.stack([elev_n, az_s, az_c, dist_n, rr_n, is_service, ones], axis=-1).astype(np.float32)

        edge_index = knn_edge_index(xyz_k, k=knn_k)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            service_idx=torch.tensor(service_idx),
            sat_names=names_k,
            t_index=idx,
        )
        data.t_unix = torch.tensor(t_unix, dtype=torch.float64)

        # attach validity flags
        data.service_valid = torch.tensor(int(service_valid[idx]), dtype=torch.long)
        data.observed = torch.tensor(int(observed_mask[idx]), dtype=torch.long)

        graphs.append(data)

        if (idx + 1) % 500 == 0:
            print(f"[INFO] processed {idx+1}/{len(df)}")

    graph_times = np.array([float(g.t_unix.item()) for g in graphs], dtype=np.float64)

    # also save validity arrays for debugging / masking
    torch.save(
        {
            "graphs": graphs,
            "graph_times": graph_times,
            "service_valid": service_valid.astype(np.int8),
            "observed_mask": observed_mask.astype(np.int8),
        },
        out_pt
    )

def check_graphs_pt_time_spacing(graphs_pt_path: str, atol: float = 1e-3, max_show: int = 20):
    """ check if the interval of graph_times is 1s"""
    ckpt = torch.load(graphs_pt_path, map_location="cpu", weights_only=False)

    graphs = ckpt.get("graphs", None)
    if graphs is None:
        raise RuntimeError("graphs.pt missing key: 'graphs'")

    if "graph_times" in ckpt:
        gt = np.asarray(ckpt["graph_times"], dtype=np.float64)
        src = "graph_times"
    else:
        gt = np.asarray([float(g.t_unix.item()) for g in graphs], dtype=np.float64)
        src = "graphs[*].t_unix"

    if gt.size < 2:
        print(f"[CHECK] {graphs_pt_path}: only {gt.size} timestamps, nothing to diff.")
        return

    d = np.diff(gt)
    print(f"[CHECK] source={src}, T={gt.size}")
    print(f"[CHECK] time range: [{gt[0]:.6f}, {gt[-1]:.6f}] (span {gt[-1]-gt[0]:.3f}s)")
    print(f"[CHECK] diff stats: min={d.min():.6f}, median={np.median(d):.6f}, max={d.max():.6f}")

    non_increasing = np.where(d <= 0)[0]  # i means gt[i+1] <= gt[i]
    if non_increasing.size > 0:
        print(f"[ERR] NOT strictly increasing: {non_increasing.size} cases (show up to {max_show})")
        for i in non_increasing[:max_show]:
            print(f"  i={i}: gt[i]={gt[i]:.6f}, gt[i+1]={gt[i+1]:.6f}, diff={d[i]:.6f}")
    else:
        print("[OK] strictly increasing")

    off_1s = np.where(np.abs(d - 1.0) > atol)[0]
    if off_1s.size == 0:
        print(f"[OK] all diffs ~= 1.0s within atol={atol}")
    else:
        print(f"[WARN] diffs NOT ~1.0s: {off_1s.size} cases (show up to {max_show}), atol={atol}")
        for i in off_1s[:max_show]:
            print(f"  i={i}: gt[i]={gt[i]:.6f}, gt[i+1]={gt[i+1]:.6f}, diff={d[i]:.6f}")

    gt_sec = np.floor(gt).astype(np.int64)
    ds = np.diff(gt_sec)

    back = np.where(ds < 0)[0]
    dup  = np.where(ds == 0)[0]
    gap  = np.where(ds > 1)[0]

    print(f"[CHECK] integer-second diffs stats: min={ds.min()}, median={int(np.median(ds))}, max={ds.max()}")
    if back.size > 0:
        print(f"[ERR] second-level time goes backward: {back.size} (show up to {max_show})")
        for i in back[:max_show]:
            print(f"  i={i}: sec[i]={gt_sec[i]}, sec[i+1]={gt_sec[i+1]}, ds={ds[i]}")
    if dup.size > 0:
        print(f"[WARN] duplicated seconds (same floor second): {dup.size} (show up to {max_show})")
        for i in dup[:max_show]:
            print(f"  i={i}: sec={gt_sec[i]} repeated, gt[i]={gt[i]:.6f}, gt[i+1]={gt[i+1]:.6f}")
    if gap.size > 0:
        print(f"[WARN] missing seconds (gap>1): {gap.size} (show up to {max_show})")
        for i in gap[:max_show]:
            print(f"  i={i}: sec[i]={gt_sec[i]}, sec[i+1]={gt_sec[i+1]}, gap={gt_sec[i+1]-gt_sec[i]}s")
            lo = max(0, i-2); hi = min(gt.size-1, i+3)
            for j in range(lo, hi):
                print(f"    j={j}: gt={gt[j]:.6f} sec={gt_sec[j]}")
            print("    ...")

    if (back.size == 0) and (dup.size == 0) and (gap.size == 0):
        print("[OK] second-level timeline is continuous (no gaps/dups/backward).")


    if "graph_times" in ckpt:
        gt2 = np.asarray([float(g.t_unix.item()) for g in graphs], dtype=np.float64)
        max_abs = np.max(np.abs(gt - gt2))
        print(f"[CHECK] graph_times vs graphs[*].t_unix max_abs_diff={max_abs:.6f}s")
        if max_abs > 1e-6:
            bad = np.where(np.abs(gt - gt2) > 1e-6)[0]
            print(f"[WARN] mismatch count={bad.size} (show up to {max_show})")
            for i in bad[:max_show]:
                print(f"  i={i}: graph_times={gt[i]:.6f}, t_unix={gt2[i]:.6f}, diff={gt[i]-gt2[i]:.6f}")

def show_graph_time_gaps(graphs_pt="graphs.pt", service_csv=None, topn=30):
    ckpt = torch.load(graphs_pt, map_location="cpu", weights_only=False)
    gt = np.asarray(ckpt["graph_times"], dtype=np.float64)  # (T,)
    d = np.diff(gt)

    gap_idx = np.where(d != 1.0)[0]  # i means gt[i+1]-gt[i] != 1
    print(f"[GAP] total gaps = {gap_idx.size}")

    if gap_idx.size == 0:
        return

    order = np.argsort(-d[gap_idx])
    gap_idx = gap_idx[order]

    if service_csv is not None:
        df = pd.read_csv(service_csv)
        ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    else:
        df = None
        ts = None

    for k, i in enumerate(gap_idx[:topn]):
        g0 = gt[i]
        g1 = gt[i+1]
        gap = int(round(g1 - g0))
        print(f"[GAP#{k}] i={i} : gt[i]={g0:.3f}, gt[i+1]={g1:.3f}, diff={g1-g0:.3f} (missing {gap-1} seconds)")

        if df is not None:
            print(f"        CSV row {i}:   Timestamp={df.loc[i,'Timestamp']}")
            print(f"        CSV row {i+1}: Timestamp={df.loc[i+1,'Timestamp']}")
            t0 = ts.iloc[i].timestamp()
            t1 = ts.iloc[i+1].timestamp()
            print(f"        parsed diff = {t1 - t0:.3f}s")

if __name__ == "__main__":
    SERVICE_CSV = "../training_data/serving_satellite_20250813.csv"     # Timestamp, Connected_Satellite, Distance
    TLE_TXT = "../training_data/tle_merged.txt"
    OUT_PT = "graphs.pt"

    UT_LAT = 48.461209188573065       # TODO: UT latitude
    UT_LON = -123.3117205      # TODO: UT longitude
    UT_ALT_M = 64.62196379199824     # TODO: UT altitude (m)

    build_graphs(
        service_csv=SERVICE_CSV,
        tle_path=TLE_TXT,
        ut_lat_deg=UT_LAT,
        ut_lon_deg=UT_LON,
        ut_alt_m=UT_ALT_M,
        out_pt=OUT_PT,
        top_k=32,
        min_elev_deg=10.0,
        knn_k=6
    )

    check_graphs_pt_time_spacing("../../202601/dgcn/graphs.pt", atol=1e-3, max_show=20)
    show_graph_time_gaps("graphs.pt", "../training_data/serving_satellite_20250813.csv", topn=30)