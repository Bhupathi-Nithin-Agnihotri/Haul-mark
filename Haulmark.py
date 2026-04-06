

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster       import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model  import Ridge
from sklearn.metrics       import mean_squared_error
data_dir = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge/"
train_files = [
    "telemetry_2026-01-01_2026-01-10.parquet",
    "telemetry_2026-01-11_2026-01-20.parquet",
    "telemetry_2026-02-01_2026-02-10.parquet",
    "telemetry_2026-02-11_2026-02-20.parquet",
    "telemetry_2026-03-01_2026-03-11.parquet",
]
test_files = [
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
]
refuel_files = [
    "rfid_refuels_2026-01-01_2026-02-28.parquet",
    "rfid_refuels_2026-01-01_2026-03-31.parquet",
]
fleet_f = "fleet.csv"
id_mp_f = "id_mapping_new.csv"

df_tr      = pd.concat([pd.read_parquet(data_dir + f) for f in train_files], ignore_index=True)
df_te      = pd.concat([pd.read_parquet(data_dir + f) for f in test_files],  ignore_index=True)
df_rf      = pd.concat([pd.read_parquet(data_dir + f) for f in refuel_files], ignore_index=True)
id_mapping = pd.read_csv(data_dir + id_mp_f)
fleet      = pd.read_csv(data_dir + fleet_f)

df_rf.drop_duplicates(inplace=True)
df_rf.sort_values(['vehicle', 'ts'], inplace=True)
print(f"Train rows : {len(df_tr):,}  |  Test rows : {len(df_te):,}  |  Refuel rows : {len(df_rf):,}")


drop_cols = [
    'received_ts', 'gnss_pdop', 'gnss_hdop',
    'gsm_signal', 'gsm_operator',
    'battery_level', 'battery_current', 'battery_voltage', 'external_voltage',
    'hmr_dpr', 'prod_hr_dpr', 'idle_hr_dpr',
    'maint_hr_dpr', 'bd_hr_dpr', 'km_dpr',
    'total_trip', 'tonnage',
    'rain_loss', 'dense_fog',
]
df_tr = df_tr.drop(columns=drop_cols, errors='ignore')
df_te = df_te.drop(columns=drop_cols, errors='ignore')


if 'operator_id' in df_tr.columns:
    nan_pct = df_tr['operator_id'].isna().mean()
    print(f"operator_id NaN rate: {nan_pct:.1%}")
    USE_OPERATOR = nan_pct < 0.90
    if not USE_OPERATOR:
        print("  → Too sparse, dropping.")
        df_tr.drop(columns=['operator_id'], inplace=True, errors='ignore')
        df_te.drop(columns=['operator_id'], inplace=True, errors='ignore')
    else:
        print("  → Keeping! Will target-encode as driver_fuel_mean.")
else:
    USE_OPERATOR = False
    print("operator_id not in data.")

df_tr = df_tr[df_tr['satellites'] >= 4]
df_tr.drop(columns=['satellites'], inplace=True)
df_te.drop(columns=['satellites'], inplace=True)
df_tr = df_tr[df_tr['vehicle'].str.startswith("Dump", na=False)]
df_te = df_te[df_te['vehicle'].str.startswith("Dump", na=False)]
df_rf = df_rf[df_rf['vehicle'].str.startswith("Dump", na=False)]
def reduce_memory(df):
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes('int64').columns:
        df[col] = df[col].astype('int32')
    return df
df_tr = reduce_memory(df_tr)
# =====================================================================
# DUMP DETECTION — Step 1: Extract Stop Events from Raw Telemetry
# =====================================================================
# Core idea: a dumper repeats a fixed loop each cycle.
# The dump site is the location visited most frequently with consistent
# short dwell times. We detect it purely from GPS + speed — no analog_input_1 needed.

SPEED_STOP_THRESH = 2.0    # km/h — below this = stopped
MIN_STOP_DUR_SEC  = 60     # minimum stop duration to count as a "stop event"

def extract_stop_events(df):
    """
    Identify contiguous stopped segments per vehicle+shift.
    Returns:
      - stop_events: one row per stop event with centroid, duration, ts_start/end
      - df with '_seg' and '_is_stopped' columns added (for mapping back later)
    """
    df = df.sort_values(['vehicle','ts']).copy()

    df['_is_stopped'] = (df['speed'] < SPEED_STOP_THRESH).astype(np.int8)

    # New segment whenever: stopped/moving transitions, or vehicle/shift changes
    v_grp = df.groupby('vehicle', sort=False)
    shift_change = (
        (df['_is_stopped'] != v_grp['_is_stopped'].shift(1)) |
        (df['vehicle']     != df['vehicle'].shift(1)) |
        (df['date_dpr']    != v_grp['date_dpr'].shift(1)) |
        (df['shift_dpr']   != v_grp['shift_dpr'].shift(1))
    )
    df['_seg'] = shift_change.cumsum().astype(np.int32)

    stopped = df[df['_is_stopped'] == 1]

    events = (
        stopped
        .groupby(['vehicle','date_dpr','shift_dpr','_seg'], observed=True)
        .agg(
            lat_c    = ('latitude',  'mean'),
            lon_c    = ('longitude', 'mean'),
            alt_c    = ('altitude',  'mean'),
            ts_start = ('ts', 'min'),
            ts_end   = ('ts', 'max'),
            n_pings  = ('speed', 'count'),
        )
        .reset_index()
    )

    events['duration_sec'] = (events['ts_end'] - events['ts_start']).dt.total_seconds()
    events = events[events['duration_sec'] >= MIN_STOP_DUR_SEC].reset_index(drop=True)

    return events, df[['_seg','_is_stopped']]

print("Extracting stop events (train)…")
stop_ev_tr, seg_cols_tr = extract_stop_events(df_tr)
df_tr[['_seg','_is_stopped']] = seg_cols_tr[['_seg','_is_stopped']].values

print("Extracting stop events (test)…")
stop_ev_te, seg_cols_te = extract_stop_events(df_te)
df_te[['_seg','_is_stopped']] = seg_cols_te[['_seg','_is_stopped']].values

print(f"Train stop events : {len(stop_ev_tr):,}")
print(f"Test  stop events : {len(stop_ev_te):,}")
print(f"\nDuration stats (train):")
print(stop_ev_tr['duration_sec'].describe())

df_rf = df_rf.sort_values(['vehicle', 'ts'])
 
df_shift_rf = (
    df_rf.groupby(['vehicle', 'date_dpr', 'shift_dpr'], observed=True)
    .agg(fuel_rf=('litres', 'sum'))
    .reset_index()
)
def compute_actual_consumption(df, df_rf_raw):
    """
    For each (vehicle, date_dpr, shift_dpr):
      - Sort telemetry by ts
      - Identify refuel events WITHIN the shift from raw refuel data
      - Actual consumed = (first_fuel - last_fuel) + refuels_in_shift
    This is far more accurate than just first-last (which goes NEGATIVE on refuel shifts)
    """
    df = df.sort_values(['vehicle', 'ts'])
 
    # First/last fuel reading per shift
    grp = df.groupby(['vehicle', 'date_dpr', 'shift_dpr'], observed=True)
    fuel_bounds = grp['fuel_volume'].agg(
        fuel_first='first',
        fuel_last='last',
        n_readings='count'
    ).reset_index()
 
    # Shift start/end timestamps (to join refuels)
    ts_bounds = grp['ts'].agg(
        shift_start='min',
        shift_end='max'
    ).reset_index()
 
    fuel_bounds = fuel_bounds.merge(ts_bounds, on=['vehicle', 'date_dpr', 'shift_dpr'])
 
    # Merge refuels that fall within each shift's time window
    # Use a cross-join approach via merge + filter
    rf_in_shift = df_rf_raw[['vehicle', 'ts', 'litres']].copy()
    rf_in_shift = rf_in_shift.rename(columns={'ts': 'rf_ts', 'litres': 'rf_litres'})
 
    merged = fuel_bounds.merge(rf_in_shift, on='vehicle', how='left')
    merged = merged[
        (merged['rf_ts'] >= merged['shift_start']) &
        (merged['rf_ts'] <= merged['shift_end'])
    ]
    refuels_in_shift = (
        merged.groupby(['vehicle', 'date_dpr', 'shift_dpr'])['rf_litres']
        .sum()
        .reset_index()
        .rename(columns={'rf_litres': 'refuels_in_shift'})
    )
 
    fuel_bounds = fuel_bounds.merge(
        refuels_in_shift, on=['vehicle', 'date_dpr', 'shift_dpr'], how='left'
    )
    fuel_bounds['refuels_in_shift'] = fuel_bounds['refuels_in_shift'].fillna(0)
 
    # Actual consumed = drop in tank + whatever was added back in
    fuel_bounds['fuel_consumed'] = (
        fuel_bounds['fuel_first'] - fuel_bounds['fuel_last']
        + fuel_bounds['refuels_in_shift']
    )
 
    # Sanity filter: drop implausible values
    # (negative = sensor error, >1500 = impossible for any dumper)
    fuel_bounds = fuel_bounds[
        (fuel_bounds['fuel_consumed'] >= 0) &
        (fuel_bounds['fuel_consumed'] <= 1500) &
        (fuel_bounds['n_readings'] >= 10)  # at least 10 GPS pings
    ]
 
    return fuel_bounds[['vehicle', 'date_dpr', 'shift_dpr', 'fuel_consumed',
                         'fuel_first', 'fuel_last', 'n_readings', 'refuels_in_shift']]

# df_rf_raw = df_rf.copy()
# del df_rf
 
fuel_tr = compute_actual_consumption(df_tr, df_rf)
print(f"Train shift-rows with valid consumption: {len(fuel_tr):,}")
print(fuel_tr['fuel_consumed'].describe())

def compute_economy_speed_range(shift_summary, fuel_summary,
                                 fallback_low=20, fallback_high=50,
                                 n_bins=12, pad_kmh=8):
    """
    Find the speed band with the lowest median fuel/km from training data.

    Uses p75_speed as a cruising-speed proxy per shift (more stable than
    mean_speed which is pulled down by idle time).

    Parameters
    ----------
    shift_summary : DataFrame with [total_dist_km, p75_speed] — from
                    engineer_shift_features on the TRAIN set.
                    Call this AFTER Cell 12 but pass fuel_tr from Cell 7.
    fuel_summary  : fuel_tr DataFrame with [vehicle, date_dpr, shift_dpr,
                    fuel_consumed].
    pad_kmh       : km/h added on each side of the optimal bin.

    Returns
    -------
    (economy_low, economy_high) as floats — plug into ECONOMY_LOW / ECONOMY_HIGH.
    """
    df = shift_summary.merge(
        fuel_summary[['vehicle', 'date_dpr', 'shift_dpr', 'fuel_consumed']],
        on=['vehicle', 'date_dpr', 'shift_dpr'],
        how='inner',
    )

    # Only well-driven shifts with meaningful distance
    df = df[(df['total_dist_km'] > 5) & (df['p75_speed'] > 5) & (df['p75_speed'] < 90)]
    if len(df) < 50:
        print(f"  Not enough data ({len(df)} rows) — using fallback {fallback_low}/{fallback_high}")
        return fallback_low, fallback_high

    df['fuel_per_km'] = df['fuel_consumed'] / df['total_dist_km']

    # Bin cruising speed
    df['speed_bin'] = pd.cut(df['p75_speed'], bins=n_bins)
    bin_stats = (
        df.groupby('speed_bin', observed=True)['fuel_per_km']
        .agg(median='median', count='count')
        .dropna()
    )
    # Require at least 10 shifts per bin to trust the median
    bin_stats = bin_stats[bin_stats['count'] >= 10]

    if bin_stats.empty:
        print(f"  Bins too sparse — using fallback {fallback_low}/{fallback_high}")
        return fallback_low, fallback_high

    best_bin   = bin_stats['median'].idxmin()
    center     = (best_bin.left + best_bin.right) / 2
    eco_low    = round(float(max(5,  best_bin.left  - pad_kmh)), 1)
    eco_high   = round(float(min(90, best_bin.right + pad_kmh)), 1)

    print(f"  Optimal cruising speed centre : {center:.1f} km/h")
    print(f"  Economy band from train data  : {eco_low} - {eco_high} km/h")
    print(f"  (fallback would have been     : {fallback_low} - {fallback_high} km/h)")
    print(f"  Bin stats (median fuel/km):\n{bin_stats.sort_index().to_string()}")
    return eco_low, eco_high
# =====================================================================
# DUMP DETECTION — Step 2: Approach & Departure Speed per Stop
# =====================================================================
# Loaded truck (heavy) → approaches dump site SLOWER
# After tipping (empty) → departs FASTER
# So: departure_speed > approach_speed  →  positive asymmetry  →  dump signal
# At load site: opposite (leaves heavier → departs slower)

SPEED_WINDOW_SEC = 120   # look 2 min before/after each stop

def add_approach_depart_speed(df_sorted, stop_events, window_sec=SPEED_WINDOW_SEC):
    """
    For each stop event, compute mean speed of moving rows in the
    window_sec before ts_start and after ts_end.
    Uses per-(vehicle,date) numpy arrays for speed.
    """
    window_ns = int(window_sec * 1e9)   # pandas Timestamp is nanosecond-based

    # Moving rows only, keep minimal columns
    moving = (
        df_sorted[df_sorted['speed'] >= SPEED_STOP_THRESH]
        [['vehicle','date_dpr','ts','speed']]
        .copy()
    )
    moving['ts_ns'] = moving['ts'].astype(np.int64)

    # Build lookup dict for fast access
    mov_grps = {k: v[['ts_ns','speed']].values
                for k, v in moving.groupby(['vehicle','date_dpr'], observed=True)}

    app_spd, dep_spd = [], []
    for row in stop_events.itertuples(index=False):
        arr = mov_grps.get((row.vehicle, row.date_dpr))
        if arr is None or len(arr) == 0:
            app_spd.append(np.nan); dep_spd.append(np.nan); continue

        t0 = row.ts_start.value    # nanoseconds
        t1 = row.ts_end.value
        ts  = arr[:,0]
        spd = arr[:,1]

        app_mask = (ts >= t0 - window_ns) & (ts < t0)
        dep_mask = (ts >  t1) & (ts <= t1 + window_ns)

        app_spd.append(float(spd[app_mask].mean()) if app_mask.any() else np.nan)
        dep_spd.append(float(spd[dep_mask].mean()) if dep_mask.any() else np.nan)

    stop_events = stop_events.copy()
    stop_events['approach_speed'] = app_spd
    stop_events['depart_speed']   = dep_spd
    # Positive  →  left faster than arrived  →  dump signal
    # Negative  →  left slower than arrived  →  load signal
    stop_events['speed_asym'] = stop_events['depart_speed'] - stop_events['approach_speed']
    return stop_events

print("Computing approach/departure speeds (train)…")
df_tr_sorted = df_tr.sort_values(['vehicle','ts'])
stop_ev_tr = add_approach_depart_speed(df_tr_sorted, stop_ev_tr)

print("Computing approach/departure speeds (test)…")
df_te_sorted = df_te.sort_values(['vehicle','ts'])
stop_ev_te = add_approach_depart_speed(df_te_sorted, stop_ev_te)

print("Done.")
print(stop_ev_tr[['duration_sec','approach_speed','depart_speed','speed_asym']].describe().round(2))

# =====================================================================
# DUMP DETECTION — Step 3: Identify Dump Site per (Vehicle, Date)
# =====================================================================
# Per (vehicle, date), cluster stop centroids → 2-3 clusters.
# Score each cluster on three signals:
#   1. n_visits        — dump site is visited MOST (one per haul cycle)
#   2. cv_duration     — dump site has CONSISTENT dwell (mechanical process)
#   3. speed_asym_mean — dump site has POSITIVE asymmetry (leaves lighter)
# Highest combined score = dump cluster.

from sklearn.metrics import silhouette_score as _sil

def identify_dump_site_per_day(stop_events):
    """
    Adds columns to stop_events:
      stop_cluster   - which cluster this stop belongs to
      is_dump_stop   - 1 if this stop is at the identified dump site
      dump_score     - score of this stop's cluster (diagnostic)
    """
    results = []

    for (vehicle, date), grp in stop_events.groupby(['vehicle','date_dpr'], observed=True):
        grp = grp.copy()
        n = len(grp)

        if n < 4:
            grp['stop_cluster'] = 0
            grp['is_dump_stop'] = 0
            grp['dump_score']   = 0.0
            results.append(grp)
            continue

        coords = grp[['lat_c','lon_c']].values.astype(float)

        # Handle degenerate case: all stops at same location
        coord_spread = np.ptp(coords, axis=0).max()
        if coord_spread < 1e-5:
            grp['stop_cluster'] = 0
            # Single location: use duration+asymmetry heuristic
            is_dump = (
                (grp['duration_sec'] < 300) &          # short stop (< 5 min)
                (grp['speed_asym'].fillna(0) > 0)      # leaves faster
            ).astype(int)
            grp['is_dump_stop'] = is_dump
            grp['dump_score']   = 0.0
            results.append(grp)
            continue

        # Find best k
        best_k, best_sil = 2, -1.1
        for k in range(2, min(5, n)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            labels = km.fit_predict(coords)
            if len(np.unique(labels)) < k:
                continue
            try:
                s = _sil(coords, labels)
            except Exception:
                continue
            if s > best_sil:
                best_sil, best_k = s, k

        km_f = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=200)
        grp['stop_cluster'] = km_f.fit_predict(coords)

        # Score each cluster
        cstats = (
            grp.groupby('stop_cluster')
            .agg(
                n_visits   = ('duration_sec', 'count'),
                mean_dur   = ('duration_sec', 'mean'),
                std_dur    = ('duration_sec', 'std'),
                asym_mean  = ('speed_asym',   'mean'),
            )
            .reset_index()
        )
        cstats['std_dur']  = cstats['std_dur'].fillna(0)
        cstats['asym_mean'] = cstats['asym_mean'].fillna(0)
        # Coefficient of variation: lower = more consistent
        cstats['cv_dur']   = cstats['std_dur'] / cstats['mean_dur'].clip(lower=1)

        # Normalise each signal to [0,1] before combining
        def norm01(s):
            r = s.max() - s.min()
            return (s - s.min()) / r if r > 1e-9 else s * 0

        cstats['s_visits'] = norm01(cstats['n_visits'])
        cstats['s_consist'] = norm01(1 / (cstats['cv_dur'] + 0.1))   # high = consistent
        cstats['s_asym']   = norm01(cstats['asym_mean'])

        # Weighted sum: visits most important, then consistency, then asymmetry
        cstats['dump_score'] = (
            0.45 * cstats['s_visits'] +
            0.35 * cstats['s_consist'] +
            0.20 * cstats['s_asym']
        )

        dump_cluster_id = int(cstats.loc[cstats['dump_score'].idxmax(), 'stop_cluster'])

        grp['is_dump_stop'] = (grp['stop_cluster'] == dump_cluster_id).astype(np.int8)
        grp['dump_score']   = grp['stop_cluster'].map(
            cstats.set_index('stop_cluster')['dump_score']
        )
        results.append(grp)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    return out

print("Identifying dump site per (vehicle, date) — train…")
stop_ev_tr = identify_dump_site_per_day(stop_ev_tr)
print("Identifying dump site per (vehicle, date) — test…")
stop_ev_te = identify_dump_site_per_day(stop_ev_te)

n_dump_stops_tr = stop_ev_tr['is_dump_stop'].sum()
n_total_stops_tr = len(stop_ev_tr)
print(f"\nTrain: {n_dump_stops_tr:,} / {n_total_stops_tr:,} stop events labelled as dump  ({100*n_dump_stops_tr/n_total_stops_tr:.1f}%)")
print(f"Test : {stop_ev_te['is_dump_stop'].sum():,} / {len(stop_ev_te):,}")

# =====================================================================
# DUMP DETECTION — Step 4: Mark is_dumping in Raw Telemetry
# =====================================================================
# The _seg column from step 1 links every raw row to a stop event.
# Rows in a dump-stop segment → is_dumping=1.
# Moving rows → is_dumping=0 (never dumping while moving).
# Optional: where analog_input_1 is reliable, use it as ground truth override.

def mark_is_dumping(df, stop_events_labeled):
    """Map dump stop segments back to individual telemetry rows."""
    dump_segs = set(
        stop_events_labeled.loc[stop_events_labeled['is_dump_stop'] == 1, '_seg']
    )
    # Rows in a dump-stop segment AND stopped
    is_dump = (
        df['_seg'].isin(dump_segs) & (df['_is_stopped'] == 1)
    ).astype(np.int8)
    return is_dump

df_tr['is_dumping'] = mark_is_dumping(df_tr, stop_ev_tr)
df_te['is_dumping'] = mark_is_dumping(df_te, stop_ev_te)

# ── Optional: override with reliable analog_input_1 ground truth
DUMP_COL   = 'analog_input_1'
DUMP_THRESH = 2.5

has_dump_col = DUMP_COL in df_tr.columns
if has_dump_col:
    def dump_switch_reliability(df, col=DUMP_COL, thresh=DUMP_THRESH):
        if col not in df.columns:
            return pd.DataFrame(columns=['vehicle','date_dpr','reliable'])
        stats = (
            df.groupby(['vehicle','date_dpr'])[col]
            .agg(n_valid=lambda x: x.notna().sum(),
                 max_volt='max',
                 dump_frac=lambda x: (x >= thresh).mean(),
                 base_frac=lambda x: (x.fillna(0) < 0.5).mean())
            .reset_index()
        )
        stats['reliable'] = (
            (stats['n_valid']   >= 50)  &
            (stats['max_volt']  >= 3.5) &
            (stats['dump_frac'] >= 0.02) &
            (stats['dump_frac'] <= 0.60) &
            (stats['base_frac'] >= 0.40)
        )
        return stats

    rel_tr = dump_switch_reliability(df_tr)
    df_tr  = df_tr.merge(rel_tr[['vehicle','date_dpr','reliable']],
                          on=['vehicle','date_dpr'], how='left')
    df_tr['reliable'] = df_tr['reliable'].fillna(False)

    # Where analog_input_1 is reliable, its reading overrides the inferred label
    override_mask = df_tr['reliable']
    df_tr.loc[override_mask, 'is_dumping'] = (
        (df_tr.loc[override_mask, DUMP_COL] >= DUMP_THRESH).astype(np.int8)
    )
    n_overridden = override_mask.sum()
    print(f"analog_input_1 override applied to {n_overridden:,} rows "
          f"({100*n_overridden/len(df_tr):.1f}%)")
    df_tr.drop(columns=['reliable'], inplace=True, errors='ignore')
else:
    print("No analog_input_1 column — using stop-event detection only.")

pct_tr = 100 * df_tr['is_dumping'].mean()
pct_te = 100 * df_te['is_dumping'].mean()
print(f"\nFinal is_dumping=1  →  train: {pct_tr:.2f}%  |  test: {pct_te:.2f}%")

ECONOMY_LOW, ECONOMY_HIGH = 20, 40
SPEED_STOP = 2
def haversine_series(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))


def engineer_shift_features(df, fleet_df):
    df = df.sort_values(['vehicle', 'ts']).copy()
    g  = df.groupby('vehicle', sort=False)

    # ── Point-to-point distance & elevation deltas
    df['lat_prev'] = g['latitude'].shift(1)
    df['lon_prev'] = g['longitude'].shift(1)
    df['alt_prev'] = g['altitude'].shift(1)
    df['ts_prev']  = g['ts'].shift(1)

    same_shift = (
        (df['date_dpr']  == g['date_dpr'].shift(1)) &
        (df['shift_dpr'] == g['shift_dpr'].shift(1))
    )
    df.loc[~same_shift, ['lat_prev', 'lon_prev', 'alt_prev', 'ts_prev']] = np.nan

    mask = df['lat_prev'].notna()
    df.loc[mask, 'seg_dist_m'] = haversine_series(
        df.loc[mask, 'lat_prev'], df.loc[mask, 'lon_prev'],
        df.loc[mask, 'latitude'], df.loc[mask, 'longitude'],
    )
    df['seg_dist_m'] = df['seg_dist_m'].clip(lower=0)
    df.loc[mask, 'alt_delta'] = df.loc[mask, 'altitude'] - df.loc[mask, 'alt_prev']
    df.loc[mask, 'dt_sec']    = (df.loc[mask, 'ts'] - df.loc[mask, 'ts_prev']).dt.total_seconds()

    # ── Stop / moving flags
    df['is_stopped'] = ((df['speed'] < SPEED_STOP) & (df['ignition'] == 1)).astype(int)
    df['is_moving']  = (df['speed'] >= SPEED_STOP).astype(int)

    # ── ★ Economy speed flags
    df['in_economy']   = ((df['speed'] >= ECONOMY_LOW) & (df['speed'] <= ECONOMY_HIGH)).astype(int)
    df['is_overspeed'] = (df['speed'] > ECONOMY_HIGH).astype(int)
    df['is_crawling']  = ((df['speed'] > SPEED_STOP) & (df['speed'] < ECONOMY_LOW)).astype(int)
    # Distance in each band (per ping)
    seg = df['seg_dist_m'].fillna(0)
    df['dist_economy'] = df['in_economy']   * seg
    df['dist_over']    = df['is_overspeed'] * seg
    df['dist_crawl']   = df['is_crawling']  * seg

    # ── ★ Heading / angle features
    has_heading = 'heading' in df.columns
    if has_heading:
        df['hdg_prev'] = g['heading'].shift(1)
        raw_delta      = (df['heading'] - df['hdg_prev']).abs()
        df['hdg_delta']     = raw_delta.where(raw_delta <= 180, 360 - raw_delta)
        df.loc[~same_shift, 'hdg_delta'] = np.nan
        df['is_sharp_turn'] = (df['hdg_delta'] > 45).astype(int)
        df['is_harsh_turn'] = (df['hdg_delta'] > 90).astype(int)

    # ── ★ Gradient / climb-work features
    df.loc[mask, 'grade_pct'] = (
        df.loc[mask, 'alt_delta'] / df.loc[mask, 'seg_dist_m'].clip(lower=0.5)
    ).clip(-50, 50) * 100
    df['is_uphill']   = (df['alt_delta'].fillna(0) >  0.5).astype(int)
    df['is_downhill'] = (df['alt_delta'].fillna(0) < -0.5).astype(int)
    # Climb work proxy: Σ(alt_gain × dist)  →  proportional to gravitational work done
    df['climb_work']  = (df['alt_delta'].clip(lower=0) * seg).fillna(0)

    # ── ★ operator_id: modal operator per shift (one driver dominates each shift)
    has_op = USE_OPERATOR and 'operator_id' in df.columns

    # ── Aggregate
    grp = df.groupby(['vehicle', 'date_dpr', 'shift_dpr'], observed=True)

    agg_dict = dict(
        # Kinematics
        total_dist_m      = ('seg_dist_m',    'sum'),
        mean_speed        = ('speed',          'mean'),
        max_speed         = ('speed',          'max'),
        p75_speed         = ('speed',          lambda x: x.quantile(0.75)),
        std_speed         = ('speed',          'std'),
        # Elevation
        alt_std           = ('altitude',       'std'),
        alt_range         = ('altitude',       lambda x: x.max() - x.min()),
        net_lift          = ('altitude',       lambda x: x.iloc[-1] - x.iloc[0]),
        gross_elev_gain   = ('alt_delta',      lambda x: x[x > 0].sum()),
        gross_elev_loss   = ('alt_delta',      lambda x: x[x < 0].abs().sum()),
        # Stop-go
        n_stops           = ('is_stopped',     lambda x: (x.diff() == 1).sum()),
        pct_stopped       = ('is_stopped',     'mean'),
        pct_moving        = ('is_moving',      'mean'),
        # Shift meta
        n_pings           = ('speed',          'count'),
        ignition_on_frac  = ('ignition',       'mean'),
        shift_duration_hr = ('ts',             lambda x: (x.max() - x.min()).total_seconds() / 3600),
        # Positions
        lat_start         = ('latitude',       'first'),
        lon_start         = ('longitude',      'first'),
        lat_end           = ('latitude',       'last'),
        lon_end           = ('longitude',      'last'),
        alt_start         = ('altitude',       'first'),
        alt_end           = ('altitude',       'last'),
        # ★ Economy speed
        pct_economy_speed = ('in_economy',     'mean'),
        pct_overspeed     = ('is_overspeed',   'mean'),
        pct_crawling      = ('is_crawling',    'mean'),
        dist_economy_km   = ('dist_economy',   lambda x: x.sum() / 1000),
        dist_overspeed_km = ('dist_over',      lambda x: x.sum() / 1000),
        dist_crawl_km     = ('dist_crawl',     lambda x: x.sum() / 1000),
        # ★ Gradient / climb
        pct_uphill        = ('is_uphill',      'mean'),
        pct_downhill      = ('is_downhill',    'mean'),
        total_climb_work  = ('climb_work',     'sum'),
        mean_grade_uphill = ('grade_pct',      lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        max_grade         = ('grade_pct',      lambda x: x.abs().max()),
    )

    if has_heading:
        agg_dict.update(dict(
            # ★ Heading / angle
            heading_std          = ('heading',       'std'),
            mean_heading_change  = ('hdg_delta',     'mean'),
            n_sharp_turns        = ('is_sharp_turn', 'sum'),
            n_harsh_turns        = ('is_harsh_turn', 'sum'),
        ))

    agg = grp.agg(**agg_dict).reset_index()

    # ── Derived features
    agg['total_dist_km']   = agg['total_dist_m'] / 1000
    agg['dist_per_hr']     = agg['total_dist_km'] / agg['shift_duration_hr'].clip(lower=0.1)
    agg['net_lift']        = agg['net_lift'].fillna(0)
    agg['gross_elev_gain'] = agg['gross_elev_gain'].fillna(0)
    agg['gross_elev_loss'] = agg['gross_elev_loss'].fillna(0)
    agg['stop_density']    = agg['n_stops'] / agg['total_dist_km'].clip(lower=0.1)

    # ★ Economy ratio: fraction of moving time at economy speed
    agg['economy_ratio']     = agg['pct_economy_speed'] / agg['pct_moving'].clip(lower=0.01)
    # ★ Overspeed intensity: how far above the limit on average when speeding
    agg['overspeed_excess']  = (agg['max_speed'] - ECONOMY_HIGH).clip(lower=0) * agg['pct_overspeed']
    # ★ Time outside economy (in minutes), useful absolute scale
    agg['time_out_economy_min'] = (
        (agg['pct_overspeed'] + agg['pct_crawling']) * agg['shift_duration_hr'] * 60
    )
    # ★ Climb work per km (how hilly is the route?)
    agg['climb_work_per_km'] = agg['total_climb_work'] / agg['total_dist_km'].clip(lower=0.1)

    if has_heading:
        agg['sharp_turns_per_km']    = agg['n_sharp_turns'] / agg['total_dist_km'].clip(lower=0.1)
        agg['heading_change_per_km'] = (
            agg['mean_heading_change'].fillna(0) * agg['n_pings']
            / agg['total_dist_km'].clip(lower=0.1)
        )

    # Calendar
    shift_map = {'A': 0, 'B': 1, 'C': 2}
    agg['shift_enc'] = agg['shift_dpr'].astype(str).map(shift_map).fillna(-1).astype(int)
    agg['dow']       = pd.to_datetime(agg['date_dpr']).dt.dayofweek

    # Fleet info
    agg = agg.merge(fleet_df[['vehicle', 'tankcap', 'dump_switch']], on='vehicle', how='left')

    # Vehicle label encoding
    le = LabelEncoder()
    agg['vehicle_enc'] = le.fit_transform(agg['vehicle'].astype(str))

    # ★ operator_id: modal operator per shift
    if has_op:
      op_mode = (
          df.groupby(['vehicle', 'date_dpr', 'shift_dpr'], observed=True)['operator_id']
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
          .reset_index()
          .rename(columns={'operator_id': 'operator_mode'})
      )
      agg = agg.merge(op_mode, on=['vehicle', 'date_dpr', 'shift_dpr'], how='left')

    return agg, le

print("Engineering train features…")
shift_tr, le_vehicle = engineer_shift_features(df_tr, fleet)
print(f"  → {len(shift_tr):,} shift rows, {shift_tr.shape[1]} columns")

print("Engineering test features…")
shift_te, _ = engineer_shift_features(df_te, fleet)
shift_te['vehicle_enc'] = shift_te['vehicle'].map(
    dict(zip(le_vehicle.classes_, le_vehicle.transform(le_vehicle.classes_)))
).fillna(-1).astype(int)
print(f"  → {len(shift_te):,} shift rows")
print("Computing economy speed range from training data…")
# First pass needed to get shift_tr with p75_speed (use defaults for now)
# shift_tr_tmp, _ = engineer_shift_features(df_tr, fleet)   # already done above
# ECONOMY_LOW, ECONOMY_HIGH = compute_economy_speed_range(shift_tr, fuel_tr)

# # Re-run feature engineering with the empirical limits baked in
# print("\nRe-running feature engineering with empirical economy limits…")
# shift_tr, le_vehicle = engineer_shift_features(df_tr, fleet)
# shift_te, _          = engineer_shift_features(df_te, fleet)
# shift_te['vehicle_enc'] = shift_te['vehicle'].map(
#     dict(zip(le_vehicle.classes_, le_vehicle.transform(le_vehicle.classes_)))
# ).fillna(-1).astype(int)
# print("Done.")

def add_route_clusters(tr, te, n_clusters=20):
    coords = ['lat_start', 'lon_start', 'lat_end', 'lon_end']
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tr = tr.copy(); te = te.copy()
    tr['route_id'] = km.fit_predict(tr[coords].fillna(0).values.astype(np.float64))
    te['route_id'] = km.predict(   te[coords].fillna(0).values.astype(np.float64))
    return tr, te, km

shift_tr, shift_te, route_km = add_route_clusters(shift_tr, shift_te, n_clusters=20)
print("Route clusters added")

# =====================================================================
# DUMP DETECTION — Step 5: Dump-Cycle Features per Shift
# =====================================================================
# These are derived from stop_events (already labelled), not raw rows,
# so they are fast to compute.

def compute_dump_shift_features(stop_events, df_raw):
    """
    From labelled stop events + raw telemetry, compute per-shift dump features.
    """
    dump_se  = stop_events[stop_events['is_dump_stop'] == 1].copy()
    grp_cols = ['vehicle','date_dpr','shift_dpr']

    # ── From stop events directly
    dump_agg = dump_se.groupby(grp_cols, observed=True).agg(
        n_dumps           = ('duration_sec', 'count'),
        dump_duration_hr  = ('duration_sec', lambda x: x.sum() / 3600),
        avg_dump_dur_min  = ('duration_sec', lambda x: x.mean() / 60),
        cv_dump_dur       = ('duration_sec', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        # Speed asymmetry: mean over all dump stops in shift
        mean_speed_asym   = ('speed_asym',   'mean'),
        mean_approach_spd = ('approach_speed','mean'),
        mean_depart_spd   = ('depart_speed',  'mean'),
    ).reset_index()

    # ── Cycle time: time between consecutive dump starts in same shift
    dump_se_sorted = dump_se.sort_values(['vehicle','date_dpr','shift_dpr','ts_start'])
    dump_se_sorted['ts_prev_dump'] = dump_se_sorted.groupby(grp_cols, observed=True)['ts_start'].shift(1)
    dump_se_sorted['cycle_time_min'] = (
        (dump_se_sorted['ts_start'] - dump_se_sorted['ts_prev_dump'])
        .dt.total_seconds() / 60
    )
    cycle_agg = (
        dump_se_sorted.dropna(subset=['cycle_time_min'])
        .groupby(grp_cols, observed=True)['cycle_time_min']
        .mean()
        .reset_index(name='avg_cycle_time_min')
    )

    # ── pct_dumping from raw rows (fraction of shift time in dump state)
    pct_agg = (
        df_raw.groupby(grp_cols, observed=True)['is_dumping']
        .mean()
        .reset_index(name='pct_dumping')
    )

    # ── Merge all together
    out = dump_agg.merge(cycle_agg, on=grp_cols, how='left')
    out = out.merge(pct_agg,   on=grp_cols, how='left')
    out['dumps_per_hr'] = out['n_dumps'] / out['dump_duration_hr'].clip(lower=1e-3)
    # Interaction: how much dump work happened
    out['dump_work_score'] = out['n_dumps'] * out['avg_dump_dur_min']

    # Fill shifts where no dump was detected
    fill_zero = ['n_dumps','dump_duration_hr','avg_dump_dur_min','cv_dump_dur',
                 'dumps_per_hr','dump_work_score','pct_dumping']
    for c in fill_zero:
        out[c] = out[c].fillna(0)

    fill_nan = ['mean_speed_asym','mean_approach_spd','mean_depart_spd','avg_cycle_time_min']
    for c in fill_nan:
        out[c] = out[c].fillna(out[c].median() if c in out else np.nan)

    return out

DUMP_FEAT_COLS = [
    'n_dumps', 'dump_duration_hr', 'avg_dump_dur_min', 'cv_dump_dur',
    'pct_dumping', 'dumps_per_hr', 'dump_work_score',
    'mean_speed_asym', 'mean_approach_spd', 'mean_depart_spd',
    'avg_cycle_time_min',
]

print("Computing dump-cycle shift features (train)…")
dump_feats_tr = compute_dump_shift_features(stop_ev_tr, df_tr)
print("Computing dump-cycle shift features (test)…")
dump_feats_te = compute_dump_shift_features(stop_ev_te, df_te)

# Merge into shift-level DataFrames (after engineer_shift_features)
shift_tr = shift_tr.merge(
    dump_feats_tr[['vehicle','date_dpr','shift_dpr']+DUMP_FEAT_COLS],
    on=['vehicle','date_dpr','shift_dpr'], how='left'
)
shift_te = shift_te.merge(
    dump_feats_te[['vehicle','date_dpr','shift_dpr']+DUMP_FEAT_COLS],
    on=['vehicle','date_dpr','shift_dpr'], how='left'
)
for c in DUMP_FEAT_COLS:
    shift_tr[c] = shift_tr[c].fillna(0)
    shift_te[c] = shift_te[c].fillna(0)

# Correlation check
merged_check = shift_tr.merge(
    fuel_tr[['vehicle','date_dpr','shift_dpr','fuel_consumed']],
    on=['vehicle','date_dpr','shift_dpr'], how='inner'
)
print("\nCorrelation with fuel_consumed:")
for c in DUMP_FEAT_COLS:
    print(f"  {c:28s} : {merged_check[c].corr(merged_check['fuel_consumed']):+.4f}")



def target_encode(train_df, test_df, col, target_col, smoothing=20):
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
    stats['smooth'] = (
        (stats['count'] * stats['mean'] + smoothing * global_mean)
        / (stats['count'] + smoothing)
    )
    return (
        train_df[col].map(stats['smooth']).fillna(global_mean),
        test_df[col].map(stats['smooth']).fillna(global_mean),
    )
shift_tr = shift_tr.merge(
    fuel_tr[['vehicle', 'date_dpr', 'shift_dpr', 'fuel_consumed']],
    on=['vehicle', 'date_dpr', 'shift_dpr'],
    how='inner',
)
print(f"Train rows after joining target: {len(shift_tr):,}")
print(shift_tr['fuel_consumed'].describe())

shift_tr = shift_tr.merge(df_shift_rf, on=['vehicle', 'date_dpr', 'shift_dpr'], how='left')
shift_te = shift_te.merge(df_shift_rf, on=['vehicle', 'date_dpr', 'shift_dpr'], how='left')
shift_tr['fuel_rf'] = shift_tr['fuel_rf'].fillna(0)
shift_te['fuel_rf'] = shift_te['fuel_rf'].fillna(0)


# Why this is the single biggest win:
#   A vehicle's fuel use is very stable across shifts (same route, same engine,
#   similar operator). Knowing last-shift fuel reduces uncertainty massively.
#
# Leakage safety in time-based CV:
#   Val fold is always a later month → lag values come from earlier training rows
#   → no leakage. Handled explicitly in the training loop below.

def add_vehicle_lag_features(shift_tr, shift_te):
    shift_tr = shift_tr.copy().sort_values(['vehicle', 'date_dpr', 'shift_dpr'])
    shift_te = shift_te.copy()

    shift_tr['prev_fuel']     = shift_tr.groupby('vehicle')['fuel_consumed'].shift(1)
    shift_tr['rolling3_fuel'] = (
        shift_tr.groupby('vehicle')['fuel_consumed']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    shift_tr['rolling7_fuel'] = (
        shift_tr.groupby('vehicle')['fuel_consumed']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    shift_tr['rolling3_std'] = (
        shift_tr.groupby('vehicle')['fuel_consumed']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    )

    # Fill NaN (first shift per vehicle) with vehicle mean
    veh_mean = shift_tr.groupby('vehicle')['fuel_consumed'].mean()
    veh_std  = shift_tr.groupby('vehicle')['fuel_consumed'].std()
    for col in ['prev_fuel', 'rolling3_fuel', 'rolling7_fuel']:
        shift_tr[col] = shift_tr[col].fillna(shift_tr['vehicle'].map(veh_mean))
    shift_tr['rolling3_std'] = shift_tr['rolling3_std'].fillna(
        shift_tr['vehicle'].map(veh_std).fillna(shift_tr['fuel_consumed'].std())
    )

    # Test set: use last N training shifts per vehicle
    sorted_tr = shift_tr.sort_values(['vehicle', 'date_dpr', 'shift_dpr'])
    last1 = sorted_tr.groupby('vehicle')['fuel_consumed'].last()
    last3 = sorted_tr.groupby('vehicle')['fuel_consumed'].apply(lambda x: x.tail(3).mean())
    last7 = sorted_tr.groupby('vehicle')['fuel_consumed'].apply(lambda x: x.tail(7).mean())
    std3  = sorted_tr.groupby('vehicle')['fuel_consumed'].apply(lambda x: x.tail(3).std())

    global_mean = shift_tr['fuel_consumed'].mean()
    global_std  = shift_tr['fuel_consumed'].std()

    shift_te['prev_fuel']     = shift_te['vehicle'].map(last1).fillna(global_mean)
    shift_te['rolling3_fuel'] = shift_te['vehicle'].map(last3).fillna(global_mean)
    shift_te['rolling7_fuel'] = shift_te['vehicle'].map(last7).fillna(global_mean)
    shift_te['rolling3_std']  = shift_te['vehicle'].map(std3).fillna(global_std)

    return shift_tr, shift_te

shift_tr, shift_te = add_vehicle_lag_features(shift_tr, shift_te)
lag_corr = shift_tr['prev_fuel'].corr(shift_tr['fuel_consumed'])
print(f"Lag features added. prev_fuel ↔ fuel_consumed correlation: {lag_corr:.3f}")

shift_tr['month'] = pd.to_datetime(shift_tr['date_dpr']).dt.month

vehicle_te_tr, vehicle_te_te = target_encode(shift_tr, shift_te, 'vehicle_enc',  'fuel_consumed')
route_te_tr,   route_te_te   = target_encode(shift_tr, shift_te, 'route_id',     'fuel_consumed')
shift_tr['vehicle_fuel_mean'] = vehicle_te_tr
shift_te['vehicle_fuel_mean'] = vehicle_te_te
shift_tr['route_fuel_mean']   = route_te_tr
shift_te['route_fuel_mean']   = route_te_te

HAS_OPERATOR = 'operator_mode' in shift_tr.columns
if HAS_OPERATOR:
    op_tr, op_te = target_encode(shift_tr, shift_te, 'operator_mode', 'fuel_consumed')
    shift_tr['driver_fuel_mean'] = op_tr
    shift_te['driver_fuel_mean'] = op_te

HAS_HEADING = 'heading_std' in shift_tr.columns

FEATURES = [
    # Kinematics
    'total_dist_km', 'mean_speed', 'max_speed', 'p75_speed', 'std_speed',
    'alt_std', 'alt_range', 'net_lift',
    'gross_elev_gain', 'gross_elev_loss',
    'n_stops', 'pct_stopped', 'stop_density',
    'shift_duration_hr', 'dist_per_hr',
    'n_pings', 'ignition_on_frac',

    # ★ Economy speed (YOUR IDEA)
    'pct_economy_speed', 'pct_overspeed', 'pct_crawling',
    'dist_economy_km', 'dist_overspeed_km', 'dist_crawl_km',
    'economy_ratio', 'overspeed_excess', 'time_out_economy_min',

    # ★ Heading / angle (YOUR IDEA) — only if heading column exists
    'heading_std', 'mean_heading_change',
    'n_sharp_turns', 'n_harsh_turns',
    'sharp_turns_per_km', 'heading_change_per_km',

    # ★ Gradient / climb work
    'pct_uphill', 'pct_downhill',
    'total_climb_work', 'climb_work_per_km',
    'mean_grade_uphill', 'max_grade',

    # Calendar
    'shift_enc', 'dow',

    # Vehicle / fleet
    'vehicle_enc', 'tankcap', 'dump_switch',

    # Route
    'route_id',

    # Refuel in shift
    'fuel_rf',

    # Dump-cycle features
    'n_dumps', 'dump_duration_hr', 'avg_dump_dur_min', 'cv_dump_dur',
    'pct_dumping', 'dumps_per_hr', 'dump_work_score',
    'mean_speed_asym', 'mean_approach_spd', 'mean_depart_spd',
    'avg_cycle_time_min',

    # Target encodings
    'vehicle_fuel_mean', 'route_fuel_mean',

    # ★ Vehicle lag features — BIGGEST WIN
    'prev_fuel', 'rolling3_fuel', 'rolling7_fuel', 'rolling3_std',
]

HAS_OPERATOR = 'operator_mode' in shift_tr.columns

if HAS_OPERATOR:
    # Collect all known operators from training data
    known_operators = set(shift_tr['operator_mode'].dropna().unique())

    # ★ NEW: flag whether this shift's operator was seen in training
    #   For train rows: always 0 (they're all in training by definition)
    #   For test rows : 1 if operator_mode not in known_operators
    shift_tr['is_new_driver'] = 0  # train: always known
    shift_te['is_new_driver'] = (
        ~shift_te['operator_mode']
        .isin(known_operators)
        .fillna(True)      # NaN operator_mode = unknown = treat as new
    ).astype(int)

    new_driver_pct = shift_te['is_new_driver'].mean()
    print(f"New drivers in test: {new_driver_pct:.1%} of test shifts")

    # Smoothing = 50 (was 20) — operator signal is sparser than vehicle/route,
    # so we want to trust the global mean more aggressively for rare drivers.
    op_tr, op_te = target_encode(
        shift_tr, shift_te,
        col='operator_mode', target_col='fuel_consumed',
        smoothing=50,      # ← raised from 20
    )
    shift_tr['driver_fuel_mean'] = op_tr
    shift_te['driver_fuel_mean'] = op_te

# Keep only features present in both train and test
FEATURES = [f for f in FEATURES if f in shift_tr.columns and f in shift_te.columns]
if HAS_OPERATOR:
    FEATURES.append('driver_fuel_mean')
    FEATURES.append('is_new_driver')

X    = shift_tr[FEATURES].copy()
y    = shift_tr['fuel_consumed'].values
X_te = shift_te[FEATURES].copy()

print(f"Feature matrix : {X.shape}")
print(f"Target stats   : mean={y.mean():.1f}  std={y.std():.1f}  max={y.max():.1f}")

lgb_params = dict(
    objective          = 'regression',
    metric             = 'rmse',
    n_estimators       = 5000,        # high ceiling; early stopping decides
    learning_rate      = 0.015,       # was 0.02 → slower = better generalisation
    num_leaves         = 47,          # was 63 → smaller trees = less overfit
    min_child_samples  = 25,          # was 10 → each leaf needs more support
    feature_fraction   = 0.70,        # was 0.8 → more randomness per tree
    bagging_fraction   = 0.75,        # was 0.8
    bagging_freq       = 5,
    reg_alpha          = 0.5,         # was 0.1 → stronger L1
    reg_lambda         = 1.0,         # was 0.1 → stronger L2
    min_split_gain     = 0.01,        # ★ NEW: require real gain before splitting
    random_state       = 42,
    n_jobs             = -1,
    verbose            = -1,
)

fold_configs = [([1], [2]), ([1, 2], [3])]

oof_lgb    = np.zeros(len(X))
test_lgb   = np.zeros(len(X_te))
best_iters = []

for fold_i, (train_months, val_months) in enumerate(fold_configs):
    tr_idx  = shift_tr['month'].isin(train_months).values
    val_idx = shift_tr['month'].isin(val_months).values

    X_tr_f,  y_tr_f  = X[tr_idx].copy(),  y[tr_idx]
    X_val_f, y_val_f = X[val_idx].copy(), y[val_idx]

    # ── Leak-free target encoding inside fold
    fold_df = shift_tr[tr_idx].copy()
    v_tr, v_val = target_encode(fold_df, shift_tr[val_idx], 'vehicle_enc',  'fuel_consumed')
    r_tr, r_val = target_encode(fold_df, shift_tr[val_idx], 'route_id',     'fuel_consumed')
    X_tr_f['vehicle_fuel_mean']  = v_tr.values
    X_tr_f['route_fuel_mean']    = r_tr.values
    X_val_f['vehicle_fuel_mean'] = v_val.values
    X_val_f['route_fuel_mean']   = r_val.values
    if HAS_OPERATOR and 'driver_fuel_mean' in FEATURES:
      # Higher smoothing inside fold too
      op_tr, op_val = target_encode(
          fold_df, shift_tr[val_idx], 'operator_mode', 'fuel_consumed', smoothing=50
      )
      X_tr_f['driver_fuel_mean']  = op_tr.values
      X_val_f['driver_fuel_mean'] = op_val.values

      # is_new_driver for val fold: operators in val but not in fold_df training
      if 'is_new_driver' in FEATURES:
          fold_known_ops = set(fold_df['operator_mode'].dropna().unique())
          val_is_new = (
              ~shift_tr[val_idx]['operator_mode'].isin(fold_known_ops)
          ).astype(int).values
          X_val_f['is_new_driver'] = val_is_new
          X_tr_f['is_new_driver']  = 0   # all train-fold operators are "known"

    # ── Leak-free lag features inside fold
    # Val lag = last value seen in the training portion → no future leakage
    fold_sorted = shift_tr[tr_idx].sort_values(['vehicle', 'date_dpr', 'shift_dpr'])
    fold_last1  = fold_sorted.groupby('vehicle')['fuel_consumed'].last().to_dict()
    fold_last3  = fold_sorted.groupby('vehicle')['fuel_consumed'].apply(lambda x: x.tail(3).mean()).to_dict()
    fold_last7  = fold_sorted.groupby('vehicle')['fuel_consumed'].apply(lambda x: x.tail(7).mean()).to_dict()
    gm = float(y_tr_f.mean())

    val_vehicles = shift_tr[val_idx]['vehicle']
    for col, mapping in [('prev_fuel', fold_last1),
                          ('rolling3_fuel', fold_last3),
                          ('rolling7_fuel', fold_last7)]:
        if col in FEATURES:
            X_val_f[col] = val_vehicles.map(mapping).fillna(gm).values

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr_f, y_tr_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    best_iters.append(model.best_iteration_)

    val_pred = np.clip(model.predict(X_val_f), 0, None)
    oof_lgb[val_idx] = val_pred
    fold_rmse = np.sqrt(mean_squared_error(y_val_f, val_pred))
    print(f"  Fold {fold_i+1} | val={val_months} | RMSE={fold_rmse:.2f} | trees={model.best_iteration_}")

    test_lgb += np.clip(model.predict(X_te), 0, None) / len(fold_configs)

val_mask = shift_tr['month'].isin([2, 3]).values
oof_rmse = np.sqrt(mean_squared_error(y[val_mask], oof_lgb[val_mask]))
print(f"\nOOF RMSE (months 2+3): {oof_rmse:.2f}")
print(f"Best iterations: {best_iters} → mean = {int(np.mean(best_iters))}")


STACK_COLS = ['prev_fuel', 'rolling3_fuel', 'vehicle_fuel_mean']
STACK_COLS = [c for c in STACK_COLS if c in FEATURES]

def build_stack_X(lgb_pred, shift_df):
    parts = [
        lgb_pred.reshape(-1, 1),
        (lgb_pred ** 2).reshape(-1, 1),       # capture non-linear correction
    ]
    parts += [shift_df[c].values.reshape(-1, 1) for c in STACK_COLS]
    return np.hstack(parts)

X_stack_tr = build_stack_X(oof_lgb[val_mask], shift_tr[val_mask])
y_stack_tr = y[val_mask]

ridge = Ridge(alpha=10.0)
ridge.fit(X_stack_tr, y_stack_tr)

oof_stacked_rmse = np.sqrt(mean_squared_error(y_stack_tr, ridge.predict(X_stack_tr)))
print(f"OOF RMSE after Ridge stacking: {oof_stacked_rmse:.2f}")

X_stack_te  = build_stack_X(test_lgb, shift_te)
test_stacked = np.clip(ridge.predict(X_stack_te), 0, None)


mean_best_iter = int(np.mean(best_iters))
final_params   = lgb_params.copy()
final_params['n_estimators'] = mean_best_iter

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=500)])
print(f"Final model trained — {mean_best_iter} trees on {len(X):,} shift-rows.")

# Final test predictions: blend full-data LGB (80%) + stacked correction (20%)
test_preds_lgb   = np.clip(final_model.predict(X_te), 0, None)
test_preds_final = 0.80 * test_preds_lgb + 0.20 * test_stacked
shift_te['fuel_consumed_pred'] = test_preds_final

fi = pd.DataFrame({
    'feature':    FEATURES,
    'importance': final_model.feature_importances_,
}).sort_values('importance', ascending=False)
print("\nTop 20 features:")
print(fi.head(20).to_string(index=False))


shift_te['shift_dpr'] = shift_te['shift_dpr'].astype(str)
shift_te['date_dpr']  = shift_te['date_dpr'].astype(str)

vehicle_train_mean = shift_tr.groupby('vehicle')['fuel_consumed'].mean().to_dict()
global_train_mean  = shift_tr['fuel_consumed'].mean()

submission = (
    id_mapping
    .merge(
        shift_te[['vehicle', 'date_dpr', 'shift_dpr', 'fuel_consumed_pred']],
        left_on  = ['vehicle', 'date', 'shift'],
        right_on = ['vehicle', 'date_dpr', 'shift_dpr'],
        how='left',
    )
    .rename(columns={'fuel_consumed_pred': 'Predicted'})
)[['id', 'vehicle', 'Predicted']]

submission['Predicted'] = submission.apply(
    lambda row: vehicle_train_mean.get(row['vehicle'], global_train_mean)
    if pd.isna(row['Predicted']) else row['Predicted'],
    axis=1,
)
submission = submission[['id', 'Predicted']]
print(f"Submission rows : {len(submission):,}  |  Missing : {submission['Predicted'].isna().sum()}")
print(submission['Predicted'].describe())

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved → /kaggle/working/submission.csv")

