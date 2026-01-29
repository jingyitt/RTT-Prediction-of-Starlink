'''
Start from `main_improved_attention_model()`:
    read RTT CSV → run/cache DGCN features → causally align to RTT with `align_graph_feat_to_rtt(mode="floor")` and trim to the overlapping time range
    → chronological train/val/test split with a `gap` to avoid sliding-window leakage → train, save best, then evaluate.

The data construction is in `RTTDataset.__getitem__`: build `X_feat = [RTT, dX, rolling mean, rolling std, (optional) graph features]`,
    then generate `pos_enc`, `switch_feat`, `far_bucket_feat/far_pos_enc`, and `window_mask`; the target `y_rtt` is the future delta relative to `last_x`.

Model flow in `ImprovedAttentionFusionRTTTransformer.forward`: embedding + positional encoding → add switch features → downsample → bottom Transformer
    → aggregate into window tokens via `window_mask` → concatenate far-bucket tokens → top Transformer
    → fuse and output future deltas.
'''
import os, random
# ---- thread limiting (global) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.preprocessing import RobustScaler
from dgcn.dgcn import infer_graph_features_per_second, align_graph_feat_to_rtt
# Use DGCN
# Dataset
def worker_init_fn(worker_id: int):
    # Set an additional limit within each worker process to prevent each worker from spawning multiple threads
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)

class RTTDataset(Dataset):
    def __init__(self, data, seq_len, num_steps_ahead, timestamps, d_model, switch_indices,
                 k_sec=3.0,  # Local statistical window (seconds)
                 far_context_sec=180.0,  # Multi-scale used below
                 max_far_buckets=12,  # Multi-scale used below: 180s / 15s = 12
                 down_stride=4,
                 max_windows=64,
                 window_size_sec=15.0,
                 sat_feat_full=None, sat_valid_full=None
                ):
        assert d_model % 2 == 0, "d_model must be even for sin/cos positional encoding."
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.num_steps_ahead = num_steps_ahead
        self.length = max(0, len(data) - seq_len - num_steps_ahead + 1)
        if self.length == 0:
            print(f"[WARN] Dataset too small: len(data)={len(data)}, seq_len={seq_len}, ahead={num_steps_ahead}")
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.d_model = d_model
        self.switch_indices = np.asarray(sorted(set(switch_indices)), dtype=np.int64)
        self.down_stride = int(down_stride)
        self.max_windows = int(max_windows)
        self.window_size_sec = float(window_size_sec)
        self.div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model)
        ).astype(np.float32)

        self.k_sec = float(k_sec)
        self.far_context_sec = float(far_context_sec)
        self.max_far_buckets = int(max_far_buckets)

        # pre-calculation dt_global / k_points
        if len(self.timestamps) >= 2:
            dt_global = float(np.median(np.diff(self.timestamps)))
            if dt_global <= 0:
                dt_global = 0.015
        else:
            dt_global = 0.015
        self.dt_global = dt_global

        k_points = int(round(self.k_sec / dt_global))
        k_points = max(5, k_points)
        self.k_points = k_points

        # pre-calculation dX / rolling mean/std
        x_all = self.data.astype(np.float32)
        self.dX_full = np.diff(x_all, prepend=x_all[0]).astype(np.float32)  # global difference

        # pre-calculation prefix sums (used for causal rolling mean/std within a window)
        x = self.data.astype(np.float64)
        self.csum = np.cumsum(np.insert(x, 0, 0.0))
        self.csum2 = np.cumsum(np.insert(x * x, 0, 0.0))

        # pre-calculation far bucket global table
        self.bucket_id_full, self.bucket_feat_table, self.bucket_center_time = self._precompute_bucket_table()

        self.sat_feat_full = None if sat_feat_full is None else np.asarray(sat_feat_full, dtype=np.float32)
        self.sat_valid_full = None if sat_valid_full is None else np.asarray(sat_valid_full, dtype=np.float32)

        self.switch_feat_full = self._precompute_switch_feat_global(sigma_sec=0.8)

        # cache switch times for fast "about-to-switch" label/weight
        if self.switch_indices is not None and len(self.switch_indices) > 0:
            sw = self.switch_indices[(self.switch_indices >= 0) & (self.switch_indices < len(self.timestamps))]
            self.switch_times = self.timestamps[sw].astype(np.float64)
        else:
            self.switch_times = np.zeros((0,), dtype=np.float64)

    def __len__(self):
        return self.length

    def _precompute_switch_feat_global(self, sigma_sec=0.8, eps=1e-6):
        ts = self.timestamps.astype(np.float64)
        rel_t64 = ts - ts[0]
        rel_t = rel_t64.astype(np.float32)

        dX = self.dX_full.astype(np.float32)
        abs_d = np.abs(dX)

        # Bucket by global 15s grid
        bucket_id = np.floor((ts - ts[0]) / self.window_size_sec).astype(np.int64)
        nb = int(bucket_id.max()) + 1

        # each bucket's argmax(|dX|)
        order = np.lexsort((-abs_d, bucket_id))
        b_sorted = bucket_id[order]
        first = np.r_[0, 1 + np.nonzero(b_sorted[1:] != b_sorted[:-1])[0]]
        sw_pos = order[first]

        counts = np.bincount(bucket_id, minlength=nb)
        valid = counts[bucket_id[sw_pos]] > 1
        sw_pos = np.sort(sw_pos[valid])

        if sw_pos.size == 0:
            return np.zeros((len(ts), 4), dtype=np.float32)

        sw_strength = abs_d[sw_pos].astype(np.float32)
        scale = np.percentile(sw_strength, 90).astype(np.float32) + eps
        sw_strength = np.clip(sw_strength / scale, 0.0, 3.0).astype(np.float32)

        # ===== bump: CAUSAL Gaussian accumulation (only past) =====
        dt = float(self.dt_global)
        sigma_pts = max(1, int(round(float(sigma_sec) / dt)))
        half = int(4 * sigma_pts)

        offs = np.arange(-half, half + 1, dtype=np.float32)
        t = offs * np.float32(dt)
        kernel_full = np.exp(-0.5 * (t / np.float32(sigma_sec)) ** 2).astype(np.float32)

        # keep only causal part: offs <= 0
        kernel = kernel_full[offs <= 0].copy()  # length = half+1
        # optional normalize (not required, but keeps scale stable)
        kernel = kernel / (kernel.sum() + 1e-6)

        imp = np.zeros((len(ts),), dtype=np.float32)
        imp[sw_pos] = sw_strength

        # causal convolution: bump[t] = sum_{k<=t} imp[k] * kernel[t-k]
        # easiest: use np.convolve with 'full' then take first N
        bump_full = np.convolve(imp, kernel[::-1], mode="full").astype(np.float32)
        bump = bump_full[:len(ts)]
        bump = np.clip(bump, 0.0, 5.0).astype(np.float32)

        # ===== prev_dist: keep (causal) =====
        sw_t = rel_t[sw_pos].astype(np.float32)
        prev_idx = np.searchsorted(sw_t, rel_t, side="right") - 1

        prev_dist = np.ones((len(ts),), dtype=np.float32)
        has_prev = prev_idx >= 0
        prev_dt = np.zeros((len(ts),), dtype=np.float32)
        prev_dt[has_prev] = rel_t[has_prev] - sw_t[prev_idx[has_prev]]
        prev_dist[has_prev] = np.clip(prev_dt[has_prev] / np.float32(self.window_size_sec), 0.0, 1.0)

        # ===== replace next_dist with bucket_phase (causal) =====
        bucket_phase = ((rel_t64 % self.window_size_sec) / self.window_size_sec).astype(np.float32)
        bucket_phase = np.clip(bucket_phase, 0.0, 1.0)

        # ===== hazard_pre: "about to hit boundary" signal (causal, periodic) =====
        # time_to_next in [0, window_size_sec]
        time_in_bucket = (rel_t64 % self.window_size_sec)
        time_to_next = self.window_size_sec - time_in_bucket  # near 0 => close to boundary
        tau = 1.0  # seconds, you can try 0.5~2.0
        hazard_pre = np.exp(-time_to_next / tau).astype(np.float32)  # close to boundary -> ~1
        hazard_pre = np.clip(hazard_pre, 0.0, 1.0)

        return np.stack([bump, prev_dist, bucket_phase, hazard_pre], axis=-1).astype(np.float32)

    def _precompute_bucket_table(self):
        """
        Global 15s bucket table: one token per bucket [mean, std, slope, sw_flag]
        Buckets are divided by a fixed grid aligned with timestamps[0]: bid = floor((t - t0)/15)
        Vectorized O(N) version: avoids scanning via for b in range(nb) + (bid==b)
        """
        ts = self.timestamps.astype(np.float64)  # 用 float64 计算更稳
        x = self.data.astype(np.float64)

        t0 = ts[0]
        bid = np.floor((ts - t0) / self.window_size_sec).astype(np.int64)  # (N,)
        bid[bid < 0] = 0
        nb = int(bid.max()) + 1

        # mean/std: aggregate all at once using bincount
        counts = np.bincount(bid, minlength=nb).astype(np.float64)  # (nb,)
        sum_x = np.bincount(bid, weights=x, minlength=nb).astype(np.float64)
        sum_x2 = np.bincount(bid, weights=x * x, minlength=nb).astype(np.float64)

        mean = np.zeros(nb, dtype=np.float64)
        var = np.zeros(nb, dtype=np.float64)

        nonempty = counts > 0
        mean[nonempty] = sum_x[nonempty] / counts[nonempty]
        var[nonempty] = sum_x2[nonempty] / counts[nonempty] - mean[nonempty] * mean[nonempty]
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        # slope: only the start/end points of each bucket are required (leveraging the monotonicity of bid).
        slope = np.zeros(nb, dtype=np.float64)
        dur = np.zeros(nb, dtype=np.float64)

        # Finding bucket boundaries: positions where bid changes. bid is monotonic => each bucket is a contiguous segment
        N = len(bid)
        if N > 0:
            change = np.flatnonzero(bid[1:] != bid[:-1]) + 1
            starts = np.r_[0, change]
            ends = np.r_[change, N]

            b_of_seg = bid[starts]  # Each segment's bucket ID.

            x0 = x[starts]
            x1 = x[ends - 1]
            t0_seg = ts[starts]
            t1_seg = ts[ends - 1]

            dur_seg = (t1_seg - t0_seg)
            dur[b_of_seg] = dur_seg

            # dur too small -> slope=0
            ok = dur_seg > 1e-6
            slope_seg = np.zeros_like(dur_seg, dtype=np.float64)
            slope_seg[ok] = (x1[ok] - x0[ok]) / dur_seg[ok]
            slope[b_of_seg] = slope_seg

        # sw_flag: map switch_indices to buckets
        sw_bucket_has = np.zeros(nb, dtype=np.float64)
        if self.switch_indices is not None and len(self.switch_indices) > 0:
            sw = self.switch_indices
            sw = sw[(sw >= 0) & (sw < len(bid))]
            if len(sw) > 0:
                sw_bid = bid[sw]
                sw_bid = sw_bid[(sw_bid >= 0) & (sw_bid < nb)]
                sw_bucket_has[sw_bid] = 1.0

        # feat[mean,std,slope,sw_flag]
        feat = np.zeros((nb, 4), dtype=np.float32)
        feat[:, 0] = mean.astype(np.float32)
        feat[:, 1] = std.astype(np.float32)
        feat[:, 2] = slope.astype(np.float32)
        feat[:, 3] = sw_bucket_has.astype(np.float32)

        center_time = (t0 + (np.arange(nb, dtype=np.float64) + 0.5) * self.window_size_sec).astype(np.float32)

        return bid.astype(np.int64), feat, center_time

    def _get_far_bucket_by_index(self, idx: int):
        """
          far_feat: (max_far_buckets, 4)
          centers:  (max_far_buckets,)
          t_start:  for far_pos_enc's base_time（t_end - far_context_sec）
        Take the previous max_far_buckets relative to the current point's bucket """
        cur_bid = int(self.bucket_id_full[idx])
        end_bid = cur_bid  # excluding the current bucket
        start_bid = end_bid - self.max_far_buckets

        far_feat = np.zeros((self.max_far_buckets, 4), dtype=np.float32)

        # Provide an "ideal center" for missing sections to ensure the pos_enc (positional encoding) starts from 0 relative to t_start
        t_end = float(self.timestamps[idx])
        t_start = t_end - self.far_context_sec
        centers = t_start + (np.arange(self.max_far_buckets, dtype=np.float32) + 0.5) * self.window_size_sec

        # Intersection of valid bucket ranges.
        b0 = max(start_bid, 0)
        b1 = min(end_bid, self.bucket_feat_table.shape[0])
        if b1 <= b0:
            return far_feat, centers.astype(np.float32), np.float32(t_start)

        # Copy the bucket table to the right side of far_feat (maintaining the "closer to current is further to the right" order).
        # Example: max=12, start_bid=-3, end_bid=9 -> The valid range is [0, 9), which is placed in the last 9 slots of far_feat.
        dst_start = b0 - start_bid
        dst_end = dst_start + (b1 - b0)
        far_feat[dst_start:dst_end] = self.bucket_feat_table[b0:b1]

        # The center time for pos_enc also corresponds to the actual bucket center (not the ideal center).
        centers[dst_start:dst_end] = self.bucket_center_time[b0:b1]

        return far_feat.astype(np.float32), centers.astype(np.float32), np.float32(t_start)

    def _rolling_mean_std_window_causal(self, start_idx: int, L: int, k: int):
        """
        Within a sample window [start_idx,start_idx+L), perform "intra-window causal rolling mean/std",
        without using any history from outside the window.

        For i-th point in the window(0..L-1):
          mean[i] = mean(x[start_idx + max(0, i-k+1) : start_idx+i+1])
        Return:
          mean_: (L,)
          std_:  (L,)
        """
        # Local position 0..L-1
        pos = np.arange(L, dtype=np.int64)

        # Intra-window causal left boundary (local)
        l_local = np.maximum(0, pos - k + 1)
        r_local = pos + 1

        # Map to the global prefix-sum indices
        l = start_idx + l_local
        r = start_idx + r_local

        cnt = (r - l).astype(np.float32)

        sum_ = self.csum[r] - self.csum[l]
        sum2_ = self.csum2[r] - self.csum2[l]

        mean_ = sum_ / cnt
        var_ = np.maximum(sum2_ / cnt - mean_ * mean_, 0.0)
        std_ = np.sqrt(var_).astype(np.float32)

        return mean_.astype(np.float32), std_.astype(np.float32)

    def __getitem__(self, idx):
        X_rtt = self.data[idx:idx + self.seq_len]  # (L,)

        # Directly slice the global difference (maintaining the difference at the window's first point =0)
        dX = self.dX_full[idx:idx + self.seq_len].copy()
        dX[0] = 0.0

        # rolling mean/std
        mean_, std_ = self._rolling_mean_std_window_causal(
            start_idx=idx,
            L=self.seq_len,
            k=self.k_points
        )

        # multi-channel input (L,4)
        X_feat = np.stack([X_rtt, dX, mean_, std_], axis=-1).astype(np.float32)

        # Concatenate exogenous features from the graph model (aligned to each RTT point)
        if self.sat_feat_full is not None:
            ext = self.sat_feat_full[idx:idx + self.seq_len]  # (L, Fg)
            if ext.shape[0] == self.seq_len:
                if self.sat_valid_full is not None:
                    v = self.sat_valid_full[idx:idx + self.seq_len]  # (L,1) float32 {0,1}
                    if v.shape[0] == self.seq_len:
                        # zero-out invalid exogenous features
                        ext = ext.copy()
                        ext[v[:, 0] < 0.5] = 0.0
                        # append validity flag so "all-zero" is not ambiguous
                        X_feat = np.concatenate([X_feat, ext, v], axis=-1).astype(np.float32)
                    else:
                        X_feat = np.concatenate([X_feat, ext], axis=-1).astype(np.float32)
                else:
                    X_feat = np.concatenate([X_feat, ext], axis=-1).astype(np.float32)

        last = self.data[idx + self.seq_len - 1]  # last point in X window
        y_abs = self.data[idx + self.seq_len:idx + self.seq_len + self.num_steps_ahead]
        y_rtt = (y_abs - last).astype(np.float32)

        timestamps_x = self.timestamps[idx: idx + self.seq_len]
        timestamps_y = self.timestamps[idx + self.seq_len: idx + self.seq_len + self.num_steps_ahead]

        start_idx = idx

        # position embedding
        pos_enc = window_positional_encoding_from_time(
            timestamps_x, self.d_model, div_term=self.div_term, window_size_sec=self.window_size_sec
        )

        switch_feat = self.switch_feat_full[idx: idx + self.seq_len]

        anchor = idx + self.seq_len - 1
        far_bucket_feat, centers, t_start = self._get_far_bucket_by_index(anchor)

        far_pos_enc = window_positional_encoding_from_time(
            centers, self.d_model, div_term=self.div_term,
            window_size_sec=self.window_size_sec,
            base_time=t_start
        )

        # generate window_mask (index space after downsampling)
        windows = build_window_indices_snap(
            timestamps_x=timestamps_x,
            start_idx=idx,
            seq_len=self.seq_len,
            switch_indices=self.switch_indices,
            window_size_sec=self.window_size_sec
        )

        s = self.down_stride
        L = self.seq_len
        L2 = (L - s) // s + 1  # floor pool output length (AvgPool1d)

        scaled = []
        for (a, b) in windows:
            # (a,b) is [a,b) in [0,L]
            if b <= a:
                continue
            aa = a // s
            bb = ((b - 1) // s) + 1  # match floor pooling
            aa = max(0, min(aa, L2))
            bb = max(0, min(bb, L2))
            if bb > aa:
                scaled.append((aa, bb))

        # Truncate to max_windows
        if self.max_windows is not None and len(scaled) > self.max_windows:
            new_boundaries = np.linspace(0, L2, num=self.max_windows + 1).round().astype(int)
            new_boundaries[0] = 0
            new_boundaries[-1] = L2
            new_boundaries = np.unique(new_boundaries)
            scaled = [(int(new_boundaries[i]), int(new_boundaries[i + 1]))
                      for i in range(len(new_boundaries) - 1)
                      if new_boundaries[i + 1] > new_boundaries[i]]

        # convert to mask: (max_windows, L2)
        W = len(scaled)
        Wpad = self.max_windows if self.max_windows is not None else W
        window_mask = np.zeros((Wpad, L2), dtype=np.float32)
        for i, (aa, bb) in enumerate(scaled[:Wpad]):
            window_mask[i, aa:bb] = 1.0


        # ===== auxiliary label: will a switch happen soon after anchor? =====
        # anchor time
        t_anchor = float(self.timestamps[anchor])
        horizon_sec = 1.0  # "soon" window, try different value
        t_limit = t_anchor + horizon_sec

        if self.switch_times.size > 0:
            j = np.searchsorted(self.switch_times, t_anchor, side="right")
            will_switch = (j < self.switch_times.size) and (self.switch_times[j] <= t_limit)
            y_sw = np.float32(1.0 if will_switch else 0.0)

            # weight: emphasize samples near upcoming switch
            # smaller dt => larger weight
            if j < self.switch_times.size:
                dt_next = float(self.switch_times[j] - t_anchor)
            else:
                dt_next = 1e9
            tau_w = 0.5  # seconds
            gamma = 2.0  # weight strength, try different value
            w = np.float32(1.0 + gamma * np.exp(-dt_next / tau_w))
        else:
            y_sw = np.float32(0.0)
            w = np.float32(1.0)

        return (X_feat,
                pos_enc.astype(np.float32),
                switch_feat.astype(np.float32),
                far_bucket_feat.astype(np.float32),
                far_pos_enc.astype(np.float32),
                window_mask.astype(np.float32),
                y_rtt.astype(np.float32),
                y_sw,
                w,
                timestamps_y)

def window_positional_encoding_from_time(timestamps_x, d_model, div_term, window_size_sec=15.0, base_time=None):
    ts = np.asarray(timestamps_x, dtype=np.float64)
    if base_time is None:
        base_time = float(ts[0])
    else:
        base_time = float(base_time)

    rel_t64 = ts - base_time          # float64 differencing, precision-safe.
    rel_t = rel_t64.astype(np.float32)

    L = len(rel_t)
    pe = np.zeros((L, d_model), dtype=np.float32)

    position = rel_t.reshape(L, 1)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    phase = (rel_t64 % window_size_sec) / window_size_sec * (2.0 * np.pi)
    if d_model >= 4:
        pe[:, 0] += np.sin(phase).astype(np.float32)
        pe[:, 1] += np.cos(phase).astype(np.float32)
    return pe

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # MAPE
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    mape = np.mean(
        np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    ) * 100

    # sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    smape = np.mean(diff) * 100

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # MSE
    mse = mean_squared_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mse)

    # RMSLE
    y_true_log = y_true.copy()
    y_pred_log = y_pred.copy()
    y_true_log[y_true_log < 0] = 0
    y_pred_log[y_pred_log < 0] = 0
    rmsle = np.sqrt(mean_squared_log_error(y_true_log + 1, y_pred_log + 1))

    return mape, smape, mse, rmse, mae, rmsle

# Model
class ImprovedAttentionFusionRTTTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_steps_ahead,
                 dropout=0.1, down_stride=4):
        super().__init__()
        self.d_model = d_model
        self.down_stride = int(down_stride)

        self.embedding = nn.Linear(input_dim, d_model)

        # switch flag -> d_model + learnable intensity
        self.switch_proj = nn.Linear(4, d_model, bias=False)
        self.switch_alpha = nn.Parameter(torch.tensor(0.0))  # initial 0, let the model learn use or not

        # downsampling (L -> L/stride)
        if self.down_stride > 1:
            self.down_pool = nn.AvgPool1d(kernel_size=self.down_stride, stride=self.down_stride)

        bottom_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        top_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )

        self.bottom_transformer = nn.TransformerEncoder(bottom_encoder_layer, num_layers=num_layers)
        self.top_transformer = nn.TransformerEncoder(top_encoder_layer, num_layers=num_layers)

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.fc_out = nn.Linear(d_model, num_steps_ahead)
        self.far_embedding = nn.Linear(4, d_model)  # embedding of far bucket token
        self.out_norm = nn.LayerNorm(d_model)
        self.fc_sw = nn.Linear(d_model, 1)  # switch soon?

    def forward(self, src, pos_encoding, switch_flag, window_mask,
                far_bucket_feat=None, far_pos_enc=None):
        """ src: (B, L, C)
        pos_encoding: (B, L, D)
        switch_flag: (B, L, 4)
        window_indices: list[(s,e)] """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src + pos_encoding.to(src.device)

        sw = self.switch_proj(switch_flag.to(src.device))
        src = src + torch.tanh(self.switch_alpha) * sw

        # downsampling
        if self.down_stride > 1:
            # src: (B,L,D) -> (B,D,L) -> pool -> (B,L',D)
            src = self.down_pool(src.transpose(1, 2)).transpose(1, 2)
        L2_real = src.shape[1]
        assert window_mask.shape[-1] == L2_real, f"mask L2={window_mask.shape[-1]} != pooled L2={L2_real}"

        bottom_output = self.bottom_transformer(src)  # (B, L, D)

        # window_mask: (B, W, L2)  float32 {0,1}
        wm = window_mask.to(bottom_output.device)  # (B,W,L2)

        # numerator: (B,W,D) = sum_{l} wm[b,w,l] * bottom_output[b,l,:]
        # counts:    (B,W,1) = sum_{l} wm[b,w,l]
        numerator = torch.einsum("bwl,bld->bwd", wm, bottom_output)
        counts = wm.sum(dim=-1, keepdim=True).clamp_min(1.0)
        top_input_cur = numerator / counts  # (B,W,D)

        # ===== far bucket tokens =====
        if far_bucket_feat is not None:
            fb = far_bucket_feat.to(src.device)  # (B, Bk, 4)
            fb = self.far_embedding(fb)  # (B, Bk, D)
            if far_pos_enc is not None:
                fb = fb + far_pos_enc.to(src.device)  # (B, Bk, D)
            top_input = torch.cat([fb, top_input_cur], dim=1)  # (B, Bk+W, D)
        else:
            top_input = top_input_cur  # (B, W, D)

        top_output = self.top_transformer(top_input)  # (B, T, D)

        # Use a short recent window as query to preserve intra-bucket small fluctuations
        k = min(16, bottom_output.shape[1])  # try 8/16/32
        bottom_last = bottom_output[:, -k:, :].mean(dim=1, keepdim=True)  # (B,1,D)

        # bottom_last do attention for all top tokens, not only top_last
        attn_output, _ = self.attention(bottom_last, top_output, top_output)  # (B,1,D)
        fused_feature = attn_output.squeeze(1)  # (B, D)

        fused_feature = self.out_norm(fused_feature)
        rtt_pred = self.fc_out(fused_feature)
        sw_logit = self.fc_sw(fused_feature).squeeze(-1)  # (B,)
        return rtt_pred, sw_logit

# handover point detection
def detect_switch_points(df, window_size_sec=15.0, min_points_per_bucket=2):
    """
    Select the position with the maximum |dX| within each 15s bucket as the switch point (consistent with switch_feat)
    - Uses RTT difference: dX = RTT[i] - RTT[i-1]
    - If the number of points in a bucket is < min_points_per_bucket, skip it (consistent with counts>1 in switch_feat)
    Returns: List of global indices (0..N-1 after df.reset_index())
    """
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    x = df["RTT"].to_numpy(dtype=np.float64)
    N = len(x)
    if N <= 1:
        return []

    # Relative time bucketing
    rel_t = ts - ts[0]
    bucket_id = np.floor(rel_t / float(window_size_sec)).astype(np.int64)
    bucket_id = np.maximum(bucket_id, 0)
    nb = int(bucket_id.max()) + 1

    # dX & |dX|
    dX = np.diff(x, prepend=x[0])  # dX[0]=0
    abs_d = np.abs(dX)

    # Get argmax(|dX|) for each bucket (vectorized implementation, consistent with switch_feat)
    order = np.lexsort((-abs_d, bucket_id))  # bucket ascending, abs_d descending
    b_sorted = bucket_id[order]
    first = np.r_[0, 1 + np.nonzero(b_sorted[1:] != b_sorted[:-1])[0]]
    sw_pos = order[first]  # Candidate points for each bucket (global indices)

    # Filtering: Discard if the number of points in the bucket is too small (avoid single-point buckets)
    counts = np.bincount(bucket_id, minlength=nb)
    valid = counts[bucket_id[sw_pos]] >= int(min_points_per_bucket)
    sw_pos = sw_pos[valid]

    # Sorted output
    sw_pos = np.sort(sw_pos).astype(int).tolist()
    return sw_pos


def build_window_indices_snap(timestamps_x, start_idx, seq_len, switch_indices, window_size_sec=15.0):
    """Snap (alignment) strategy:
    - First, divide the X sequence into several 15s buckets based on timestamps_x.
    - If a switch point exists within a bucket (mapped from global switch_indices to local),
      use the "position of the switch point in the bucket" as the bucket boundary; otherwise, use the "time boundary of the bucket".
    - Final boundaries = [0] + (one boundary point generated per bucket) + [seq_len]
    Returns: window_indices: List[(s,e)]"""
    # timestamps_x: torch.Tensor shape (L,) or numpy shape (L,)
    if isinstance(timestamps_x, torch.Tensor):
        ts = timestamps_x.detach().cpu().numpy()
    else:
        ts = np.asarray(timestamps_x)

    # 1) Calculate 15s buckets within each sample (local indices)
    t0 = ts[0]
    rel = ts - t0
    bucket_id = np.floor(rel / window_size_sec).astype(np.int64)

    # Find start and end indices for each bucket (local)
    # e.g., bucket_id = [0,0,0,1,1,2,2,2] -> buckets: (0,3),(3,5),(5,8)
    change = np.where(bucket_id[1:] != bucket_id[:-1])[0] + 1
    bucket_starts = np.concatenate(([0], change))
    bucket_ends = np.concatenate((change, [len(ts)]))

    # 2) Get switch points falling within this sample window (global -> local)
    # Those global switch points in [start_idx, start_idx + seq_len)
    # switch_indices must be sorted
    left = np.searchsorted(switch_indices, start_idx, side="left")
    right = np.searchsorted(switch_indices, start_idx + seq_len, side="left")
    sw_in_window_global = switch_indices[left:right]

    # Map to local indices and store in a set for efficient lookup
    sw_local_set = set()
    for s in sw_in_window_global:
        loc = int(s - start_idx)
        if 0 <= loc < seq_len:
            sw_local_set.add(loc)

    # 3) Perform snap for each bucket: use switch point if present, otherwise use bucket end boundary
    boundaries = [0]
    for s, e in zip(bucket_starts, bucket_ends):
        if e <= s:
            continue

        # Switch points in bucket: strictly within (s, e)
        candidates = [p for p in sw_local_set if s < p < e]

        if len(candidates) == 0:
            # ★ No switch point: cut by bucket time boundary
            snap_point = e
        else:
            # Switch point present: use the switch point closest to the end of the bucket,
            # and add +1 after it (to ensure non-empty windows)
            snap_point = min(max(candidates) + 1, e)

        boundaries.append(snap_point)

    boundaries.append(seq_len)

    # 4) Deduplicate + sort + generate window_indices
    boundaries = sorted(set([b for b in boundaries if 0 <= b <= seq_len]))
    window_indices = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
                      if boundaries[i + 1] > boundaries[i]]

    max_windows = 64
    if max_windows is not None and len(window_indices) > max_windows:
        new_boundaries = np.linspace(0, seq_len, num=max_windows + 1).round().astype(int)
        new_boundaries[0] = 0
        new_boundaries[-1] = seq_len
        new_boundaries = np.unique(new_boundaries)
        window_indices = [(int(new_boundaries[i]), int(new_boundaries[i + 1]))
                          for i in range(len(new_boundaries) - 1)
                          if new_boundaries[i + 1] > new_boundaries[i]]

    return window_indices

# ---- graph feature cache (global) ----
_GRAPH_CACHE = {}

def get_cached_graph_features(graph_ckpt_path: str, graphs_pt_path: str, device: str):
    """
    Cache graph inference results so we run infer_graph_features_per_second() only once
    for the same (ckpt, graphs.pt, device).
    Returns:
      per_sec_feat: np.ndarray (T, F)
      graph_times_1s: np.ndarray (T,)
    """
    if (not os.path.exists(graph_ckpt_path)) or (not os.path.exists(graphs_pt_path)):
        return None, None

    key = (os.path.abspath(graph_ckpt_path), os.path.abspath(graphs_pt_path), device)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    with torch.no_grad():
        per_sec_feat, graphs_1s, graph_times_1s, _ = infer_graph_features_per_second(
            graph_ckpt_path, graphs_pt_path, device=device
        )

    try:
        del graphs_1s
    except Exception:
        pass

    if torch.is_tensor(per_sec_feat):
        per_sec_feat = per_sec_feat.detach().cpu().numpy()
    graph_times_1s = np.asarray(graph_times_1s, dtype=np.float64)

    _GRAPH_CACHE[key] = (per_sec_feat, graph_times_1s)
    return per_sec_feat, graph_times_1s


# main
def main_improved_attention_model(filenames, seq_len, num_steps_ahead, d_model):
    results_dir = 'results_attention'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Graph model inference and align to RTT
    graph_ckpt_path = "./dgcn/dgcn_two_stage_union_soft_K1.pt"
    graphs_pt_path = "./dgcn/graphs.pt"
    g_device = "cuda" if torch.cuda.is_available() else "cpu"

    use_graph = (os.path.exists(graph_ckpt_path) and os.path.exists(graphs_pt_path))
    print(
        f"[GRAPH] use_graph={use_graph}, ckpt_ok={os.path.exists(graph_ckpt_path)}, graphs_ok={os.path.exists(graphs_pt_path)}")

    per_sec_feat = None
    graph_times_1s = None
    gt = None
    gt_min = None
    gt_max = None
    if use_graph:
        per_sec_feat, graph_times_1s = get_cached_graph_features(graph_ckpt_path, graphs_pt_path, g_device)
        gt = np.asarray(graph_times_1s, dtype=np.float64)
        gt_min, gt_max = float(gt.min()), float(gt.max())

    for filename in filenames:
        # metrics file
        base_filename = os.path.basename(filename).replace('.csv', '')
        metrics_file_path = os.path.join(results_dir, f'attention_metrics_{base_filename}_{num_steps_ahead}.txt')
        dir_path = os.path.dirname(metrics_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(f"Processing file: {filename}")
        data = pd.read_csv(filename).reset_index(drop=True)
        data = data.sort_values("timestamp").reset_index(drop=True)
        timestamps = data['timestamp'].values
        rtt_values = data['RTT'].values

        ts = timestamps.astype(np.float64)

        # default: RTT-only, keep all
        mask = np.ones(len(ts), dtype=bool)
        sat_feat_full = None
        sat_valid_full = None

        if use_graph:
            aligned_feat, valid_mask = align_graph_feat_to_rtt(
                timestamps_rtt=timestamps,
                graph_times=graph_times_1s,
                per_sec_feat=per_sec_feat,
                mode="floor",
                observed_col=3,
                service_valid_col=4,
                min_valid=0.5
            )
            sat_feat_full = aligned_feat[:, 0:3].astype(np.float32)
            sat_valid_full = valid_mask.astype(np.float32).reshape(-1, 1)

            mask = (ts >= gt_min) & (ts <= gt_max)

            keep = int(mask.sum())
            drop = int(len(ts) - keep)
            print(f"[ALIGN] trim RTT to graph overlap: gt_min={gt_min:.3f}, gt_max={gt_max:.3f}, "
                  f"keep={keep}, drop={drop}")
        else:
            print("[ALIGN] graph disabled -> no overlap trimming, use RTT only.")

        # ---- apply mask ONCE ----
        timestamps = timestamps[mask]
        rtt_values = rtt_values[mask]
        data = data.iloc[mask].reset_index(drop=True)

        if sat_feat_full is not None:
            sat_feat_full = sat_feat_full[mask]
            sat_valid_full = sat_valid_full[mask]

        # ---- recompute ts after trimming ----
        ts = timestamps.astype(np.float64)

        # extra alignment checks only when graph enabled
        if use_graph:
            oob_left = int((ts < gt_min).sum())
            oob_right = int((ts > gt_max).sum())
            print(f"[CHECK] OOB RTT points vs graph_times: left={oob_left}, right={oob_right}")

            idx_map = np.searchsorted(gt, ts, side="right") - 1
            idx_map = np.clip(idx_map, 0, len(gt) - 1)
            for p in [0, len(ts) // 2, len(ts) - 1]:
                print(f"[CHECK] sample p={p}: ts={ts[p]:.6f} -> gt[{idx_map[p]}]={gt[idx_map[p]]:.6f}, "
                      f"delta={(ts[p] - gt[idx_map[p]]):.3f}s, feat={sat_feat_full[p]}")
            print()
        # uodate ts
        ts = timestamps.astype(np.float64)

        print("\n[CHECK] RTT timestamps range:",
              f"min={ts.min():.6f}, max={ts.max():.6f}, N={len(ts)}")

        dt = np.diff(ts)
        print("[CHECK] RTT dt: min={:.6f} ms, median={:.6f} ms, neg_count={}".format(
            dt.min() * 1000.0, np.median(dt) * 1000.0, int((dt < 0).sum())
        ))
        if use_graph:
            oob_left = int((ts < gt_min).sum())
            oob_right = int((ts > gt_max).sum())
            print(f"[CHECK] OOB RTT points vs graph_times: left={oob_left}, right={oob_right}")

            # see examples of aligned indices: First / Middle / Last
            idx = np.searchsorted(gt, ts, side="right") - 1
            idx = np.clip(idx, 0, len(gt) - 1)

            for p in [0, len(ts) // 2, len(ts) - 1]:
                print(f"[CHECK] sample p={p}: ts={ts[p]:.6f} -> gt[{idx[p]}]={gt[idx[p]]:.6f}, "
                      f"delta={(ts[p] - gt[idx[p]]):.3f}s, feat={sat_feat_full[p]}")
            print()

        # Split by chronological order to avoid sliding window leakage
        N = len(rtt_values)
        train_ratio, val_ratio = 0.8, 0.1
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))

        # gap, avoid any sliding window straddling the set boundaries
        gap = seq_len + num_steps_ahead

        train_range = (0, train_end)
        val_range = (train_end + gap, val_end)
        test_range = (val_end + gap, N)

        def make_start_indices(rng, seq_len, num_steps_ahead):
            a, b = rng
            max_start = b - seq_len - num_steps_ahead
            if max_start < a:
                return []
            return list(range(a, max_start + 1))

        train_ids = make_start_indices(train_range, seq_len, num_steps_ahead)
        val_ids = make_start_indices(val_range, seq_len, num_steps_ahead)
        test_ids = make_start_indices(test_range, seq_len, num_steps_ahead)

        if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:
            print("[WARN] Not enough data after time split + gap. "
                  f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}. "
                  "Try smaller seq_len or remove/ reduce gap.")
            continue

        # scaler is fitted exclusively on the training partition to prevent temporal data leakage
        scaler = RobustScaler()
        scaler.fit(rtt_values[train_range[0]:train_range[1]].reshape(-1, 1))
        rtt_values_norm = scaler.transform(rtt_values.reshape(-1, 1)).flatten().astype(np.float32)

        df = pd.DataFrame({'timestamp': timestamps, 'RTT': rtt_values})
        switch_indices = np.asarray(detect_switch_points(df, window_size_sec=15.0, min_points_per_bucket=2),
                                    dtype=np.int64)

        switch_indices = np.asarray(sorted(set(switch_indices.tolist())), dtype=np.int64)

        # construct dataset
        dataset = RTTDataset(
            data=rtt_values_norm,
            seq_len=seq_len,
            num_steps_ahead=num_steps_ahead,
            timestamps=timestamps,
            d_model=d_model,
            switch_indices=switch_indices,
            sat_feat_full=sat_feat_full,
            sat_valid_full=sat_valid_full,
            k_sec=3.0,
            far_context_sec=180.0,
            max_far_buckets=12
        )

        # input_dim（4 + graph generated dim）
        sample0 = dataset[train_ids[0]]
        input_dim = sample0[0].shape[-1]  # sample0[0] = X_feat
        print("[INFO] Transformer input_dim =", input_dim)

        train_dataset = torch.utils.data.Subset(dataset, train_ids)
        val_dataset = torch.utils.data.Subset(dataset, val_ids)
        test_dataset = torch.utils.data.Subset(dataset, test_ids)

        print(f"Total raw length: {N}")
        print(f"Split ranges: train={train_range}, val={val_range}, test={test_range}, gap={gap}")
        print(f"Windows: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        batch_size = 64

        use_gpu = torch.cuda.is_available()
        num_workers = 4  # 4 cpu

        dl_kwargs = {}
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=use_gpu,
            persistent_workers=(num_workers > 0),
            worker_init_fn=worker_init_fn,
            **dl_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_gpu,
            persistent_workers=(num_workers > 0),
            worker_init_fn=worker_init_fn,
            **dl_kwargs
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_gpu,
            persistent_workers=(num_workers > 0),
            worker_init_fn=worker_init_fn,
            **dl_kwargs
        )

        # Initial
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImprovedAttentionFusionRTTTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=8,
            num_layers=2,
            num_steps_ahead=num_steps_ahead,
            dropout=0.1,
            down_stride=4,
        )

        if torch.cuda.device_count() > 1 and batch_size > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training!")
            model = torch.nn.DataParallel(model)

        model = model.to(device)

        # criterion = nn.HuberLoss(delta=1.0)
        # criterion = nn.SmoothL1Loss(beta=1.0)
        criterion_rtt = nn.HuberLoss(delta=1.0, reduction="none")
        criterion_sw = nn.BCEWithLogitsLoss(reduction="mean")
        lambda_sw = 0.2
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=1e-6
        )

        base_filename = os.path.basename(filename).replace('.csv', '')
        model_dir = 'models_attention'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_save_path = os.path.join(model_dir, f"attention_{base_filename}_{num_steps_ahead}.pth")

        num_epochs = 100
        accum_steps = 1

        early_stop_patience = 10
        no_improve = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            train_loss_sum = torch.zeros((), device=device)
            train_loss_n = 0
            optimizer.zero_grad(set_to_none=True)

            # ===== Train =====
            for step, (X_batch, pos_batch, sw_batch, far_bucket_batch, far_pos_batch,
                       window_mask_batch, y_rtt_batch, y_sw_batch, w_batch, timestamps_y_batch) in enumerate(
                train_loader):

                X_batch = X_batch.to(device, non_blocking=True)
                pos_batch = pos_batch.to(device, non_blocking=True)
                sw_batch = sw_batch.to(device, non_blocking=True)
                far_bucket_batch = far_bucket_batch.to(device, non_blocking=True)
                far_pos_batch = far_pos_batch.to(device, non_blocking=True)
                window_mask_batch = window_mask_batch.to(device, non_blocking=True)
                y_rtt_batch = y_rtt_batch.to(device, non_blocking=True)
                y_sw_batch = y_sw_batch.to(device, non_blocking=True)
                w_batch = w_batch.to(device, non_blocking=True)

                outputs_rtt, sw_logit = model(
                    X_batch, pos_batch, sw_batch, window_mask_batch,
                    far_bucket_feat=far_bucket_batch,
                    far_pos_enc=far_pos_batch
                )

                # RTT weighted huber
                per_elem = criterion_rtt(outputs_rtt, y_rtt_batch)  # (B, ahead)
                per_sample = per_elem.mean(dim=-1)  # (B,)
                loss_rtt = (per_sample * w_batch).mean()

                # auxiliary switch classification
                loss_sw = criterion_sw(sw_logit, y_sw_batch)

                loss = loss_rtt + lambda_sw * loss_sw

                train_loss_sum += loss.detach()
                train_loss_n += 1

                (loss / accum_steps).backward()

                do_step = ((step + 1) % accum_steps == 0)
                is_last = (step + 1 == len(train_loader))
                if do_step or is_last:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            avg_train_loss = (train_loss_sum / max(1, train_loss_n)).item()  # epoch 末只同步一次
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}")

            # ===== Val =====
            model.eval()
            val_loss_sum = torch.zeros((), device=device)
            val_loss_n = 0
            with torch.no_grad():
                for (X_batch, pos_batch, sw_batch, far_bucket_batch, far_pos_batch,
                     window_mask_batch, y_rtt_batch, y_sw_batch, w_batch, timestamps_y_batch) in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    pos_batch = pos_batch.to(device, non_blocking=True)
                    sw_batch = sw_batch.to(device, non_blocking=True)
                    far_bucket_batch = far_bucket_batch.to(device, non_blocking=True)
                    far_pos_batch = far_pos_batch.to(device, non_blocking=True)
                    window_mask_batch = window_mask_batch.to(device, non_blocking=True)
                    y_rtt_batch = y_rtt_batch.to(device, non_blocking=True)

                    outputs_rtt, sw_logit = model(
                        X_batch, pos_batch, sw_batch, window_mask_batch,
                        far_bucket_feat=far_bucket_batch,
                        far_pos_enc=far_pos_batch
                    )
                    per_elem = criterion_rtt(outputs_rtt, y_rtt_batch)
                    loss = per_elem.mean()
                    val_loss_sum += loss.detach()
                    val_loss_n += 1
                avg_val_loss = (val_loss_sum / max(1, val_loss_n)).item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.6f}")

            # update learning rate
            scheduler.step(avg_val_loss)

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}/{num_epochs}, LR: {cur_lr:.8f}")

            # save best
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                no_improve = 0
                try:
                    torch.save(get_state_dict(model), model_save_path)
                except Exception as e:
                    print(f"[WARN] Failed to save model: {e}")
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"[EarlyStop] No improvement for {early_stop_patience} epochs. Stop at epoch {epoch + 1}.")
                    break

        if os.path.exists(model_save_path):
            sd = torch.load(model_save_path, map_location=device)
            load_state_dict(model, sd)
        else:
            print(f"Model file not found: {model_save_path}")
            torch.save(get_state_dict(model), model_save_path)
            print(f"Current model saved to: {model_save_path}")

        model.eval()

        # ===== Test =====
        model.eval()
        y_true_rtt, y_pred_rtt, collect_timestamps = [], [], []
        start_time = time.time()

        with torch.no_grad():
            for (X_batch, pos_batch, sw_batch, far_bucket_batch, far_pos_batch,
                 window_mask_batch, y_rtt_batch, y_sw_batch, w_batch, timestamps_y_batch) in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                pos_batch = pos_batch.to(device, non_blocking=True)
                sw_batch = sw_batch.to(device, non_blocking=True)
                far_bucket_batch = far_bucket_batch.to(device, non_blocking=True)
                far_pos_batch = far_pos_batch.to(device, non_blocking=True)
                window_mask_batch = window_mask_batch.to(device, non_blocking=True)
                y_rtt_batch = y_rtt_batch.to(device, non_blocking=True)

                outputs_rtt, sw_logit = model(
                    X_batch, pos_batch, sw_batch, window_mask_batch,
                    far_bucket_feat=far_bucket_batch,
                    far_pos_enc=far_pos_batch
                )

                last_x = X_batch[:, -1, 0]  # X_rtt(norm)
                outputs_abs = outputs_rtt + last_x.unsqueeze(-1)
                y_true_abs = y_rtt_batch + last_x.unsqueeze(-1)

                y_true_rtt.append(y_true_abs.detach().cpu().numpy())
                y_pred_rtt.append(outputs_abs.detach().cpu().numpy())

                if torch.is_tensor(timestamps_y_batch):
                    collect_timestamps.append(timestamps_y_batch.detach().cpu().numpy())
                else:
                    collect_timestamps.append(np.asarray(timestamps_y_batch))

        test_time = time.time() - start_time

        y_true_rtt = np.concatenate(y_true_rtt, axis=0)  # (Nwin,ahead)
        y_pred_rtt = np.concatenate(y_pred_rtt, axis=0)  # (Nwin,ahead)
        collect_timestamps = np.concatenate(collect_timestamps, axis=0)  # (Nwin,ahead)

        flat_timestamps = collect_timestamps.reshape(-1)
        y_true_rtt = y_true_rtt.reshape(-1)
        y_pred_rtt = y_pred_rtt.reshape(-1)

        # Inverse transform the prediction results
        y_true_rtt = scaler.inverse_transform(y_true_rtt.reshape(-1, 1)).flatten()
        y_pred_rtt = scaler.inverse_transform(y_pred_rtt.reshape(-1, 1)).flatten()

        # metrics
        mape, smape, mse, rmse, mae, rmsle = calculate_metrics(y_true_rtt, y_pred_rtt)
        print(f"Final Results for {filename}:")
        print(f"Test MAPE: {mape:.2f}%")
        print(f"Test sMAPE: {smape:.2f}%")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSLE: {rmsle:.4f}")
        print(f"Test Time: {test_time:.4f} seconds")

        flat_timestamps = collect_timestamps.flatten()
        prediction_results = pd.DataFrame({
            'Timestamp': flat_timestamps,
            'True_RTT': y_true_rtt,
            'Predicted_RTT': y_pred_rtt,
        })
        prediction_results_path = os.path.join(results_dir, f"attention_{base_filename}_{num_steps_ahead}.csv")
        prediction_results.to_csv(prediction_results_path, index=False)

        with open(metrics_file_path, 'a') as f:
            f.write(f"Filename: {filename}\n")
            f.write(f"Num Steps Ahead: {num_steps_ahead}\n")
            f.write("Final Results:\n")
            f.write(f"Test MAPE: {mape:.2f}%\n")
            f.write(f"Test sMAPE: {smape:.2f}%\n")
            f.write(f"Test MSE: {mse:.4f}\n")
            f.write(f"Test RMSE: {rmse:.4f}\n")
            f.write(f"Test MAE: {mae:.4f}\n")
            f.write(f"Test RMSLE: {rmsle:.4f}\n")
            f.write("-" * 50 + "\n")

        # scatter figure
        scatter_image_path = os.path.join(results_dir, f"scatter_attention_{base_filename}_{num_steps_ahead}.png")
        plt.figure(figsize=(16, 8))
        plt.scatter(range(len(y_true_rtt)), y_true_rtt, label='True RTT', marker='o', s=5)
        plt.scatter(range(len(y_pred_rtt)), y_pred_rtt, label='Predicted RTT (Final)', marker='x', s=5)
        plt.xlabel('Time')
        plt.ylabel('RTT')
        plt.title(f'Scatter Plot for {filename} (Steps Ahead: {num_steps_ahead})')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(scatter_image_path, dpi=300)
        plt.close()

def get_state_dict(m):
    return m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict()

def load_state_dict(m, sd):
    if isinstance(m, torch.nn.DataParallel):
        m.module.load_state_dict(sd)
    else:
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        m.load_state_dict(sd)

if __name__ == "__main__":
    filenames = ['../traces/6h.csv']
    seq_len = 3000
    num_steps_ahead = 1
    d_model = 64
    main_improved_attention_model(filenames, seq_len, num_steps_ahead, d_model)
