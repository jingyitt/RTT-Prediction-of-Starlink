import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
def fit_standardizer(x, eps=1e-6):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.maximum(sd, eps)
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_standardizer(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32)
# ========= Graph Inference: per second -> (switching_probability, next_confidence, margin) =========
@torch.no_grad()
def infer_graph_features_per_second(graph_ckpt_path, graphs_pt_path, device="cpu"):
    ckpt = torch.load(graph_ckpt_path, map_location="cpu")
    in_dim = ckpt["in_dim"]
    hidden_dim = ckpt["hidden_dim"]
    emb_dim = ckpt["emb_dim"]

    encoder = GraphSAGEEncoder(in_dim, hidden_dim=hidden_dim, out_dim=emb_dim, dropout=0.0).to(device)
    next_head = NextSatHead(emb_dim).to(device)
    switch_head = SwitchHead(emb_dim, hidden=64).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    next_head.load_state_dict(ckpt["next_head"])
    switch_head.load_state_dict(ckpt["switch_head"])
    encoder.eval(); next_head.eval(); switch_head.eval()

    gpt = torch.load(graphs_pt_path, map_location="cpu", weights_only=False)
    graphs = gpt["graphs"]
    T = len(graphs)

    # Retrieve the ground-truth timeline for the graphs, otherwise, read from Data.t_unix
    if "graph_times" in gpt:
        graph_times = np.asarray(gpt["graph_times"], dtype=np.float64)  # (T,)
    else:
        graph_times = np.array(
            [float(getattr(g, "t_unix").item()) for g in graphs],
            dtype=np.float64
        )

    per_sec = np.zeros((T, 5), dtype=np.float32)  # [sw_prob, next_conf, margin, observed, service_valid]

    for t in range(T):
        g = graphs[t]
        sv = int(getattr(g, "service_valid", 0))
        obs = int(getattr(g, "observed", 0))
        per_sec[t, 3] = obs
        per_sec[t, 4] = sv
        if g.x is None or g.x.numel() == 0:
            continue
        edge_index = g.edge_index if (g.edge_index is not None) else torch.empty((2, 0), dtype=torch.long)

        x = g.x.to(device)
        edge_index = edge_index.to(device)

        node_emb = encoder(x, edge_index)
        logits = next_head(node_emb)

        ci = int(g.service_idx.item()) if hasattr(g, "service_idx") else -1
        h_cur = node_emb[ci] if (0 <= ci < node_emb.size(0)) else node_emb.mean(dim=0)
        sw_prob = torch.sigmoid(switch_head(h_cur)).item()

        used_logits = logits.clone()
        if 0 <= ci < used_logits.numel():
            used_logits[ci] = -1e9
        p = torch.softmax(used_logits, dim=0)
        top2 = torch.topk(p, k=min(2, p.numel())).values
        next_conf = float(top2[0].item())
        margin = float((top2[0] - top2[1]).item()) if top2.numel() > 1 else float(top2[0].item())

        per_sec[t, 0:3] = (sw_prob, next_conf, margin)

    # normalize
    valid = per_sec[:, 4] > 0
    if valid.any():
        mu, sd = fit_standardizer(per_sec[valid, 0:3])
        per_sec_norm = per_sec.copy()
        per_sec_norm[valid, 0:3] = apply_standardizer(per_sec[valid, 0:3], mu, sd)
        per_sec = per_sec_norm
    else:
        mu = np.zeros((3,), dtype=np.float32)
        sd = np.ones((3,), dtype=np.float32)

    return per_sec, graphs, graph_times, (mu, sd)

import numpy as np

def align_graph_feat_to_rtt(
    timestamps_rtt,
    graph_times,
    per_sec_feat,
    mode="floor",
    observed_col=None,
    service_valid_col=None,
    min_valid=0.5,
    return_index=False,
):
    """ Align per-second graph features to higher-rate RTT timestamps.
        timestamps_rtt: (N,) Unix timestamps for RTT samples (float).
        graph_times:    (T,) Unix timestamps for per-second graphs (float), typically at 1-second spacing.
        per_sec_feat:   (T, F) Feature matrix for each per-second graph. It is recommended to include
                        validity flags such as `observed` and/or `service_valid` as additional columns.
        mode:
            - "floor":   Use the last graph where graph_time <= rtt_time (strictly causal, no look-ahead).
            - "nearest": Use the closest graph in time (may look ahead up to ~0.5s, not strictly causal).
        observed_col: Column index of the `observed` flag in per_sec_feat.
        service_valid_col: Column index of the `service_valid` flag in per_sec_feat, If None, defaults to the last column.
        min_valid: Threshold for validity flags. Typically 0.5 if flags are 0/1.
        return_index: If True, also return the aligned graph indices for debugging.
    Returns:
        aligned_feat: (N, F) Aligned features for each RTT timestamp.
        valid_mask:   (N,) bool mask indicating whether the aligned graph feature is considered valid
                      (useful for filtering samples or zero-weighting loss).
        idx (optional): (N,) int indices of the per-second graphs selected for each RTT sample.
    """
    ts = np.asarray(timestamps_rtt, dtype=np.float64)
    gt = np.asarray(graph_times, dtype=np.float64)
    feat = np.asarray(per_sec_feat, dtype=np.float32)

    # Ensure (graph_times, per_sec_feat) are sorted by time
    order = np.argsort(gt)
    gt = gt[order]
    feat = feat[order]

    # Compute alignment indices
    if mode == "floor":
        # Last element <= ts (strictly causal)
        idx = np.searchsorted(gt, ts, side="right") - 1
        idx = np.clip(idx, 0, len(gt) - 1)
    elif mode == "nearest":
        # Closest element (may look ahead)
        j = np.searchsorted(gt, ts, side="left")
        j0 = np.clip(j - 1, 0, len(gt) - 1)
        j1 = np.clip(j,     0, len(gt) - 1)
        choose_right = (np.abs(gt[j1] - ts) < np.abs(gt[j0] - ts))
        idx = np.where(choose_right, j1, j0)
    else:
        raise ValueError("mode must be 'floor' or 'nearest'")

    aligned = feat[idx]  # (N, F)

    # Build a validity mask. By default, require service_valid == 1 (or >= min_valid).
    if service_valid_col is None:
        service_valid_col = aligned.shape[1] - 1
    sv = aligned[:, service_valid_col]
    valid_mask = sv >= float(min_valid)

    # also require observed == 1 (or >= min_valid), meaning use only truly observed seconds
    if observed_col is not None:
        ob = aligned[:, observed_col]
        valid_mask = valid_mask & (ob >= float(min_valid))

    if return_index:
        return aligned, valid_mask, idx
    return aligned, valid_mask

# -------------------------
# Model
# -------------------------
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # (num_nodes, out_dim)


class NextSatHead(nn.Module):
    """Assign a score to each candidate node:logit_i = w^T h_i"""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.scorer = nn.Linear(emb_dim, 1)

    def forward(self, node_emb):
        # node_emb: (num_nodes, emb_dim)
        return self.scorer(node_emb).squeeze(-1)  # (num_nodes,)

class SwitchHead(nn.Module):
    """Input: Embedding of the current serving satellite (h_cur).
    Output: Logit of the switching probability."""
    def __init__(self, emb_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_cur):
        # h_cur: (emb_dim,) or (B, emb_dim)
        return self.mlp(h_cur).squeeze(-1)  # scalar or (B,)

# Build training pairs
def build_nextsat_pairs(graphs, K: int = 1):
    """Returns list[Data] where each Data is derived from graph_t,
    with label_next_idx being the index of the serving satellite from graph_{t+K} within the candidate set of graph_t.
    additionally includes:
      - cur_service_idx as the current serving satellite's position in graph_t (g.service_idx)
      - is_switch as a 0/1 flag indicating if a handover occurs within the next K steps (next != cur).
      skipping the entry if not found."""
    pairs = []
    T = len(graphs)
    for t in range(T - K):
        g = graphs[t]
        g_next = graphs[t + K]

        if not hasattr(g, "sat_names") or not hasattr(g_next, "sat_names"):
            continue
        if g.x is None or g.x.numel() == 0:
            continue

        # Current serving satellite index (the local index within g.sat_names).
        cur_idx = int(g.service_idx.item())
        if cur_idx < 0:
            continue
        if cur_idx >= len(g.sat_names):
            continue

        # Future serving satellite name (found using g_next.service_idx within g_next.sat_names).
        next_serv_idx = int(g_next.service_idx.item())
        if next_serv_idx < 0:
            continue
        if next_serv_idx >= len(g_next.sat_names):
            continue
        next_serv_name = g_next.sat_names[next_serv_idx]

        # The future serving satellite must exist in the current candidate set (otherwise classification is not possible).
        if next_serv_name not in g.sat_names:
            continue

        label_next_idx = g.sat_names.index(next_serv_name)

        # Handover status: Future serving satellite at step K != current serving satellite.
        cur_serv_name = g.sat_names[cur_idx]
        is_switch = 1 if (next_serv_name != cur_serv_name) else 0

        data = Data(
            x=g.x,
            edge_index=g.edge_index,
            service_idx=g.service_idx,
            sat_names=g.sat_names,
            t_index=getattr(g, "t_index", t),
        )
        data.label_next_idx = torch.tensor(label_next_idx, dtype=torch.long)
        data.cur_service_idx = torch.tensor(cur_idx, dtype=torch.long)
        data.is_switch = torch.tensor(is_switch, dtype=torch.long)
        pairs.append(data)

    return pairs

def build_union_pairs(graphs, K: int = 1, use_union_edges: bool = True):
    """Each sample is derived from time t, but the classification space is the union(t, t+K).
      Return list[Data] where each Data's x and edge_index are based on the new unionized graph.
      Labels:
       - cur_service_idx: Position of the current serving satellite in the union list
       - label_next_idx: Position of the future serving satellite at step K in the union list
       - is_switch: Whether a handover occurs within the next K steps (next != cur)"""
    pairs = []
    T = len(graphs)

    for t in range(T - K):
        g = graphs[t]
        g_next = graphs[t + K]

        # basic checks
        if not hasattr(g, "sat_names") or not hasattr(g_next, "sat_names"):
            continue
        if g.x is None or g.x.numel() == 0:
            continue
        if g_next.x is None or g_next.x.numel() == 0:
            continue

        # current service sat in g
        cur_local = int(g.service_idx.item())
        if cur_local < 0 or cur_local >= len(g.sat_names):
            continue
        cur_name = g.sat_names[cur_local]

        # next service sat in g_next
        next_local = int(g_next.service_idx.item())
        if next_local < 0 or next_local >= len(g_next.sat_names):
            continue
        next_name = g_next.sat_names[next_local]

        # union sat list (keep stable order: all t first, then new ones from t+K)
        union_names = list(g.sat_names)
        for nm in g_next.sat_names:
            if nm not in union_names:
                union_names.append(nm)

        # mapping name -> union index
        name2u = {nm: i for i, nm in enumerate(union_names)}

        # current / next in union
        cur_u = name2u[cur_name]
        next_u = name2u[next_name]
        is_switch = 1 if (next_name != cur_name) else 0

        # build union x: take feature from g if exists else from g_next
        in_dim = g.x.size(-1)
        x_list = []
        for nm in union_names:
            if nm in g.sat_names:
                idx = g.sat_names.index(nm)
                x_list.append(g.x[idx].view(1, in_dim))
            else:
                idx2 = g_next.sat_names.index(nm)
                x_list.append(g_next.x[idx2].view(1, in_dim))

        x_union = torch.cat(x_list, dim=0)  # (Nu, in_dim)

        # reset is_service / bias columns based on t's service
        # x = [elev, azim, dist, rr, is_service, ones]
        x_union = x_union.clone()
        if x_union.size(1) >= 6:
            x_union[:, 4] = 0.0
            x_union[cur_u, 4] = 1.0
            x_union[:, 5] = 1.0

        # build union edges
        # If use_union_edges=True: map the edges of both g and g_next to the union set and merge them;
        # otherwise: use only the edges from g (operable, but with less information).
        edges = []

        def remap_edges(edge_index, names_src):
            if edge_index is None:
                return
            # edge_index: (2, E)
            for e in range(edge_index.size(1)):
                a = int(edge_index[0, e].item())
                b = int(edge_index[1, e].item())
                if a < 0 or a >= len(names_src) or b < 0 or b >= len(names_src):
                    continue
                na = names_src[a]
                nb = names_src[b]
                ua = name2u.get(na, None)
                ub = name2u.get(nb, None)
                if ua is None or ub is None:
                    continue
                edges.append((ua, ub))

        # add edges from g
        if hasattr(g, "edge_index") and g.edge_index is not None:
            remap_edges(g.edge_index, g.sat_names)

        # add edges from g_next
        if use_union_edges and hasattr(g_next, "edge_index") and g_next.edge_index is not None:
            remap_edges(g_next.edge_index, g_next.sat_names)

        if len(edges) == 0:
            edge_index_union = torch.empty((2, 0), dtype=torch.long)
        else:
            edges = list(set(edges))  # Remove duplicates
            edge_index_union = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(
            x=x_union,
            edge_index=edge_index_union,
        )
        data.sat_names = union_names
        data.cur_service_idx = torch.tensor(cur_u, dtype=torch.long)
        data.label_next_idx = torch.tensor(next_u, dtype=torch.long)
        data.is_switch = torch.tensor(is_switch, dtype=torch.long)
        data.t_index = getattr(g, "t_index", t)

        pairs.append(data)

    return pairs

def make_soft_target_from_features(x_local, true_idx, elev_col=0, range_col=2, rr_col=3,
                                   temp=5.0, eps=0.2, mask_idx=None):
    N = x_local.size(0)
    onehot = torch.zeros(N, device=x_local.device)
    onehot[true_idx] = 1.0

    if elev_col >= x_local.size(1) or range_col >= x_local.size(1):
        return onehot

    elev = x_local[:, elev_col]
    dist = x_local[:, range_col]

    # rr optional
    score = 2.0 * elev - 1.0 * dist
    if rr_col is not None and rr_col < x_local.size(1):
        rr = x_local[:, rr_col].abs()
        score = score - 1.0 * rr

    base = torch.softmax(temp * score, dim=0)
    if mask_idx is not None and 0 <= mask_idx < N:
        base = base.clone()
        base[mask_idx] = 0.0
        base = base / (base.sum() + 1e-12)
    y_soft = (1.0 - eps) * onehot + eps * base
    return y_soft / (y_soft.sum() + 1e-12)

# Train loop
def train(
    graph_pt="graphs.pt",
    save_path="dgcn_two_stage.pt",
    hidden_dim=64,
    emb_dim=64,
    lr=1e-3,
    batch_size=64,
    epochs=10,
    device=None,
    K=1,
    lambda_switch=1.0,
    lambda_next=1.0,
    switch_threshold=0.5,
    use_union_edges=True,
    use_soft_target=True,
    elev_col=0,
    range_col=2,
    temp=5.0,
    eps=0.2,
):
    ckpt = torch.load(graph_pt, map_location="cpu")
    graphs = ckpt["graphs"]

    train_data = build_union_pairs(graphs, K=K, use_union_edges=use_union_edges)
    if len(train_data) == 0:
        raise RuntimeError("No training samples built.")

    print(f"[INFO] built {len(train_data)} training samples from {len(graphs)} graphs (K={K}, union)")

    in_dim = train_data[0].x.size(-1)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = GraphSAGEEncoder(in_dim, hidden_dim=hidden_dim, out_dim=emb_dim, dropout=0.1).to(device)
    next_head = NextSatHead(emb_dim).to(device)
    switch_head = SwitchHead(emb_dim, hidden=64).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(next_head.parameters()) + list(switch_head.parameters()),
        lr=lr
    )

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    encoder.train()
    next_head.train()
    switch_head.train()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        # switch metrics
        swdet_total = 0
        swdet_correct = 0

        # next-sat ranking metrics on switch subset
        nxt_total = 0
        nxt_top1 = 0
        nxt_top5 = 0
        nxt_top10 = 0

        # final decision metrics (two-stage decision)
        final_total = 0
        final_top1 = 0
        final_top5 = 0
        final_top10 = 0

        for batch in loader:
            batch = batch.to(device)

            node_emb = encoder(batch.x, batch.edge_index)  # (sumN, emb_dim)
            logits_all = next_head(node_emb)               # (sumN,)

            ptr = batch.ptr
            y_next = batch.label_next_idx      # (B,)
            cur_idx = batch.cur_service_idx    # (B,)
            y_switch = batch.is_switch.float() # (B,) 0/1

            loss_list = []

            # Process on a per-graph basis (as the number of nodes varies per graph).
            for i in range(y_next.numel()):
                start = int(ptr[i].item())
                end = int(ptr[i + 1].item())

                local_emb = node_emb[start:end]       # (Ni, emb_dim)
                local_logits = logits_all[start:end]  # (Ni,)

                yi = int(y_next[i].item())
                ci = int(cur_idx[i].item())
                swi = int(y_switch[i].item())
                Ni = int(local_logits.numel())
                if Ni <= 0:
                    continue

                # ========== switch detection loss ==========
                # Binary classification using the current serving satellite's embedding
                if 0 <= ci < Ni:
                    h_cur = local_emb[ci]  # (emb_dim,)
                else:
                    # In extreme cases where cur_idx is out of bounds, fallback to the graph mean
                    h_cur = local_emb.mean(dim=0)

                sw_logit = switch_head(h_cur)  # scalar
                loss_sw = F.binary_cross_entropy_with_logits(
                    sw_logit.view(1),
                    torch.tensor([float(swi)], device=sw_logit.device)
                )

                # switch det acc
                sw_prob = torch.sigmoid(sw_logit).item()
                sw_pred = 1 if (sw_prob >= switch_threshold) else 0
                swdet_total += 1
                if sw_pred == swi:
                    swdet_correct += 1

                # ========== next-sat loss (only for true switch samples) ==========
                # During training, the next_loss can be calculated exclusively for samples where swi == 1
                loss_next = torch.tensor(0.0, device=local_logits.device)

                if swi == 1:
                    # To avoid triviality, mask the current serving satellite from the candidate set during a handover
                    used_logits = local_logits
                    if 0 <= ci < Ni:
                        used_logits = local_logits.clone()
                        used_logits[ci] = -1e9

                    if use_soft_target:
                        y_soft = make_soft_target_from_features(
                            x_local=batch.x[start:end],
                            true_idx=yi,
                            elev_col=elev_col, # 0
                            range_col=range_col, # 2
                            rr_col=3,
                            temp=temp,
                            eps=eps,
                            mask_idx=ci,
                        )
                        # listwise soft CE
                        logp = F.log_softmax(used_logits, dim=0)
                        loss_next = -(y_soft * logp).sum()
                    else:
                        # hard CE
                        loss_next = F.cross_entropy(used_logits.view(1, -1), torch.tensor([yi], device=used_logits.device))

                    # next-sat ranking metrics (switch subset)
                    nxt_total += 1
                    topk = torch.topk(used_logits, k=min(10, Ni)).indices
                    if int(topk[0].item()) == yi:
                        nxt_top1 += 1
                    if (topk[:min(5, Ni)] == yi).any().item():
                        nxt_top5 += 1
                    if (topk[:min(10, Ni)] == yi).any().item():
                        nxt_top10 += 1
                else:
                    # noswitch: next is the current serving satellite
                    pass

                # ========== final decision metrics ==========
                # Inference: Predict sw_pred, if no handover -> predict current; if handover -> select next_head argmax.
                if sw_pred == 0:
                    final_pred = ci
                    used_logits_for_eval = local_logits
                else:
                    used_logits_for_eval = local_logits.clone()
                    if 0 <= ci < Ni:
                        used_logits_for_eval[ci] = -1e9
                    final_pred = int(torch.argmax(used_logits_for_eval).item())

                final_total += 1
                if final_pred == yi:
                    final_top1 += 1
                topk_final = torch.topk(used_logits_for_eval, k=min(10, Ni)).indices
                if (topk_final[:min(5, Ni)] == yi).any().item():
                    final_top5 += 1
                if (topk_final[:min(10, Ni)] == yi).any().item():
                    final_top10 += 1

                # ========== total loss ==========
                loss = lambda_switch * loss_sw + lambda_next * loss_next
                loss_list.append(loss)

            if len(loss_list) == 0:
                continue

            batch_loss = torch.stack(loss_list).mean()

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            total_loss += float(batch_loss.item()) * len(loss_list)
            total_samples += len(loss_list)

        avg_loss = total_loss / max(total_samples, 1)

        swdet_acc = swdet_correct / max(swdet_total, 1)

        nxt_top1_acc = nxt_top1 / max(nxt_total, 1)
        nxt_top5_acc = nxt_top5 / max(nxt_total, 1)
        nxt_top10_acc = nxt_top10 / max(nxt_total, 1)

        final_top1_acc = final_top1 / max(final_total, 1)
        final_top5_acc = final_top5 / max(final_total, 1)
        final_top10_acc = final_top10 / max(final_total, 1)

        print(
            f"Epoch {ep:02d} | loss={avg_loss:.4f} | "
            f"switch_det_acc={swdet_acc:.3f} | "
            f"next_on_switch: top1={nxt_top1_acc:.3f}, top5={nxt_top5_acc:.3f}, top10={nxt_top10_acc:.3f} (n={nxt_total}) | "
            f"final: top1={final_top1_acc:.3f}, top5={final_top5_acc:.3f}, top10={final_top10_acc:.3f} (n={final_total})"
        )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "next_head": next_head.state_dict(),
            "switch_head": switch_head.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "emb_dim": emb_dim,
            "K": K,
            "use_soft_target": use_soft_target,
            "elev_col": elev_col,
            "range_col": range_col,
        },
        save_path,
    )
    print(f"[OK] saved trained model to {save_path}")

if __name__ == "__main__":
    K = 1
    train(
        graph_pt="graphs.pt",
        save_path=f"dgcn_two_stage_union_soft_K{K}.pt",
        hidden_dim=64,
        emb_dim=64,
        lr=1e-3,
        batch_size=64,
        epochs=30,
        K=K,
        use_union_edges=True,
        use_soft_target=True,
        elev_col=0,
        range_col=2,  # dist
        temp=5.0,
        eps=0.2,
    )