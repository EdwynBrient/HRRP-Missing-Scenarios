import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

from icassp.schedulers import CosineScheduler

DEFAULT_RP_LEN = 512
selectRP = [str(i) for i in range(DEFAULT_RP_LEN)]


def set_rp_length(length: int):
    """Update the global RP column list in place so all imports stay in sync."""
    length = int(length)
    if length <= 0:
        raise ValueError("rp_length must be positive")
    selectRP[:] = [str(i) for i in range(length)]


def plot_generated_true(HRRP, generated, scal, save_path, min_rp, max_rp, mode, epoch):
    HRRP = HRRP.detach().cpu().numpy()
    generated = np.array(generated)
    save_path = os.path.join(save_path, "figures")
    os.makedirs(save_path, exist_ok=True)
    rp_len = HRRP.shape[-1]
    little_title = [str(np.array(np.round(scal[0], 2)))+", "+str(np.array(np.round(scal[1], 2)))+", "+str(np.array(np.round(scal[2], 2))) for scal in scal]
    little_title = np.array(little_title).reshape(-1, 6)
    HRRP, generated = unnormalize_hrrp(HRRP, min_rp, max_rp), unnormalize_hrrp(generated, min_rp, max_rp)
    HRRP, generated = np.reshape(HRRP, (-1, 6, rp_len)), np.reshape(generated, (-1, 6, rp_len))
    plot_HRRP(HRRP, "True HRRP", little_title, save_path + "/" + "true_hrrps_" + mode + str(epoch) + ".png", True)
    plot_HRRP(generated, "Final Generated HRRP", little_title, save_path + "/" + "final_gen_hrrps_" + mode + str(epoch) + ".png", True)
    HRRP, generated = np.reshape(HRRP, (-1, rp_len)), np.reshape(generated, (-1, rp_len))
    plt.close("all")
    return HRRP, generated

def compute_metrics(model, dataset, test_idx, min_rp, max_rp, model_type, batch_size=30, device=None):
    """
    Run a lightweight evaluation loop on the provided indices.
    Returns a dict with averaged PSNR / COSF / MSEF values.
    """
    if not test_idx:
        return {}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    tpsnr_vals, tcosf_vals, tmsef_vals = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if model_type == "ddpm":
                vars, idx = batch
                vars = vars.to(device)
                generated, _ = model([vars])
                generated = torch.tensor(generated, dtype=torch.float32, device=device)
            else:
                vars, idx = batch
                vars = vars.to(device)
                scal = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float()
                z = torch.randn(vars.shape[0], 1, model.generator.latent_len, device=device)
                generated = model(z, scal)

            generated = unnormalize_hrrp(generated.detach().cpu(), min_rp, max_rp)
            metrics = model.compute_loss("test", idx, generated)
            tpsnr_vals.append(metrics[0].detach().cpu())
            tcosf_vals.append(metrics[1].detach().cpu())
            tmsef_vals.append(metrics[2].detach().cpu())

    tpsnr = torch.cat(tpsnr_vals).mean().item()
    tcosf = torch.cat(tcosf_vals).mean().item()
    tmsef = torch.cat(tmsef_vals).mean().item()

    return {"psnr": tpsnr, "cosf": tcosf, "msef": tmsef}

def train_val_split(dataset, generalize=False, val_size=0.2, seed=8, rng=None, test=True):
    """
    Fast split helper:
    - generalize=False: standard random_split (reproducible).
    - generalize=True & rng is None: split on unique MMSI groups (val/test by MMSI).
    - generalize=True & rng float in [0,1]: contiguous index window for val, rest in train.
    Returns: (train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, test_mmsis)
    """
    n = len(dataset)
    all_idx = np.arange(n, dtype=np.int64)

    if not generalize:
        g = torch.Generator().manual_seed(seed)
        n_val = int(round(n * val_size))
        train_len = n - n_val
        train_ds, val_ds = random_split(dataset, [train_len, n_val], generator=g)
        # No test split in this mode
        empty = []
        return train_ds, val_ds, torch.utils.data.Subset(dataset, empty), \
               list(range(train_len)), list(range(train_len, n)), empty, []

    # ----- generalize=True -----
    rng_np = np.random.default_rng(seed)

    if rng is None or rng == "None":
        # Draw unique MMSI groups (vectorized)
        mmsi_col = dataset.df["mmsi"].to_numpy()  # (N,)
        unique_mmsis = np.unique(mmsi_col)
        nb_mmsis = int(len(unique_mmsis) * val_size)

        if nb_mmsis == 0:
            # fallback: rien en val, tout en train
            train_idx = all_idx
            val_idx = np.array([], dtype=np.int64)
            test_idx = np.array([], dtype=np.int64)
            test_mmsis = np.array([], dtype=unique_mmsis.dtype)
        else:
            val_mmsis = rng_np.choice(unique_mmsis, nb_mmsis, replace=False)
            test_mmsis = np.array([], dtype=unique_mmsis.dtype)
            if test:
                # send half of val MMSIs to test
                nb_test = max(nb_mmsis // 2, 1) if nb_mmsis > 1 else 0
                if nb_test > 0:
                    test_mmsis = rng_np.choice(val_mmsis, nb_test, replace=False)
                    # MMSI de validation = val_mmsis \ test_mmsis
                    val_mmsis = np.setdiff1d(val_mmsis, test_mmsis, assume_unique=False)

            mask_val  = np.isin(mmsi_col, val_mmsis)
            mask_test = np.isin(mmsi_col, test_mmsis) if test_mmsis.size > 0 else np.zeros(n, dtype=bool)
            mask_train = ~(mask_val | mask_test)

            val_idx   = all_idx[mask_val]
            test_idx  = all_idx[mask_test]
            train_idx = all_idx[mask_train]

        # Build Subsets
        train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
        test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())

        return train_ds, val_ds, test_ds, \
               train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), test_mmsis.tolist()

    else:
        # numeric rng: contiguous index window (vectorized)
        r = float(rng)
        r = 0.0 if r < 0 else (1.0 if r > 1.0 else r)
        if r == 1.0:
            r = r - val_size  # ensure at least one sample in val

        start = int(r * n)
        n_val = int(val_size * n)
        stop  = min(start + n_val, n)
        val_idx = np.arange(start, stop, dtype=np.int64)

        if test and val_idx.size > 0:
            # half of val indices go to test (stable draw)
            test_idx = rng_np.choice(val_idx, val_idx.size // 2, replace=False)
            # remove test indices from val
            keep_mask = np.ones(val_idx.shape[0], dtype=bool)
            # mark positions belonging to test_idx
            # fast path: use a set for O(1) lookups
            test_set = set(test_idx.tolist())
            for j, v in enumerate(val_idx):
                if v in test_set:
                    keep_mask[j] = False
            val_idx = val_idx[keep_mask]
        else:
            test_idx = np.array([], dtype=np.int64)

        keep = np.ones(n, dtype=bool)
        keep[val_idx] = False
        keep[test_idx] = False
        train_idx = all_idx[keep]

        train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
        test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())
        return train_ds, val_ds, test_ds, \
               train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

def top_loss(df, idx, generated, minrp, maxrp, tol_angle=0.01):
    """Compute the min MSE loss of all HRRPs of the same ship at tol_angle (rad)"""
    # Extract relevant data
    mmsi = df.iloc[idx].mmsi
    df_mmsi = df[df.mmsi == mmsi].copy()
    df_mmsi[selectRP] = unnormalize_hrrp(df_mmsi[selectRP].to_numpy(), minrp, maxrp)
    va = df.iloc[int(idx)].viewing_angle * 6.28
    df_va_filtered = df_mmsi.viewing_angle * 6.28  # Keep original indices

    # Define range bounds
    lower_bound = (va - tol_angle) % (2 * np.pi)
    upper_bound = (va + tol_angle) % (2 * np.pi)

    # Handle cases where range wraps around 0 or 2π
    if lower_bound < upper_bound:
        mask_va = (df_va_filtered > lower_bound) & (df_va_filtered < upper_bound)
    else:
        # Wrap-around case: angles are in two separate intervals
        mask_va = (df_va_filtered > lower_bound) | (df_va_filtered < upper_bound)

    df_around = df_mmsi[mask_va]
    rp_around = df_around[selectRP].to_numpy()

    generated = np.expand_dims(generated, axis=0)
    
    # Compute MFN calculate cosine & MSE
    rp_around = torch.Tensor(rp_around)
    generated = torch.Tensor(generated)
    lpf_gen, f_gen, _ = mfn_decomposition_2D(generated, 0.5)
    lpf_true, f_true, _ = mfn_decomposition_2D(rp_around, 0.5)

    mse_matrix, cosine_matrix = f_mse(f_true, lpf_true, f_gen, lpf_gen)
    mse_matrix, cosine_matrix = mse_matrix.cpu().numpy(), cosine_matrix.cpu().numpy()
    lpf_gen, lpf_true = lpf_gen.cpu().numpy(), lpf_true.cpu().numpy()
    best_idx = int(np.argmin(mse_matrix))
    lpf_best = 0.5 * (lpf_gen[0] + lpf_true[best_idx])   # -> (L,)
    lpf_best = (lpf_best > 0).sum()  # sum of activated cells (superior to zero, where the signal stands)

    psnr_active = psnr_on_active_subset(
        x=generated[0], y=rp_around[best_idx],
        mx=lpf_gen[0],   my=lpf_true[best_idx],
        maxrp=maxrp, region="union", thr=0.0
    )
    mse_matrix = np.clip(mse_matrix, a_min=1e-12, a_max=30.)  # avoid div by zero in psnr
    return psnr_active, np.max(cosine_matrix), np.min(mse_matrix), lpf_best

def psnr_on_active_subset(x, y, mx, my, maxrp, region="union", thr=0.0, eps=1e-12):
    """
    x, y : (L,) HRRP (de-normalized, same scale as maxrp)
    mx,my: (L,) LPF/masks (e.g., lpf_gen, lpf_true)
    region: "union" | "intersection" | "true" | "pred"
    thr   : activation threshold (e.g., 0.0 or 1e-6)
    """
    ax = mx > thr
    ay = my > thr
    if region == "union":
        mask = ax | ay
    elif region == "intersection":
        mask = ax & ay
    elif region == "true":
        mask = ax
    elif region == "pred":
        mask = ay
    else:
        raise ValueError(region)
    x, y = x.squeeze(), y.squeeze()
    n = mask.sum()
    if n == 0:
        return float("nan")  # ou 100.0 selon ton choix
    mse_active = ((x - y)[mask]**2).mean()   # Standard MSE but on the active subset only
    psnr = 20.0 * torch.log10(
        torch.tensor(maxrp, dtype=x.dtype, device=x.device) / torch.sqrt(mse_active + eps)
    )
    return float(psnr.item())

def load_scheduler_from_config(config):
    sched_params = config["scheduler"]
    name = sched_params.get("name", "cosine")
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler '{name}'. Only 'cosine' is available.")
    return CosineScheduler(config["num_timesteps"], sched_params["s"], sched_params["clipping_value"])

def get_save_path_create_folder(config, seed):
    save_path = config["figure_path"].split(" ")[0] + str(date.today()) + "/" + config["figure_path"].split(" ")[1]+"_seed"+str(seed)
    os.makedirs(save_path, exist_ok=True)
    dirs = [int(i) for i in os.listdir(save_path)]
    if len(dirs) == 0:
        z = 0
    else:
        z = np.max(dirs) + 1
    save_path = save_path + "/" + str(z)
    os.makedirs(save_path, exist_ok=True)

    return save_path

def unnormalize_hrrp(hrrp, min_rp, max_rp):
    hrrp = (hrrp + 1) / 2  # Back to [0, 1]
    hrrp = hrrp * (max_rp - min_rp) + min_rp
    return hrrp

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    config.setdefault("rp_length", len(selectRP))

    return config

def uniform_filter_1d(signal, kernel_size=11):
    kernel = torch.ones(kernel_size)/kernel_size
    kernel = kernel.view(1, 1, -1).to(signal.device)
    smoothed_signal = F.conv1d(signal.unsqueeze(1), kernel, padding=int(kernel_size//2)).squeeze(1)
    return smoothed_signal

def get_df_RP_length(df, tresh=0.25, return_first_last=False, kernel_size=11):
    """
    Computes ship length using uniform filter-based detection.
    Uses first 1 and last -1 edge in the smoothed binary mask.
    """
    if isinstance(df, pd.DataFrame):
        global selectRP  # assuming it's set externally
        values = df[selectRP].values
        signal = torch.tensor(values, dtype=torch.float32)
    else:
        signal = df.float() if isinstance(df, torch.Tensor) else torch.tensor(df, dtype=torch.float32)

    smoothed = uniform_filter_1d(signal, kernel_size=kernel_size)

    # Compute threshold per signal
    if smoothed.ndim == 2:
        tresh_vals = tresh * torch.max(smoothed, dim=1, keepdim=True)[0]
        binary_mask = smoothed > tresh_vals
    else:
        tresh_val = tresh * torch.max(smoothed)
        binary_mask = (smoothed > tresh_val).unsqueeze(0)

    lengths, starts, ends = detect_ship(binary_mask)

    if not return_first_last:
        return lengths if smoothed.ndim != 1 else lengths[0]
    else:
        return (lengths, starts, ends) if smoothed.ndim != 1 else (lengths[0], starts[0], ends[0])

def mfn_decomposition_2D(RP, sigma, kernel_size=17):
    RP = RP.squeeze()
    if RP.ndim == 1:
        RP = RP.unsqueeze(0)
    num_signals, signal_length = RP.shape

    N = RP.shape[-1]
    mask = RP.new_ones(N)
    mask[185:] = 0.
    RP = RP * mask  # pas d'in-place, graphe propre

    # Get signal lengths and boundaries for all signals
    lrp, first, last = get_df_RP_length(RP, tresh=0.5, return_first_last=True)

    # Create indices as a 2D matrix: shape (num_signals, signal_length)
    indices = torch.tile(torch.arange(signal_length), (num_signals, 1))  # Shape (num_signals, signal_length)

    # --- Compute m component (Mean inside first:last) ---
    in_range = (indices >= first[:, None]) & (indices < last[:, None])  # Boolean mask
    means = torch.sum(RP * in_range, axis=1) / (torch.sum(in_range, axis=1)+1e-2)  # Compute mean only in range

    lpf = torch.zeros_like(RP)
    lpf[in_range] = torch.repeat_interleave(means[:, None], signal_length, axis=1)[in_range]  # Assign mean where in range

    # --- Compute mask ---
    mask = torch.ones_like(RP)

    # Left side mask
    left_mask = indices < first[:, None]  # Boolean mask for left side
    mask[left_mask] = torch.exp(2*(indices - first[:, None]) / (lrp[:, None] / 3))[left_mask]

    # Right side mask
    right_mask = indices >= last[:, None]  # Boolean mask for right side
    mask[right_mask] = torch.exp(2*(last[:, None] - indices) / (lrp[:, None] / 3))[right_mask]
    
    # --- Compute f component ---
    f_comp = gaussian_filter_1d(RP, kernel_size,  sigma) * mask

    # --- Compute n component ---
    n_comp = RP - f_comp

    return lpf, f_comp, n_comp

def f_mse(fx, mx, fy, my):
    if fy.shape[0] > fx.shape[0]:
        fx, mx = fx.repeat(fy.shape[0], 1), mx.repeat(fy.shape[0], 1)
    elif fy.shape[0] < fx.shape[0]:
        fy, my = fy.repeat(fx.shape[0], 1), my.repeat(fx.shape[0], 1)
    assert fx.shape[0] == fy.shape[0] and mx.shape[0] == fx.shape[0]
    num_signals = fx.shape[0]
    mse_matrix = torch.zeros((num_signals))
    cosine_matrix = torch.zeros((num_signals))
    for i in range(num_signals):
        if (mx[i]>0).sum() > (my[i]>0).sum(): # widen the smaller mask to match the larger one
            mi = mx[i]
            mj = my[i]
            mj[mi>0] = mj.max()                
        else:
            mj = my[i]
            mi = mx[i]
            mi[mj>0] = mi.max()
        if (mi>0).sum() == 0:
            fmi, lmi = 0, fx.shape[1]
        else:
            fmi, lmi = torch.argwhere(mi>0)[0][0],  torch.argwhere(mi>0)[-1][0]+1
        mse_matrix[i] = torch.mean((fx[i] - fy[i]) ** 2)/(((mx[i]>0).sum()+(my[i]>0).sum())/2)
        cosine_matrix[i] = cosine_similarity((fx[i]-mi)[fmi:lmi].reshape(1, -1), (fy[i]-mj)[fmi:lmi].reshape(1, -1)).item()
    return mse_matrix, cosine_matrix

def plot_HRRP(hrrp, global_title, little_title, file, save=False):
    assert len(hrrp.shape) == 3, "shape of hrrps must be 3, size you tried {}".format(hrrp.shape)
    fig, axs = plt.subplots(hrrp.shape[0], hrrp.shape[1], figsize=(16, 3*hrrp.shape[0]), layout='constrained')
    rp_len = hrrp.shape[2]
    for i in range(hrrp.shape[0]*hrrp.shape[1]):
        axs[i//hrrp.shape[1], i%hrrp.shape[1]].plot(range(1, rp_len+1), hrrp[i//hrrp.shape[1], i%hrrp.shape[1]])
        if isinstance(little_title, np.ndarray):
            axs[i//hrrp.shape[1], i%hrrp.shape[1]].set_title(str(little_title[i//hrrp.shape[1], i%hrrp.shape[1]]))
        if isinstance(little_title, list):
            axs[i // hrrp.shape[1], i % hrrp.shape[1]].set_title(
                str(little_title[i // hrrp.shape[1]][i % hrrp.shape[1]]))
        fig.suptitle(global_title)
    if save:
        plt.savefig(file)
    else:
        plt.show()

### =====  Translation helpers ===== ###

# --- 1D Gaussian filter (like uniform_filter_1d but Gaussian kernel) ---
def gaussian_filter_1d(signal: torch.Tensor, kernel_size=21, sigma=3.0):
    """
    signal: (B, L) or (L,)  -> returns (B, L)
    """
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)  # (1, L)
    # gaussian kernel
    x = torch.arange(kernel_size, device=signal.device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    # conv
    smoothed = F.conv1d(signal.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)
    return smoothed  # (B, L)

# --- Robust start/end detection via morphology ---
def apply_dilation_erosion(x, kernel_size=15):
    dilation = torch.nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size // 2)
    dilated = dilation(x)
    eroded = -dilation(-dilated)
    return eroded

def detect_ship(binary_mask):
    """
    binary_mask: (B, L) bool/0-1 float
    Returns: lengths, starts, ends (tensors int32)
    """
    if binary_mask.ndim == 1:
        binary_mask = binary_mask.reshape(1, -1)
    binary_mask = binary_mask.float()
    detected = apply_dilation_erosion(binary_mask)

    diff = torch.diff(detected.int(), dim=1)
    changes = torch.zeros_like(detected, dtype=torch.int32)
    changes[:, 1:] = diff

    B, L = detected.shape
    starts = torch.full((B,), -1, dtype=torch.int32)
    ends   = torch.full((B,), -1, dtype=torch.int32)

    for i in range(B):
        change_i = changes[i]
        rising_edges  = torch.where(change_i == 1)[0]
        falling_edges = torch.where(change_i == -1)[0]

        if len(rising_edges) > 0 and len(falling_edges) > 0:
            starts[i] = rising_edges[0].item()
            ends[i]   = falling_edges[-1].item()
        else:
            # fallback: around the maximum
            max_idx = torch.argmax(binary_mask[i]).item()
            starts[i] = max(0, max_idx - 1)
            ends[i]   = min(L - 1, max_idx + 1)

    lengths = ends - starts
    return lengths, starts, ends
