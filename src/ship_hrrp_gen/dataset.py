import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ship_hrrp_gen.utils import (
    selectRP,
    set_rp_length,
)

class RP_VADataset(Dataset):
    def __init__(self, config, path_rp="data/ship_hrrp.pt"):
        self.config = config
        self.lim_data = int(self.config.get("lim_data", 0))
        self.rp_length = int(self.config.get("rp_length", len(selectRP)))
        set_rp_length(self.rp_length)
        self.selectRP = selectRP

        path_rp = self._resolve_path(path_rp)
        if str(path_rp).endswith(".pt"):
            self.df = self._load_pt_as_df(path_rp)
            self.config.setdefault("disable_filters", True)
        else:
            self.df = pd.read_csv(path_rp)

        self.df = self.df.iloc[:int(self.lim_data)] if int(self.lim_data)!=0 else self.df
        self.old_va = self.df.viewing_angle.copy()
        self.min_rp, self.max_rp = self.normalize_df()
        self.preprocess_vars()
    
    def normalize_df(self):
        self.df.length = self.df.length / self.df.length.max()
        self.df.width = self.df.width / self.df.width.max()
        self.df.viewing_angle = self.df.viewing_angle / self.df.viewing_angle.max()
        min_rp, max_rp = 0., self.df[self.selectRP].max().max()
        self.df[self.selectRP] = (self.df[self.selectRP]- min_rp) / (max_rp - min_rp)
        self.df[self.selectRP] = self.df[self.selectRP] * 2 - 1
        return min_rp, max_rp

    def __len__(self):
        return len(self.df)
    
    def preprocess_vars(self):
        self.viewing_angles = torch.tensor(self.df.viewing_angle.values * self.old_va.max(), dtype=torch.float32)
        self.lengths = torch.tensor(self.df.length.values, dtype=torch.float32)
        self.widths = torch.tensor(self.df.width.values, dtype=torch.float32)
        self.hrrps = torch.tensor(np.array(self.df[self.selectRP]), dtype=torch.float32)
        self.mmsis = self.df.mmsi.unique()

    @staticmethod
    def _resolve_path(path_rp):
        """
        Resolve dataset path robustly:
        - if `path_rp` exists as-is, use it
        - else, try relative to repo root
        - else, try `repo_root/data/<filename>`
        """
        from pathlib import Path

        p = Path(str(path_rp))
        if p.exists():
            return str(p)

        repo_root = Path(__file__).resolve().parents[2]  # .../github_repo/src
        repo_root = repo_root.parent  # .../github_repo

        cand = repo_root / p
        if cand.exists():
            return str(cand)

        cand = repo_root / "data" / p.name
        if cand.exists():
            return str(cand)

        return str(p)
    
    def _load_pt_as_df(self, path_rp):
        try:
            data = torch.load(path_rp, weights_only=True)
        except TypeError:
            data = torch.load(path_rp)
        hrrps = data.get("hrrps")
        angles = data.get("aspect_angles")
        dims = data.get("ship_dims")
        if hrrps is None or angles is None or dims is None:
            raise ValueError("PT file must contain 'hrrps', 'aspect_angles', and 'ship_dims'.")
        rp_length = int(self.config.get("rp_length", hrrps.shape[1]))
        set_rp_length(rp_length)
        hrrps = hrrps[:, :rp_length].cpu().numpy()
        angles = angles.cpu().numpy()
        dims = dims.cpu().numpy()

        df = pd.DataFrame(hrrps, columns=selectRP)
        df["viewing_angle"] = angles
        df["length"] = dims[:, 0]
        df["width"] = dims[:, 1]
        df["mmsi"] = np.arange(len(df))
        return df

    def __getitem__(self, idx):
        hrrp = self.hrrps[idx]
        va = self.viewing_angles[idx:idx+1]
        length = self.lengths[idx:idx+1]
        width = self.widths[idx:idx+1]

        vars = torch.cat([hrrp.unsqueeze(0), va.unsqueeze(0), length.unsqueeze(0), width.unsqueeze(0)], dim=1)
        return vars, torch.Tensor([idx]).type(torch.uint32)
