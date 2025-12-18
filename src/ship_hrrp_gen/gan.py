import math

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from ship_hrrp_gen.utils import plot_generated_true, selectRP, top_loss, unnormalize_hrrp

class EmbedBlock(nn.Module):
    """
    abstract class
    """
    def forward(self, x, cemb):
        """
        abstract method
        """

def timestep_embedding(timesteps:torch.Tensor, dim:int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, cemb)
            else:
                x = layer(x)
        return x
    
class ResBlock(EmbedBlock):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim:int, cdim:int, dropout:float, dbcond=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.dropout = dropout
        self.dbcond = dbcond

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.c1emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, out_ch),
        )
        
        if self.dbcond is not None:
            self.c2emb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cdim, out_ch),
            )

        self.block_2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
        )

        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()

    def forward(self, x:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.c1emb_proj(cemb[0])[:, :, None]
        if self.dbcond is not None:
            latent += self.c2emb_proj(cemb[1])[:, :, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
    
class ResBlockUncond(nn.Module):
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, dropout:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dropout = dropout

        self.block_1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.block_2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            
        )
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
    
class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

class EmbedLinear(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, ):
        super().__init__()
        self.layer1 = nn.Linear(in_ch, out_ch)
        self.gain = Gain()
        self.layernorm = nn.LayerNorm(out_ch)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.gain(self.layer1(x)))

class Upsample(nn.Module):
    """
    an upsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv1d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        output = self.layer(x)
        return output

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.mod_ch = config["mod_ch"]
        self.ch_mul = config["ch_mul"]
        self.num_res_blocks = config["num_res"]
        self.conditionned = config["conditionned"]
        self.dropout = config["dropout"]
        self.rp_len = int(config.get("rp_length", len(selectRP)))
        self.cond = self.conditionned["bool"]
        self.type = self.conditionned.get("type", "scalars")
        if self.cond and self.type != "scalars":
            raise ValueError(f"Only scalar conditioning is supported in this repo (got type={self.type}).")
        self.upsample_factor = 2 ** (len(self.ch_mul) - 1)
        self.latent_len = int(config.get("latent_len", math.ceil(self.rp_len / self.upsample_factor)))
        cdim = self.mod_ch * 4

        if self.cond:
            self.scal_emb_layer = EmbedLinear(3 * self.mod_ch, cdim)

        now_ch = self.ch_mul[0] * self.mod_ch
        self.conv_in = nn.Conv1d(1, now_ch, 3, 1, 1)
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul)):
            nxt_ch = mul * self.mod_ch
            for j in range(self.num_res_blocks):
                if self.cond:
                    layers = [
                        ResBlock(now_ch, nxt_ch, cdim, cdim, self.dropout),
                    ]
                else:
                    layers = [
                        ResBlockUncond(now_ch, nxt_ch, self.dropout),
                    ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks-1:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
        self.convout = nn.Conv1d(self.ch_mul[-1] * self.mod_ch, 1, 3, 1, 1)

    def forward(self, z, conds):
        if z.shape[-1] != self.latent_len:
            z = F.interpolate(z, size=self.latent_len, mode="nearest")
        scal = conds[1] if isinstance(conds, (list, tuple)) and len(conds) == 2 else conds
        n = z.shape[0]
        scal = torch.concat([timestep_embedding(scal[:, i], self.mod_ch, 10) for i in range(scal.shape[1])], dim=1)
        if self.cond:
            scal = self.scal_emb_layer(scal).reshape(n, -1)
            cemb = [scal]
        x = self.conv_in(z)
        for block in self.upblocks:
            if self.cond:
                x = block(x, cemb)
            else:
                x = block(x, 0)
        out = self.convout(x)
        if out.shape[-1] != self.rp_len:
            out = F.interpolate(out, size=self.rp_len, mode="linear", align_corners=False)
        return out

class Discriminator(nn.Module):
    def __init__(self, rp_len: int):
        super(Discriminator, self).__init__()
        self.rp_len = int(rp_len)
        self.featureExtractor = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 5, 2, 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, 2, 2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, 5, 1, 2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )
        feature_len = self._compute_feature_len(self.rp_len)

        self.classif = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * feature_len, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _conv1d_out_length(length, kernel_size, stride, padding, dilation=1):
        return math.floor((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def _compute_feature_len(self, input_len):
        l = input_len
        l = self._conv1d_out_length(l, 5, 1, 2)
        l = self._conv1d_out_length(l, 5, 2, 2)
        l = self._conv1d_out_length(l, 5, 2, 2)
        l = self._conv1d_out_length(l, 5, 2, 2)
        l = self._conv1d_out_length(l, 5, 1, 2)
        return l

    def forward(self, input):
        emb = self.featureExtractor(input)
        emb = self.classif(emb)
        return emb

class GANlight(pl.LightningModule):
    def __init__(self, gen, disc, config, dataset, validation_indices, save_path, minmax):
        super().__init__()
        self.config = config
        self.rp_len = int(getattr(dataset, "rp_length", config.get("rp_length", len(selectRP))))
        self.config["rp_length"] = self.rp_len
        self.generator = gen(self.config)
        self.discriminator = disc(self.rp_len)
        self.dataset = dataset
        self.df = dataset.df
        self.df_va = dataset.old_va
        self.val_idx = validation_indices
        self.bs=30
        self.min_rp, self.max_rp = minmax
        self.inf_every = config["inf_every_n_epoch"]
        self.cond = config["conditionned"]["bool"]
        self.type = self.config["conditionned"]["type"]
        self.loss = self.config["loss"]
        self.save_path = save_path
        self.automatic_optimization = False
        self._last_infer_bucket = {"train": -1, "val": -1}  # step-based inference
        self._last_infer_epoch = {"train": -1, "val": -1}  # epoch-based inference

    def forward(self, inp, conds):
        return self.generator(inp, conds)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.config["lr"])
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.config["lr"])
        return [opt_g, opt_d], []

    def compute_loss(self, mode, idx, generated, sel=None):
        if sel is not None:
            t_loss = [top_loss(self.df, idx[i].cpu().item(), generated[i], self.min_rp, self.max_rp, 0.04) for i in sel.tolist()]
        else:
            t_loss = [top_loss(self.df, idx[i].cpu().item(), generated[i], self.min_rp, self.max_rp, 0.04) for i in range(generated.shape[0])]
        tpsnr, tcos_f, tmse_f, lpf = zip(*t_loss) 
        tpsnr  = torch.tensor(tpsnr,  dtype=torch.float32)
        tcos_f = torch.tensor(tcos_f, dtype=torch.float32)
        tmse_f = torch.tensor(tmse_f, dtype=torch.float32)
        lpf    = torch.tensor(lpf,    dtype=torch.float32) 
        if mode != "test":
            self.log(f"top_psnr_{mode}", tpsnr.mean(), prog_bar=True)
            self.log(f"top_cosf_{mode}", tcos_f.mean(), prog_bar=True)
            self.log(f"top_msef_{mode}", tmse_f.mean(), prog_bar=True)
        return tpsnr, tcos_f, tmse_f, lpf

    def test_inf(self, mode: str) -> bool:
        if self.current_epoch == 0:
            return False
        assert mode in ("train", "val")
        if self.config["inf_mode"] == "step":
            k = int(self.config["inf_every_n_step"])
            if k <= 0:
                return False
            next_step = self.global_step + 1
            bucket = next_step // k
            if next_step > 0 and bucket > self._last_infer_bucket[mode]:
                self._last_infer_bucket[mode] = bucket
                return True
            return False
        else:
            k = int(self.inf_every)
            if k <= 0:
                return False
            human_epoch = self.current_epoch + 1
            if human_epoch % k == 0 and self._last_infer_epoch[mode] != self.current_epoch:
                self._last_infer_epoch[mode] = self.current_epoch
                return True
            return False
    
    def training_step(self, train_batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.05, 0.05)
        vars, idx = train_batch
        RP = vars[:, :, :self.rp_len].float().to(self.device)
        scal = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float()
        z = torch.randn(RP.shape[0], 1, self.generator.latent_len).to(self.device)

        # train generator
        gen = self(z, scal)
        # adversarial loss
        self.toggle_optimizer(opt_g)
        adv_loss = F.softplus(-self.discriminator(gen)).mean()

        # L1 loss
        mse = F.mse_loss(gen, RP)
        if self.loss == "ser":
            ser_loss = F.mse_loss((gen).mean(axis=1), RP.mean(axis=1))
            mse_loss = mse + ser_loss.mean()
        else:
            mse_loss = mse

        # Total G loss
        g_loss = adv_loss + 100.*mse_loss
        self.log("g_loss", adv_loss)
        self.log("train_mse", mse_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Train discriminator
        # Valid identification
        self.toggle_optimizer(opt_d)
        real_loss = F.softplus(-self.discriminator(RP)).mean()

        # Fake identification
        fake_RP = self(z, scal)
        fake_loss = F.softplus(self.discriminator(fake_RP.detach())).mean()

        # Discriminator accuracy (only for logs)
        with torch.no_grad():
            d_accuracy_fake = 1-torch.round(torch.sigmoid(self.discriminator(fake_RP))).mean()
            d_accuracy_real = (torch.round(torch.sigmoid(self.discriminator(RP)))).mean()
            d_acc = (d_accuracy_fake + d_accuracy_real) / 2

        # Average of these loss
        d_loss = real_loss + fake_loss
        self.log("d_loss", d_loss)
        self.log("d_acc", d_acc, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        with torch.no_grad():
            if self.test_inf("train"):
                RP, generated = plot_generated_true(RP[:self.bs], fake_RP[:self.bs].cpu().numpy(), scal[:self.bs].cpu().numpy(), self.save_path, self.min_rp, self.max_rp, "train", self.current_epoch)
                self.compute_loss("train", idx, generated)

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            vars, idx = val_batch
            RP = vars[:, :, :self.rp_len].float().to(self.device)
            scal = torch.concat([vars[:, :, -3], vars[:, :, -2], vars[:, :, -1]], dim=1).float()
            z = torch.randn(RP.shape[0], 1, self.generator.latent_len).to(self.device)
            generated = self(z, scal)
            mse = F.mse_loss(generated, RP)
            if self.loss == "ser":
                ser_loss = F.mse_loss(generated.mean(axis=1), RP.mean(axis=1))
                mse_loss = mse + ser_loss.mean()
            else:
                mse_loss = mse
            d_accuracy_fake = 1-torch.round(torch.sigmoid(self.discriminator(generated))).mean()
            d_accuracy_real = (torch.round(torch.sigmoid(self.discriminator(RP)))).mean()
            d_acc = (d_accuracy_fake + d_accuracy_real) / 2        
            self.log("val_mse", mse_loss, prog_bar=True, sync_dist=True)
            self.log("val_D_acc", d_acc, prog_bar=True, sync_dist=True)
            gen = unnormalize_hrrp(generated.cpu().numpy(), self.min_rp, self.max_rp)
            B = gen.shape[0]
            take = min(B, self.config.get("val_metrics_max_samples", 1))
            sel = torch.arange(take, device=self.device)
            self.compute_loss("val", idx, gen, sel)
            if self.test_inf("val"):
                RP, generated = plot_generated_true(RP[:self.bs], generated[:self.bs].cpu(), scal[:self.bs].cpu().numpy(), self.save_path, self.min_rp, self.max_rp, "val", self.current_epoch)
