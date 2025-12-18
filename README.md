# ICASSP HRRP Training

Cleaned-up DDPM / GAN training code for HRRP radar profiles.  
Executable code lives in `src/icassp`, configs in `configs/`.

<img src="assets/ship_hrrp.png" alt="Generated samples overview" height="600" />

## Quick requirements
- Python ≥ 3.9
- PyTorch (CPU or CUDA, matching your GPU if available)
- The generated demo set (`data/ship_hrrp.pt`).

## Installation
```bash
cd github_repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # installs the icassp package from src/
```

## Run a training
```bash
python -m icassp.train \
  --config configs/gan_scalars_serloss.yaml \
  --data data/ship_hrrp.pt \
  --seed 42 \
  --num-workers 0
```
Useful flags:
- `--num-workers`: dataloader workers (set 0 on small CPUs).
- `--skip-eval`: skip test metrics if no test split.
- `--fast-dev-run`: quick pipeline sanity-check (1 train/val/test batch).

## Quick sanity check
```bash
python -m icassp.train \
  --config configs/gan_scalars_serloss.yaml \
  --data data/ship_hrrp.pt \
  --seed 0 \
  --num-workers 0 \
  --skip-eval \
  --fast-dev-run
```

Artifacts (checkpoints, figures, TensorBoard logs) are written under `results/` following the `figure_path` in the config.

## Layout
- `src/icassp/`: models (DDPM, GAN), dataset, utils, training script (`train.py`).
- `configs/`: all YAML configs.
- `requirements.txt`: Python deps.
- `.gitignore`: ignores caches, venvs, and training outputs.

## Notes
- Intended for a quick demo run on the 128 generated samples; no multi-GPU setup required.
- Final metrics rely on `compute_metrics` in `icassp.utils`. If there is no test split (`test_idx` empty), use `--skip-eval`.
