"""Loads weights sent by Theo and converts them to normal state_dict format.

For it to work, you need to place the script in the base dir and
create a file in enc/component/coolchic.py with the following content:
```
from coolchic.enc.component.coolchic import CoolChicEncoderParameter
```

This script will save the converted weights in the `nocc_weights` directory.
"""

from pathlib import Path

import torch

from coolchic.enc.component.nlcoolchic import LatentFreeCoolChicEncoder
from coolchic.hypernet.hypernet import LatentHyperNet, NOWholeNet
from coolchic.utils.paths import COOLCHIC_REPO_ROOT, RESULTS_DIR
from coolchic.utils.types import HypernetRunConfig, load_config
from scripts.gen_config_lambdas import LMBDA_TO_CONFIG_NUM


def find_dir_name(file_name: str) -> str:
    # File is called something like nocc-lmbda_0.0002.pt
    lmbda = file_name.split("_")[-1]
    if lmbda.endswith(".pth"):
        lmbda = lmbda[:-4]  # Remove .pth suffix
    lmbda = float(lmbda)
    if lmbda not in LMBDA_TO_CONFIG_NUM:
        print(f"Warning: Lambda {lmbda} not found in LMBDA_TO_CONFIG_NUM. Skipping.")
        return ""
    return f"config_{LMBDA_TO_CONFIG_NUM[lmbda]}"


w_dir = Path("updated-nocc")
w_save_dir = RESULTS_DIR / "exps/copied/nocc_weights"
w_save_dir.mkdir(exist_ok=True, parents=True)

# Create net to load the weights into.
cfg_path = COOLCHIC_REPO_ROOT / "cfg/exps/no-cchic/longer-training/config_02.yaml"
cfg = load_config(cfg_path, HypernetRunConfig)
net = NOWholeNet(cfg.hypernet_cfg)

for w_path in w_dir.glob("*.pth"):
    weights = torch.load(w_path, map_location="cpu")

    analysis_weights = weights["analysis"]
    an_trf = LatentHyperNet()
    an_trf.load_state_dict(state_dict=analysis_weights)

    cc_encoder_weights = weights["cc_enc"]
    cc_enc = LatentFreeCoolChicEncoder(net.mean_decoder.param)
    cc_encoder_weights = {
        k: v for k, v in cc_encoder_weights.items() if "latent_grids" not in k
    }
    cc_enc.load_state_dict(state_dict=cc_encoder_weights)

    net.encoder = an_trf
    net.mean_decoder = cc_enc

    dir_name = find_dir_name(w_path.name)
    if not dir_name:
        continue  # Skip if lambda is not found in LMBDA_TO_CONFIG_NUM
    lmbda_save_dir = w_save_dir / dir_name
    lmbda_save_dir.mkdir(exist_ok=True, parents=True)
    torch.save(net.state_dict(), lmbda_save_dir / "model.pt")
