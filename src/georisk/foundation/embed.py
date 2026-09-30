"""Prithvi-EO-1.0-100M encoder (IBM/NASA, Apache-2.0) -> one 768-d embedding per chip. Research use; needs torch."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"


def _files() -> tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    return tuple(Path(hf_hub_download(REPO, f)) for f in ("config.json", "Prithvi_EO_V1_100M.pt", "prithvi_mae.py"))  # type: ignore[return-value]


class PrithviEmbedder:
    def __init__(self, device: str | None = None) -> None:
        import torch

        cfg_p, w_p, code_p = _files()
        cfg = json.loads(cfg_p.read_text())["pretrained_cfg"]
        spec = importlib.util.spec_from_file_location("prithvi_mae", code_p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["prithvi_mae"] = mod
        spec.loader.exec_module(mod)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 6, 1, 1, 1)
        self.std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 6, 1, 1, 1)
        self.model = mod.PrithviMAE(
            img_size=cfg["img_size"], patch_size=tuple(cfg["patch_size"]), num_frames=1, in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"], encoder_only=True,
        )
        state = torch.load(w_p, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if k.startswith("encoder.") and "pos_embed" not in k}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        bad = [k for k in missing if "pos_embed" not in k]
        if bad or unexpected:
            raise RuntimeError(f"weights did not load cleanly: missing={bad[:3]} unexpected={list(unexpected)[:3]}")
        self.model.to(self.device).eval()

    def embed(self, chips: np.ndarray, batch: int = 16) -> np.ndarray:
        """chips: (N, 6, 224, 224) reflectance x 10000 -> (N, 768): mean of the patch tokens from the encoder."""
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(chips), batch):
                x = torch.from_numpy(chips[i : i + batch].astype("float32")).unsqueeze(2)  # (B, 6, T=1, H, W)
                x = ((x - self.mean) / self.std).to(self.device)
                latent = self.model.encoder.forward_features(x)[-1]  # no masking (the MAE default masks 75% of patches)
                out.append(latent[:, 1:, :].mean(1).float().cpu().numpy())  # drop the cls token, pool the patches
        return np.concatenate(out)
