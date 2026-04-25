from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn


def _infer_hook_module(model: nn.Module, hook_module_name: Optional[str] = None) -> nn.Module:
    if hook_module_name is not None:
        mod = dict(model.named_modules()).get(hook_module_name, None)
        if mod is None:
            raise ValueError(f"hook_module_name='{hook_module_name}' not found in model.")
        return mod

    # sensible defaults by architecture
    for name in ["blocks", "encoder", "head"]:
        if hasattr(model, name):
            return getattr(model, name)
    # fallback: whole model (not ideal)
    return model


@torch.no_grad()
def extract_hidden_states_from_loader(
    model: nn.Module,
    loader,
    device: str = "cpu",
    hook_module_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      hidden_states: [N,T,H]
      y_true: [N]
      years: [N]
      pids: [N]
    """
    model.eval()
    model.to(device)

    captured: List[torch.Tensor] = []
    ys, yrs, pids_all = [], [], []

    hook_module = _infer_hook_module(model, hook_module_name)

    def _hook_fn(_module, _inp, out):
        # Expected outputs shaped like [B,T,H] or [B,H]
        if isinstance(out, tuple):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            return

        if out.ndim == 2:          # [B,H] -> create pseudo-time T=1
            out_h = out.unsqueeze(1)
        elif out.ndim == 3:        # [B,T,H]
            out_h = out
        else:
            return

        captured.append(out_h.detach().cpu())

    handle = hook_module.register_forward_hook(_hook_fn)

    try:
        for xb, yb, yrb, pidb in loader:
            xb = xb.to(device, non_blocking=True)

            captured.clear()
            _ = model(xb)

            if len(captured) == 0:
                raise RuntimeError(
                    "No hidden states captured. Try passing hook_module_name explicitly."
                )

            h = captured[-1]  # [B,T,H]
            ys.append(yb.numpy().astype(int))
            yrs.append(yrb.numpy().astype(int))
            pids_all.extend(list(pidb))

            if "all_h" not in locals():
                all_h = [h]
            else:
                all_h.append(h)
    finally:
        handle.remove()

    hidden_states = torch.cat(all_h, dim=0).numpy().astype(np.float32)  # [N,T,H]
    y_true = np.concatenate(ys).astype(int)
    years = np.concatenate(yrs).astype(int)
    pids = np.asarray(pids_all, dtype=str)

    return hidden_states, y_true, years, pids