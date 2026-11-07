"""Thread-safe spy on scaled_dot_product_attention (lets FastV / VisionZip / SparseVLM read the exact q/k the model used)."""
import threading

import torch


class _Stop(Exception):
    pass


# ---- thread-safe spy on scaled_dot_product_attention: lets FastV read the EXACT post-RoPE q/k that the real attention used
#      (no re-implementation of RoPE/GQA, so it cannot drift from the model). Installed once; pass-through unless a thread asks.
_TL = threading.local()


def install_attention_spy():
    """Install a pass-through spy on scaled_dot_product_attention (idempotent)."""
    if not getattr(torch.nn.functional.scaled_dot_product_attention, "_qease_spy", False):
        _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
        def _spy_sdpa(q, k, v, *a, **kw):
            cap = getattr(_TL, "cap", None)
            if cap is not None:
                mode = cap.get("mode", "last_row")
                if mode == "last_row" and "q" not in cap:                      # FastV: first call, last query row only
                    cap["q"], cap["k"] = q[:, :, -1:, :].detach(), k.detach()
                elif mode == "first_full" and "q" not in cap:                  # SparseVLM: first call, all rows
                    cap["q"], cap["k"] = q.detach(), k.detach()
                elif mode == "overwrite":                                      # VisionZip: keep the LAST call (= last vision block)
                    cap["q"], cap["k"] = q.detach(), k.detach()
            return _orig_sdpa(q, k, v, *a, **kw)
        _spy_sdpa._qease_spy = True
        torch.nn.functional.scaled_dot_product_attention = _spy_sdpa
