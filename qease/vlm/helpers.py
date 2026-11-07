"""Small helpers of the VLM wrapper: image resizing, decoder lookup, JS divergence and the SparseVLM token-merging utilities."""
import math

from PIL import Image
import torch


# ---- SparseVLM v1.0 constants / helpers ported line-by-line from github.com/Gumpest/SparseVLMs
#      (llava/model/language_model/score.py, utils.py::cluster_and_merge/index_points, modelling_sparse_llama.py)
SPARSE_SCHEDULE = {192: [300, 200, 110], 128: [303, 110, 36], 96: [238, 48, 26], 64: [66, 30, 17]}   # tokens kept after each stage, for 576 visual tokens


SPARSE_LAYERS_OFFICIAL, SPARSE_NLAYERS_OFFICIAL, SPARSE_NTOK_OFFICIAL = [2, 6, 15], 32, 576


def sv_index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def sv_cluster_and_merge(x, cluster_num):
    """density-peak clustering + mean merge (official `cluster_and_merge`, einops replaced by indexing; also returns the assignment)."""
    B, N, C = x.shape
    distance = (x.unsqueeze(2) - x.unsqueeze(1)).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)
    dist_nearest, _ = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
    _, index_down = torch.topk(dist * density, k=cluster_num, dim=-1)
    dist_matrix = sv_index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix.argmin(dim=1)
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)       # centres merge to themselves
    token_weight = x.new_ones(B, N, 1)
    idx = idx_cluster + torch.arange(B, device=x.device)[:, None] * cluster_num
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(0, idx.reshape(B * N), token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]
    x_merged = x.new_zeros(B * cluster_num, C)
    x_merged.index_add_(0, idx.reshape(B * N), (x * norm_weight).reshape(B * N, C).type(x.dtype))
    return x_merged.reshape(B, cluster_num, C), idx_cluster


def smart_resize(img, n_tokens, factor=28):
    """Resize to a multiple of 28 with <= n_tokens merged visual tokens (Qwen2-VL geometry)."""
    max_pixels = n_tokens * factor * factor
    w, h = img.size
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    return img.resize((wb, hb), Image.BICUBIC)


def find_decoder(model):
    for path in ["model.language_model", "language_model", "model.text_model", "text_model", "model.model", "model"]:
        obj, ok = model, True
        for a in path.split("."):
            if hasattr(obj, a): obj = getattr(obj, a)
            else: ok = False; break
        if ok and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
            return obj
    raise RuntimeError("could not locate the text decoder of this VLM")


def js_div(logp_ref, logp):
    """Jensen-Shannon divergence (nats) between rows of log-probs. ref: (V,) or (B,V); logp: (B,V)."""
    p = logp_ref.exp(); q = logp.exp(); m = 0.5 * (p + q)
    lm = torch.log(m.clamp_min(1e-30))
    return 0.5 * (p * (logp_ref - lm)).sum(-1) + 0.5 * (q * (logp - lm)).sum(-1)
