"""Model loading, sanity checks and self-tests of the baseline ports."""
import traceback

import torch
from qease import state
from qease.generation import norm_ans
from qease.utils import log
from qease.vlm.helpers import js_div
from qease.vlm.runner import VLMRunner


def run_load():
    """Stage 'load': load the VLM on every GPU, run the sanity checks and the self-tests of the ported baselines."""
    if not state.RUNNERS:
        for d in state.DEVICES:
            log(f"loading {state.CFG['MODEL_ID']} on {d} ...")
            state.RUNNERS.append(VLMRunner(d))
    r0 = state.RUNNERS[0]
    log(f"decoder params (non-embedding): {r0.llm_params/1e9:.2f}B | image token id {r0.img_tok}")

    # --- sanity: attention-masking == physical pruning == full model when everything is kept
    _rec = next(r for r in state.POOL if r["ds"] in ("textvqa", "vqav2", "gqa", "aokvqa", "vsr", "clevr", "docvqa"))
    with torch.inference_mode():
        _ctx = r0.prepare(_rec)
        _lp_full = r0.logp_full(_ctx)
        _G = _ctx["G"]
        _gk = torch.zeros(1, _G, dtype=torch.bool, device=r0.device); _gk[0, torch.randperm(_G)[: max(1, _G // 4)]] = True
        _lp_mask = r0.logp_for_token_masks(_ctx, r0.group_to_token_mask(_ctx, _gk))[0]
        _lp_phys = r0.prefill_pruned(_ctx, _gk[0])
        _ref = r0.model(**_ctx["inputs"], use_cache=False).logits[0, -1].float().log_softmax(-1)
        log(f"SANITY  G={_G} tokens={_ctx['n_vis']} grid={_ctx['H']}x{_ctx['W']} win={_ctx['win']} L={_ctx['L']}")
        log(f"  JS(model.forward, my-full-keep) = {js_div(_ref, _lp_full[None]).item():.2e}  (should be ~0)")
        log(f"  JS(masked-attention, physical-pruning) = {js_div(_lp_mask, _lp_phys[None]).item():.2e}  (should be ~0)")
        log(f"  JS(full, keep 25% groups) = {js_div(_lp_full, _lp_mask[None]).item():.4f}")
        log(f"  full answer (greedy): '{r0.generate(_ctx, None, 6)}'  | pruned answer: '{r0.generate(_ctx, _gk[0], 6)}'")
        # --- FastV self-tests: the baseline is only used if it reproduces the full model exactly when nothing is pruned
        if state.CFG["FASTV"]:
            try:
                _K = state.CFG["FASTV_K"]
                _h, _pe, _sc = r0.fastv_prefix(_ctx, _K)
                _keep_all = torch.ones(_ctx["L"], dtype=torch.bool, device=r0.device)
                _d_fv = js_div(_lp_full, r0.fastv_logp(_ctx, _h, _pe, _keep_all, _K)[None]).item()
                _a_ref, _a_fv = r0.generate(_ctx, None, 6), r0.fastv_generate(_ctx, _keep_all, _K, 6)
                _sc_ok = bool(torch.isfinite(_sc).all()) and float(_sc.sum()) > 0 and _sc.numel() == _ctx["n_vis"]
                log(f"FASTV self-test: JS(no pruning, manual layer loop vs model)={_d_fv:.2e} (need <5e-3) | scores finite/shape ok={_sc_ok} "
                    f"| sum of visual attention={float(_sc.sum()):.3f} | greedy answers equal: '{_a_ref}' vs '{_a_fv}'")
                state.FASTV_OK = (_d_fv < 5e-3) and _sc_ok and (norm_ans(_a_ref) == norm_ans(_a_fv) if True else _a_ref.strip().lower() == _a_fv.strip().lower())
                if not state.FASTV_OK: log("FASTV self-test FAILED -> FastV baseline disabled (see numbers above).")
            except Exception as e:
                state.FAILS.append(("fastv-selftest", "-", repr(e)[:200], traceback.format_exc()))
                log(f"FASTV self-test crashed -> baseline disabled: {repr(e)[:200]}")
        log(f"FASTV_OK={state.FASTV_OK}")
        # --- VisionZip self-test: attention capture shape/normalisation, identity at full budget, token accounting
        if state.CFG["VISIONZIP"]:
            try:
                r0.vz_on = True
                _c2 = r0.prepare(_rec)
                assert _c2["vz"] is not None, _c2["vz_err"]
                _vz, _n = _c2["vz"], _c2["n_vis"]
                _ok_norm = abs(_vz["att_total"] - _vz["T"]) < 0.02 * _vz["T"]            # every softmax row sums to 1 -> total mass = #patches
                _e_all, _k_all = r0.vz_embed(_c2, _n)
                _d_id = js_div(r0.logp_full(_c2), r0.logp_prefill(_c2, _e_all, _k_all)[None]).item()
                _tot = max(2, _n // 4)
                _e_h, _k_h = r0.vz_embed(_c2, _tot)
                _lp_h = r0.logp_prefill(_c2, _e_h, _k_h)
                _cnt_ok = int(_k_h[_c2["img_idx"]].sum()) == _tot
                log(f"VISIONZIP self-test: patches={_vz['T']} tokens={_n} attention-mass ok={_ok_norm} | JS(full budget vs model)={_d_id:.2e} (need <5e-3) "
                    f"| kept {int(_k_h[_c2['img_idx']].sum())}/{_tot} tokens | finite={bool(torch.isfinite(_lp_h).all())}")
                state.VZ_OK = bool(_ok_norm and _d_id < 5e-3 and _cnt_ok and torch.isfinite(_lp_h).all())
            except Exception as e:
                state.FAILS.append(("visionzip-selftest", "-", repr(e)[:200], traceback.format_exc()))
                log(f"VISIONZIP self-test crashed -> baseline disabled: {repr(e)[:200]}")
            for _r in state.RUNNERS: _r.vz_on = bool(state.VZ_OK and any(s in state.CFG["STAGES"] for s in ("eval", "gen")))
            log(f"VZ_OK={state.VZ_OK}")
        # --- SparseVLM self-test: engine reproduces the model with pruning off; plan replay reproduces the pruned run; sane token schedule
        if state.CFG["SPARSEVLM"]:
            try:
                _lp0 = r0.sparsevlm_run(_ctx, None)[0]
                _d0 = js_div(_lp_full, _lp0[None]).item()
                _ret = state.CFG["SPARSE_RETAIN"][0]
                _lp1, _st, _plan = r0.sparsevlm_run(_ctx, _ret)
                _lp2 = r0.sparsevlm_run(_ctx, _ret, plan=_plan)[0]
                _d2 = js_div(_lp1, _lp2[None]).item()
                _nl = len(r0.decoder.layers)
                _avg = sum(_st["vis_per_layer"]) / _nl / _ctx["n_vis"]
                log(f"SPARSEVLM self-test: JS(no pruning vs model)={_d0:.2e} (need <5e-3) | JS(plan replay vs live run)={_d2:.2e} (need <5e-3) "
                    f"| stages={len(_plan)} | avg visual tokens/layer={_avg:.3f} of N (official {_ret}/576={_ret/576:.3f}) | per-layer visual tokens={_st['vis_per_layer']}")
                state.SV_OK = bool(_d0 < 5e-3 and _d2 < 5e-3 and len(_plan) == 3 and torch.isfinite(_lp1).all() and _st["vis_per_layer"][-1] < _ctx["n_vis"])
            except Exception as e:
                state.FAILS.append(("sparsevlm-selftest", "-", repr(e)[:200], traceback.format_exc()))
                log(f"SPARSEVLM self-test crashed -> baseline disabled: {repr(e)[:200]}")
            log(f"SV_OK={state.SV_OK}")
    del _ctx, _lp_full, _lp_mask, _lp_phys, _ref
