# %% ------------------------------------------------------------ 2. DATA POOL
import datasets as hfds

POOL_DIR = BASE / CFG["POOL_NAME"]
def _to_pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict) and x.get("bytes") is not None:
        return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    raise ValueError(f"cannot decode image of type {type(x)}")

def _as_list(x):
    if isinstance(x, (list, tuple)):
        return [str(a) for a in x]
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return [str(a) for a in v] if isinstance(v, (list, tuple)) else [x]
        except Exception:
            return [x]
    return [str(x)]

def _stream(repo, split, config=None, shuffle=True, buf=300):
    kw = dict(split=split, streaming=True)
    ds = hfds.load_dataset(repo, config, **kw) if config else hfds.load_dataset(repo, **kw)
    return ds.shuffle(seed=CFG["SEED"], buffer_size=buf) if shuffle else ds

# each iterator yields (image_key, image_or_path, question, [answers], extra_dict)
def iter_vqav2(n):
    from huggingface_hub import hf_hub_download
    import pyarrow as pa, pyarrow.parquet as pq
    rng = random.Random(CFG["SEED"])
    shards = rng.sample(range(111), CFG["VQAV2_SHARDS"])
    per = math.ceil(n / len(shards))
    for s in shards:
        p = hf_hub_download("HuggingFaceM4/VQAv2", f"default/vq_av2-train-{s:05d}-of-00111.parquet",
                            repo_type="dataset", revision="refs/convert/parquet")
        tbl = pq.ParquetFile(p).read()
        idx = rng.sample(range(tbl.num_rows), min(per, tbl.num_rows))
        for row in tbl.take(pa.array(idx)).to_pylist():
            yield (row["image_id"], row["image"], row["question"],
                   [a["answer"] for a in row["answers"]], dict(answer_type=row.get("answer_type")))

def iter_gqa(n):
    split, n_img = CFG["GQA_SPLIT"], CFG["GQA_IMAGES"]
    repo = "lmms-lab-encoder/GQA"
    paths = {}
    for ex in hfds.load_dataset(repo, f"{split}_balanced_images", split=split, streaming=True):
        pth = POOL_DIR / "images" / f"gqa_{ex['id']}.jpg"
        if not pth.exists():
            _save_img(_to_pil(ex["image"]), pth, 1024)
        paths[str(ex["id"])] = pth
        if len(paths) >= n_img:
            break
    ins = hfds.load_dataset(repo, f"{split}_balanced_instructions", split=split)
    ins = ins.filter(lambda e: str(e["imageId"]) in paths)
    order = list(range(len(ins))); random.Random(CFG["SEED"]).shuffle(order)
    for i in order[:n]:
        e = ins[i]
        yield (e["imageId"], paths[str(e["imageId"])], e["question"], [e["answer"]], dict(types=str(e["types"])[:80]))

def iter_textvqa(n):
    for ex in _stream("lmms-lab-encoder/textvqa", "train"):
        yield (ex["image_id"], ex["image"], ex["question"], _as_list(ex["answers"]), {})

def iter_docvqa(n):
    for ex in _stream("lmms-lab-encoder/DocVQA", "validation", config="DocVQA", buf=100):
        yield (ex["docId"], ex["image"], ex["question"], _as_list(ex["answers"]), {})

def iter_aokvqa(n):
    for ex in _stream("HuggingFaceM4/A-OKVQA", "train"):
        yield (ex["question_id"], ex["image"], ex["question"], _as_list(ex["direct_answers"]), {})

def iter_vsr(n):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("juletxara/visual-spatial-reasoning", "random/visual-spatial-reasoning-train-00000-of-00002.parquet",
                        repo_type="dataset", revision="refs/convert/parquet")
    tbl = pq.read_table(p, columns=["image", "image_link", "caption", "label"]).to_pylist()
    random.Random(CFG["SEED"]).shuffle(tbl)
    for r in tbl:
        q = f'Is the following statement true or false? "{r["caption"]}"\nAnswer with True or False.'
        yield (r["image_link"], r["image"], q, ["true" if int(r["label"]) == 1 else "false"], {"raw_prompt": True})

def iter_clevr(n):
    if CFG["CLEVR_TRY_ORIGINAL"]:
        try:
            for ex in _stream("HuggingFaceM4/clevr", "train"):
                yield (ex.get("image_id", ex.get("question_index", random.random())), ex["image"],
                       ex["question"], [str(ex["answer"])], {})
            return
        except Exception as e:
            log("original CLEVR loader failed (script-based dataset):", repr(e)[:120], "-> using parquet fallback")
    for i, ex in enumerate(_stream("MMInstruction/Clevr_CoGenT_ValA", "train")):
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", ex["solution"])
        yield (f"clevr{i}", ex["image"], ex["problem"], [m.group(1) if m else ex["solution"]], {})

LOADERS = dict(vqav2=iter_vqav2, gqa=iter_gqa, textvqa=iter_textvqa, docvqa=iter_docvqa,
               aokvqa=iter_aokvqa, vsr=iter_vsr, clevr=iter_clevr)

def _save_img(img, path, max_side):
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        img = img.resize((max(1, round(w * max_side / s)), max(1, round(h * max_side / s))), Image.BICUBIC)
    img.save(path, quality=92)

def build_pool():
    for ds_name, n in CFG["N_POOL"].items():
        done, meta = POOL_DIR / f"{ds_name}.done", POOL_DIR / f"{ds_name}.jsonl"
        if done.exists():
            log(f"[pool] {ds_name}: cached"); continue
        log(f"[pool] building {ds_name} (n={n})")
        cnt, t0 = 0, time.time()
        with open(meta, "w") as fo:
            try:
                for key, img, q, ans, extra in tqdm(LOADERS[ds_name](n), total=n, desc=ds_name):
                    if cnt >= n or STOP.is_set():
                        break
                    try:
                        uid = f"{ds_name}_{cnt}"
                        rel = f"images/{uid}.jpg"
                        pil = Image.open(img) if isinstance(img, (str, Path)) else _to_pil(img)
                        _save_img(pil, POOL_DIR / rel, 1600 if ds_name == "docvqa" else 1024)
                        w, h = Image.open(POOL_DIR / rel).size
                        fo.write(json.dumps(dict(ds=ds_name, uid=uid, img=rel, W=w, H=h, split=split_of(f"{ds_name}:{key}"),
                                                 question=q, answers=ans, extra=extra)) + "\n")
                        cnt += 1
                    except Exception as e:
                        FAILS.append(("pool", ds_name, repr(e), traceback.format_exc()))
            except Exception as e:
                FAILS.append(("pool-loader", ds_name, repr(e), traceback.format_exc()))
                log(f"[pool] {ds_name} loader stopped early: {repr(e)[:150]}  (kept {cnt} samples)")
        if cnt > 0:
            done.write_text(str(cnt))
        log(f"[pool] {ds_name}: {cnt} samples in {time.time()-t0:.0f}s")

def load_pool():
    recs = []
    for ds_name in CFG["N_POOL"]:
        p = POOL_DIR / f"{ds_name}.jsonl"
        if p.exists():
            recs += [json.loads(l) for l in open(p)]
    return recs

if "pool" in CFG["STAGES"]:
    build_pool()
POOL = load_pool()
BY_SPLIT = defaultdict(list)
for r in POOL:
    BY_SPLIT[r["split"]].append(r)
log("pool:", {k: len(v) for k, v in BY_SPLIT.items()}, "| per dataset:",
    {d: sum(1 for r in POOL if r["ds"] == d) for d in CFG["N_POOL"]})

def interleave_by_ds(recs, limit, seed=0):
    """Round-robin across datasets so a cap keeps all sources represented."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in recs: groups[r["ds"]].append(r)
    for g in groups.values(): rng.shuffle(g)
    out, i = [], 0
    while len(out) < limit and any(groups.values()):
        for g in list(groups.values()):
            if g and len(out) < limit: out.append(g.pop())
    return out

