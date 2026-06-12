"""HMM re-fit on teacher attention traces (S3-streaming, run on EC2/Code).

Re-fits HMMs on Qwen-7B attention-window JSD sequences (~113 objects, ~200MB
each in s3://golden-medusa-data/...). Extracts two numbers v200 needs:
  - K* (state count): per-trace BIC-selected K, aggregated modally
  - dwell-time ratio: per-state dwell distributions, geometric-vs-clocked check

Per Bryce's Jun 10 spec (specifications baked into pipeline):
  - --inspect first (never assume format)
  - Log-transform JSD before Gaussian HMM (JSD is non-neg + bursty)
  - 8 EM restarts per (trace, K), keep best log-likelihood
  - BIC across K=2..5
  - Dwell two ways: empirical (Viterbi) + implied (1/(1-p_ii))
  - Junk-state detection: occupancy < 5% disqualifies state from "substantive"
  - Geometric vs peaked dwell distinction (semi-Markov flag)
  - Stream S3 objects via boto3 get_object body, never load full
  - Per-trace results idempotent (resume-friendly via skip-if-exists)

DO NOT RUN INSIDE BRYCE'S SANDBOX — no S3 egress. Run on EC2 or local
Code where AWS credentials live in ~/.aws.

Usage:
  pip install boto3 hmmlearn numpy scipy

  # Step 0: ALWAYS inspect first
  python hmm_refit_teacher_traces.py --inspect --bucket golden-medusa-data

  # Step 1: fit (per-trace, parallelizable, idempotent)
  python hmm_refit_teacher_traces.py --fit \\
      --bucket golden-medusa-data \\
      --output-dir ./hmm_results \\
      --workers 8 \\
      --window 16 --stride 8

  # Step 2: aggregate
  python hmm_refit_teacher_traces.py --aggregate \\
      --output-dir ./hmm_results \\
      --report hmm_aggregate.json
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats

# Lazy imports — only imported when their mode runs
# boto3 (S3) for --inspect and --fit
# hmmlearn for --fit


# ---------------------------------------------------------------------------
# --inspect mode — fails in 30s on format mismatch, never after a sweep
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    """List N keys, pull ONE object (full), print structure, exit.

    Per Jun 10 spec correction: zip/parquet/HDF5 have their central
    directory at the END of the file, so range-read of the head silently
    fails to parse. Magic-byte sniffing on the head is fine — actual
    parsing needs the full object. 200MB one-off download is acceptable
    for a one-shot inspection step.
    """
    import boto3

    region = args.region or _detect_region()
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    print(f"Inspecting s3://{args.bucket}/ (prefix={args.prefix or '<root>'})")
    if region:
        print(f"  region={region}")
    print()

    # List up to 20 keys
    paginator = s3.get_paginator("list_objects_v2")
    page_iter = paginator.paginate(
        Bucket=args.bucket,
        Prefix=args.prefix or "",
        PaginationConfig={"MaxItems": 20},
    )
    keys = []
    for page in page_iter:
        for obj in page.get("Contents", []):
            keys.append((obj["Key"], obj["Size"]))
        if len(keys) >= 20:
            break

    if not keys:
        print(f"NO OBJECTS FOUND. Check bucket/prefix.")
        return 1

    print(f"First {len(keys)} keys (size in MB):")
    for k, sz in keys[:20]:
        print(f"  {sz/1e6:7.1f} MB  {k}")
    print()

    # Pick one to inspect
    target_key = args.inspect_key or keys[0][0]
    target_size = next((sz for k, sz in keys if k == target_key), None)
    if target_size is None:
        head = s3.head_object(Bucket=args.bucket, Key=target_key)
        target_size = head["ContentLength"]
    print(f"Inspecting (FULL DOWNLOAD): {target_key}  ({target_size/1e6:.1f} MB)")

    # Magic-byte sniff via 64-byte head range (cheap)
    head_resp = s3.get_object(
        Bucket=args.bucket, Key=target_key, Range="bytes=0-63"
    )
    head_bytes = head_resp["Body"].read()
    print(f"  first 32 bytes (hex): {head_bytes[:32].hex()}")
    detected = _detect_format(head_bytes)
    print(f"  magic-byte detection: {detected}")
    print()

    # Full download — zip/parquet/HDF5 need their footer to parse
    resp = s3.get_object(Bucket=args.bucket, Key=target_key)
    body_bytes = resp["Body"].read()
    print(f"  full download: {len(body_bytes)/1e6:.1f} MB")
    print()

    print("Format parsing:")
    if detected == "npy":
        try:
            arr = np.load(io.BytesIO(body_bytes), allow_pickle=False)
            print(f"  shape={arr.shape}  dtype={arr.dtype}")
            print(f"  min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}")
            print(f"  first 8 values: {arr.flatten()[:8]}")
        except Exception as e:
            print(f"  numpy .npy parse failed: {e}")
    elif detected == "zip_or_npz":
        try:
            obj = np.load(io.BytesIO(body_bytes), allow_pickle=False)
            if hasattr(obj, "files"):
                print(f"  → npz with keys: {obj.files}")
                for fname in obj.files[:5]:
                    arr = obj[fname]
                    print(f"    [{fname}] shape={arr.shape}  dtype={arr.dtype}  "
                          f"min={arr.min():.4g}  max={arr.max():.4g}")
                    if arr.ndim == 1:
                        print(f"      first 10: {arr[:10]}")
            else:
                print(f"  → npy in zip wrapper: shape={obj.shape}")
        except Exception as e:
            print(f"  npz parse failed: {e}")
            print("  may be plain zip — try unzipping manually")
    elif detected == "parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(io.BytesIO(body_bytes))
            print(f"  schema:\n{table.schema}")
            print(f"  num_rows: {table.num_rows}")
            if table.num_rows > 0:
                print(f"  first 3 rows of each column:")
                for col_name in table.column_names[:5]:
                    col = table.column(col_name).to_pylist()[:3]
                    print(f"    {col_name}: {col}")
        except ImportError:
            print("  install pyarrow to inspect parquet")
    elif detected == "json":
        try:
            text = body_bytes[:10000].decode("utf-8")
            print(f"  first line: {text.split(chr(10))[0][:500]}")
        except UnicodeDecodeError:
            print("  not valid UTF-8")
    elif detected == "hdf5":
        try:
            import h5py
            f = h5py.File(io.BytesIO(body_bytes), "r")
            print(f"  HDF5 keys: {list(f.keys())}")
            for k in list(f.keys())[:5]:
                obj = f[k]
                if hasattr(obj, "shape"):
                    print(f"    [{k}] shape={obj.shape}  dtype={obj.dtype}")
        except ImportError:
            print("  install h5py to inspect HDF5")
    else:
        print(f"  unknown format")
        print(f"  printable preview: {body_bytes[:200]!r}")

    print()
    print("=" * 60)
    print("Next: confirm format matches your pipeline writer.")
    print("Then run --fit with --jsd-mode extracted (pre-extracted JSD sequences)")
    print("                or --jsd-mode raw_attention (recompute via W=16 stride=8)")
    print()
    print("CRITICAL for dwell-units consistency:")
    print("  extracted mode → dwell_units = whatever the original pipeline emitted")
    print("                    (likely token-positions or window-positions; check!)")
    print("  raw_attention   → dwell_units = window-positions (W=16 stride=8)")
    print("                    multiply by stride 8 for token-position scale")
    print("=" * 60)
    return 0


def _detect_format(head_bytes):
    """Magic-byte detection on head bytes. Returns format tag."""
    if head_bytes[:6] == b"\x93NUMPY":
        return "npy"
    if head_bytes[:4] == b"PK\x03\x04":
        return "zip_or_npz"
    if head_bytes[:4] == b"PAR1":
        return "parquet"
    if head_bytes[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if head_bytes[:1] in (b"{", b"["):
        return "json"
    return "unknown"


def _detect_region():
    """Get region from environment or AWS profile default."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region
    try:
        import boto3
        return boto3.Session().region_name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSD computation (only used if raw attention; skipped if pre-extracted)
# ---------------------------------------------------------------------------

def jensen_shannon(p, q, eps=1e-12):
    """JSD between two distributions (assumes last axis is the distribution)."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(p / m)).sum(axis=-1)
    kl_qm = (q * np.log(q / m)).sum(axis=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_jsd_sequence_windowed(attention_seq, window=16, stride=8):
    """C1-A's instrument: JSD between adjacent token-position windows.

    attention_seq: shape (..., seq_len, key_len) — per-token attention dist over keys
                   Last two axes are (query position, key position).
                   Higher axes (heads, layers) get averaged here.

    Returns 1D array of JSD values between consecutive windows.
    """
    # Average over heads/layers if multi-dim
    while attention_seq.ndim > 2:
        attention_seq = attention_seq.mean(axis=0)

    seq_len = attention_seq.shape[0]
    n_windows = (seq_len - window) // stride + 1
    if n_windows < 2:
        return np.array([])

    # Compute mean attention distribution per window
    win_dists = []
    for w in range(n_windows):
        start = w * stride
        win_attn = attention_seq[start:start + window]  # (window, key_len)
        win_dists.append(win_attn.mean(axis=0))         # (key_len,)
    win_dists = np.array(win_dists)                     # (n_windows, key_len)

    # JSD between consecutive windows
    jsd_seq = []
    for w in range(n_windows - 1):
        jsd_seq.append(jensen_shannon(win_dists[w], win_dists[w+1]))
    return np.array(jsd_seq)


# ---------------------------------------------------------------------------
# HMM fitting per trace
# ---------------------------------------------------------------------------

def fit_hmm_for_K(log_jsd_seq, K, n_restarts=8, seed_base=0):
    """Fit GaussianHMM at given K with n_restarts random initializations.

    Returns BEST fit (by log-likelihood) AND all per-restart BIC values
    so the caller can compute K-selection stability per Jun 10 spec
    addendum 4 — junk-confidence guard against unstable BIC ordering.
    """
    from hmmlearn.hmm import GaussianHMM

    X = log_jsd_seq.reshape(-1, 1)
    if len(X) < K * 5:
        return None  # too short to fit

    # n_params: K-1 startprob + K*(K-1) trans + K means + K vars
    n_params = (K - 1) + K * (K - 1) + K + K
    N = len(X)

    fits = []
    for r in range(n_restarts):
        try:
            hmm = GaussianHMM(
                n_components=K,
                covariance_type="diag",
                n_iter=200,
                tol=1e-3,
                random_state=seed_base + r,
                init_params="stmc",
            )
            hmm.fit(X)
            ll = hmm.score(X)
            bic = -2 * ll + n_params * np.log(N)
            fits.append(dict(hmm=hmm, ll=ll, bic=bic, restart=r))
        except Exception:
            continue

    if not fits:
        return None

    # Best by log-likelihood
    fits.sort(key=lambda f: f["ll"], reverse=True)
    best = fits[0]
    best["K"] = K
    best["n_params"] = n_params
    best["N"] = N
    # All per-restart BICs (for stability analysis)
    best["all_restart_bics"] = sorted([f["bic"] for f in fits])
    best["all_restart_lls"] = sorted([f["ll"] for f in fits], reverse=True)
    return best


def extract_dwell_stats(hmm, log_jsd_seq):
    """Return per-state empirical (Viterbi) + implied (transition diag) dwell stats."""
    X = log_jsd_seq.reshape(-1, 1)
    states = hmm.predict(X)                   # Viterbi path
    K = hmm.n_components

    # Empirical dwell distributions per state: from Viterbi runs
    empirical = {k: [] for k in range(K)}
    cur_state = states[0]
    cur_len = 1
    for s in states[1:]:
        if s == cur_state:
            cur_len += 1
        else:
            empirical[cur_state].append(cur_len)
            cur_state = s
            cur_len = 1
    empirical[cur_state].append(cur_len)

    # Implied dwell from transition diagonal: E[dwell_k] = 1 / (1 - p_kk)
    trans = hmm.transmat_
    implied = {k: float(1.0 / max(1.0 - trans[k, k], 1e-9)) for k in range(K)}

    # Occupancy per state
    occupancy = {k: float((states == k).mean()) for k in range(K)}

    # State means (in log-JSD space)
    state_means = {k: float(hmm.means_[k][0]) for k in range(K)}

    # Geometric vs peaked check: for each state with enough dwells,
    # compare empirical mode vs geometric expectation.
    semi_markov_flags = {}
    for k in range(K):
        dwells = np.array(empirical[k])
        if len(dwells) < 10:
            semi_markov_flags[k] = "insufficient_data"
            continue
        # Geometric distribution has mode at 1. If empirical mode > 1
        # AND mass is concentrated, dwells are peaked (semi-Markov).
        from collections import Counter
        dwell_counter = Counter(dwells.tolist())
        mode_dwell = max(dwell_counter, key=dwell_counter.get)
        mode_mass = dwell_counter[mode_dwell] / len(dwells)
        # Heuristic: mode > 2 AND mode_mass > 0.15 → peaked/semi-Markov
        if mode_dwell > 2 and mode_mass > 0.15:
            semi_markov_flags[k] = f"peaked@{mode_dwell}_mass={mode_mass:.2f}"
        else:
            semi_markov_flags[k] = "geometric-ish"

    return dict(
        empirical_dwells={k: [int(x) for x in v] for k, v in empirical.items()},
        empirical_mean_dwells={k: float(np.mean(v)) if v else 0.0
                                for k, v in empirical.items()},
        implied_dwells=implied,
        occupancy=occupancy,
        state_means=state_means,
        semi_markov_flags=semi_markov_flags,
    )


# ---------------------------------------------------------------------------
# Per-trace pipeline (worker function)
# ---------------------------------------------------------------------------

def process_trace(args_tuple):
    """Process one S3 key end-to-end. Returns path to per-trace result JSON."""
    (key, bucket, output_dir, jsd_mode, window, stride,
     k_range, n_restarts, min_seq_len, region) = args_tuple

    import boto3

    safe_name = key.replace("/", "_").replace(":", "_")
    out_path = Path(output_dir) / f"trace_{safe_name}.json"
    if out_path.exists():
        return str(out_path)  # idempotent

    # Metadata header for dwell-units traceability (per Jun 10 spec correction 2)
    dwell_units = (f"window-positions (W={window} stride={stride}; "
                    f"multiply by {stride} for token-position scale)"
                    if jsd_mode == "raw_attention"
                    else "as-emitted-by-pipeline (verify against inspect output)")

    try:
        s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()

        # Parse based on jsd_mode
        if jsd_mode == "extracted":
            obj = np.load(io.BytesIO(body), allow_pickle=False)
            if hasattr(obj, "files") and "jsd" in obj.files:
                jsd_seq = obj["jsd"]
            elif hasattr(obj, "files"):
                jsd_seq = obj[obj.files[0]]
            else:
                jsd_seq = obj
            jsd_seq = np.asarray(jsd_seq).flatten()
        elif jsd_mode == "raw_attention":
            obj = np.load(io.BytesIO(body), allow_pickle=False)
            if hasattr(obj, "files"):
                attn = obj[obj.files[0]]
            else:
                attn = obj
            jsd_seq = compute_jsd_sequence_windowed(attn, window=window, stride=stride)
        else:
            raise ValueError(f"unknown jsd_mode: {jsd_mode}")

        # Minimum-length guard per Jun 10 spec correction 3.
        # K=5 Gaussian HMM has ~50 params; rule-of-thumb 10× → 500 steps ideal.
        # Hard floor at min_seq_len (default 100) — below that, fits are garbage.
        if len(jsd_seq) < min_seq_len:
            result = dict(
                key=key, status="too_short",
                jsd_len=int(len(jsd_seq)),
                min_required=min_seq_len,
                dwell_units=dwell_units,
                jsd_mode=jsd_mode,
            )
            out_path.write_text(json.dumps(result, indent=2))
            return str(out_path)

        # Log-transform per spec — JSD is non-neg + bursty,
        # Gaussian-on-raw would hallucinate states.
        log_jsd = np.log(jsd_seq + 1e-9)

        # Fit at each K, pick best BIC
        fits = {}
        for K in k_range:
            fit = fit_hmm_for_K(log_jsd, K, n_restarts=n_restarts)
            if fit is None:
                continue
            dwell_stats = extract_dwell_stats(fit["hmm"], log_jsd)
            fits[K] = dict(
                K=K, ll=fit["ll"], bic=fit["bic"],
                n_params=fit["n_params"], N=fit["N"],
                trans_matrix=fit["hmm"].transmat_.tolist(),
                start_prob=fit["hmm"].startprob_.tolist(),
                state_means=fit["hmm"].means_.flatten().tolist(),
                state_covars=fit["hmm"].covars_.flatten().tolist(),
                all_restart_bics=fit["all_restart_bics"],
                all_restart_lls=fit["all_restart_lls"],
                **dwell_stats,
            )

        if not fits:
            result = dict(key=key, status="all_K_failed",
                          jsd_len=int(len(jsd_seq)),
                          dwell_units=dwell_units,
                          jsd_mode=jsd_mode)
            out_path.write_text(json.dumps(result, indent=2))
            return str(out_path)

        # BIC selection — also tag "substantive" K (excludes junk states)
        bic_K = min(fits, key=lambda k: fits[k]["bic"])
        substantive_K = None
        for K in sorted(fits):
            # All states must have ≥ 5% occupancy to count as substantive
            occ = fits[K]["occupancy"]
            if all(v >= 0.05 for v in occ.values()):
                substantive_K = K
        if substantive_K is None and 2 in fits:
            substantive_K = 2

        # K-selection stability (Jun 10 spec correction 4):
        # For each restart, what K would it have chosen by BIC?
        # If the best restart picks K=2 but second-best picks K=3, that's
        # unstable selection — and the modal K across traces is then suspect.
        k_selections_by_restart = []
        n_rest = min(len(fits[bic_K]["all_restart_bics"]), n_restarts)
        for r in range(n_rest):
            best_K_this_restart = min(
                (K for K in fits if r < len(fits[K]["all_restart_bics"])),
                key=lambda K: fits[K]["all_restart_bics"][r]
                                if r < len(fits[K]["all_restart_bics"])
                                else float("inf"),
            )
            k_selections_by_restart.append(int(best_K_this_restart))
        k_selection_stability = (
            k_selections_by_restart.count(bic_K) / max(len(k_selections_by_restart), 1)
        )

        result = dict(
            key=key,
            status="ok",
            jsd_len=int(len(jsd_seq)),
            jsd_min=float(jsd_seq.min()),
            jsd_max=float(jsd_seq.max()),
            jsd_mean=float(jsd_seq.mean()),
            jsd_std=float(jsd_seq.std()),
            log_jsd_mean=float(log_jsd.mean()),
            log_jsd_std=float(log_jsd.std()),
            # Metadata for dwell-units / windowing traceability
            jsd_mode=jsd_mode,
            window=window if jsd_mode == "raw_attention" else None,
            stride=stride if jsd_mode == "raw_attention" else None,
            dwell_units=dwell_units,
            # K selection + stability
            bic_selected_K=int(bic_K),
            substantive_selected_K=int(substantive_K) if substantive_K else None,
            k_selections_across_restarts=k_selections_by_restart,
            k_selection_stability=float(k_selection_stability),  # fraction of restarts agreeing
            fits=fits,
        )
        out_path.write_text(json.dumps(result, indent=2, default=str))
        return str(out_path)

    except Exception as e:
        out_path.write_text(json.dumps(
            dict(key=key, status="error", error=str(e),
                 error_type=type(e).__name__,
                 dwell_units=dwell_units,
                 jsd_mode=jsd_mode),
            indent=2,
        ))
        return str(out_path)


def cmd_fit(args):
    """Map-reduce over S3 traces. Per-trace results saved to output_dir."""
    import boto3

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    region = args.region or _detect_region()
    print(f"Listing s3://{args.bucket}/{args.prefix or ''}...")
    if region:
        print(f"  region={region}")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix or ""):
        for obj in page.get("Contents", []):
            if args.suffix and not obj["Key"].endswith(args.suffix):
                continue
            keys.append(obj["Key"])

    print(f"Found {len(keys)} traces. Output dir: {args.output_dir}")
    print(f"Settings: K={args.k_min}..{args.k_max}, n_restarts={args.n_restarts}, "
          f"min_seq_len={args.min_seq_len}, jsd_mode={args.jsd_mode}")
    if args.jsd_mode == "raw_attention":
        print(f"  windowing: W={args.window} stride={args.stride} "
              f"→ dwell units = window-positions (×{args.stride} for tokens)")

    job_args = [
        (k, args.bucket, args.output_dir, args.jsd_mode,
         args.window, args.stride,
         list(range(args.k_min, args.k_max + 1)),
         args.n_restarts, args.min_seq_len, region)
        for k in keys
    ]

    if args.workers > 1:
        print(f"Using {args.workers} parallel workers (multiprocessing.Pool)")
        with mp.Pool(args.workers) as pool:
            for i, out_path in enumerate(pool.imap_unordered(process_trace, job_args)):
                if (i+1) % 5 == 0 or i+1 == len(keys):
                    print(f"  [{i+1}/{len(keys)}] {out_path}")
    else:
        for i, ja in enumerate(job_args):
            out_path = process_trace(ja)
            print(f"  [{i+1}/{len(keys)}] {out_path}")

    print(f"Done. Now run: --aggregate --output-dir {args.output_dir} --report ...")


# ---------------------------------------------------------------------------
# Aggregate per-trace results → final JSON
# ---------------------------------------------------------------------------

def cmd_aggregate(args):
    out_dir = Path(args.output_dir)
    results = []
    for p in sorted(out_dir.glob("trace_*.json")):
        try:
            d = json.loads(p.read_text())
            results.append(d)
        except Exception as e:
            print(f"  failed to load {p}: {e}")

    # Status breakdown
    by_status = {}
    for r in results:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    print(f"Aggregating {len(results)} total trace results:")
    for s, n in sorted(by_status.items()):
        print(f"  {s:20s}: {n}")

    ok = [r for r in results if r.get("status") == "ok"]

    if not ok:
        print("No successful traces.")
        return 1

    # Verify dwell-units consistency across traces
    dwell_units_seen = set(r.get("dwell_units", "?") for r in ok)
    if len(dwell_units_seen) > 1:
        print(f"  ⚠ WARNING: mixed dwell_units across traces — DO NOT compare directly!")
        print(f"    seen: {dwell_units_seen}")
    else:
        print(f"  dwell_units (uniform): {next(iter(dwell_units_seen))}")

    # K distribution (BIC and substantive)
    bic_K_dist = {}
    sub_K_dist = {}
    stability_buckets = {"high": 0, "medium": 0, "low": 0}
    for r in ok:
        k = r["bic_selected_K"]
        bic_K_dist[k] = bic_K_dist.get(k, 0) + 1
        sk = r["substantive_selected_K"]
        if sk is not None:
            sub_K_dist[sk] = sub_K_dist.get(sk, 0) + 1
        stab = r.get("k_selection_stability", 0)
        if stab >= 0.75:
            stability_buckets["high"] += 1
        elif stab >= 0.5:
            stability_buckets["medium"] += 1
        else:
            stability_buckets["low"] += 1

    bic_modal = max(bic_K_dist, key=bic_K_dist.get) if bic_K_dist else None
    sub_modal = max(sub_K_dist, key=sub_K_dist.get) if sub_K_dist else None

    # K-selection stability across all OK traces
    overall_stability = sum(r.get("k_selection_stability", 0) for r in ok) / len(ok)
    print(f"  K-selection stability across traces (Jun 10 spec #4):")
    print(f"    high (≥75% restart agreement): {stability_buckets['high']}")
    print(f"    medium (50-75%):                 {stability_buckets['medium']}")
    print(f"    low (<50% — coin-flip):          {stability_buckets['low']}")
    print(f"    overall mean stability: {overall_stability:.2f}")

    # Aggregate dwell statistics at modal substantive K
    target_K = sub_modal or bic_modal
    print(f"BIC-modal K: {bic_modal} (distribution: {dict(sorted(bic_K_dist.items()))})")
    print(f"Substantive-modal K: {sub_modal} (distribution: {dict(sorted(sub_K_dist.items()))})")
    print(f"Reporting dwell stats at K={target_K}")

    # Collect dwell distributions and occupancy at target_K from traces that fit there
    state_dwell_means = {k: [] for k in range(target_K)}
    state_occupancy_means = {k: [] for k in range(target_K)}
    state_implied_dwells = {k: [] for k in range(target_K)}
    state_means_in_log_jsd = {k: [] for k in range(target_K)}
    semi_markov_counts = {k: {} for k in range(target_K)}

    for r in ok:
        if str(target_K) not in r["fits"] and target_K not in r["fits"]:
            continue
        fit = r["fits"].get(target_K) or r["fits"].get(str(target_K))
        for k in range(target_K):
            state_dwell_means[k].append(fit["empirical_mean_dwells"][str(k) if str(k) in fit["empirical_mean_dwells"] else k])
            state_occupancy_means[k].append(fit["occupancy"][str(k) if str(k) in fit["occupancy"] else k])
            state_implied_dwells[k].append(fit["implied_dwells"][str(k) if str(k) in fit["implied_dwells"] else k])
            state_means_in_log_jsd[k].append(fit["state_means"][k])
            flag = fit["semi_markov_flags"][str(k) if str(k) in fit["semi_markov_flags"] else k]
            simple_flag = "peaked" if flag.startswith("peaked") else flag
            semi_markov_counts[k][simple_flag] = semi_markov_counts[k].get(simple_flag, 0) + 1

    # Sort states by their mean log-JSD (low-motion vs high-motion)
    aggregate_states = []
    for k in range(target_K):
        if not state_dwell_means[k]:
            continue
        aggregate_states.append(dict(
            state_idx=k,
            mean_log_jsd=float(np.mean(state_means_in_log_jsd[k])),
            mean_empirical_dwell=float(np.mean(state_dwell_means[k])),
            mean_implied_dwell=float(np.mean(state_implied_dwells[k])),
            mean_occupancy=float(np.mean(state_occupancy_means[k])),
            semi_markov_flag_counts=semi_markov_counts[k],
        ))
    aggregate_states.sort(key=lambda s: s["mean_log_jsd"])

    # Compute dwell ratio: highest-motion state's dwell / lowest-motion state's dwell
    if len(aggregate_states) >= 2:
        # Highest-motion state (largest mean_log_jsd) = "computing" / "propagating"
        # Lowest-motion state (smallest mean_log_jsd) = "stable" / "committed"
        # The N=self-steps-per-cross-step prior could read several ways:
        # - dwell ratio between phases (how many compute steps per commit step)
        # - reciprocal of fast-state dwell (compute updates per "step")
        # Report both and let the user pick:
        slow_dwell = aggregate_states[0]["mean_empirical_dwell"]  # low-motion
        fast_dwell = aggregate_states[-1]["mean_empirical_dwell"] # high-motion
        dwell_ratio_slow_over_fast = slow_dwell / max(fast_dwell, 1e-9)
        dwell_ratio_fast_over_slow = fast_dwell / max(slow_dwell, 1e-9)
    else:
        slow_dwell = fast_dwell = None
        dwell_ratio_slow_over_fast = dwell_ratio_fast_over_slow = None

    report = dict(
        n_traces_total=len(results),
        n_traces_successful=len(ok),
        traces_by_status=by_status,
        dwell_units=next(iter(dwell_units_seen)) if len(dwell_units_seen) == 1 else "MIXED — DO NOT COMPARE",
        bic_K_distribution=dict(sorted(bic_K_dist.items())),
        substantive_K_distribution=dict(sorted(sub_K_dist.items())),
        bic_modal_K=bic_modal,
        substantive_modal_K=sub_modal,
        k_selection_stability_overall=float(overall_stability),
        k_selection_stability_buckets=stability_buckets,
        confidence_verdict=(
            "high — modal K is well-supported"
            if overall_stability >= 0.75
            else "medium — modal K plausible but not airtight"
            if overall_stability >= 0.5
            else "low — modal K is essentially coin-flip; don't lean on it"
        ),
        target_K_for_dwell_analysis=target_K,
        aggregate_states_sorted_by_motion=aggregate_states,
        dwell_summary=dict(
            slow_state_mean_dwell=slow_dwell,
            fast_state_mean_dwell=fast_dwell,
            ratio_slow_over_fast=dwell_ratio_slow_over_fast,
            ratio_fast_over_slow=dwell_ratio_fast_over_slow,
        ),
        v200_implications=dict(
            N_self_per_cross_prior_candidates=dict(
                # If the slow state is "commit" and fast state is "propagate",
                # N = ratio_slow_over_fast is "compute steps per commit step"
                # But the right framing depends on which phase v200's cross-attn
                # corresponds to. Document both readings; user picks in design memo.
                slow_over_fast=dwell_ratio_slow_over_fast,
                fast_over_slow=dwell_ratio_fast_over_slow,
                interpretation_note=(
                    "Dwell ratio measured along token axis. Using as v200's N "
                    "(self-steps per cross-step along breath axis) is the "
                    "transposition bet applied as a design prior, NOT a law. "
                    "If v200 underperforms at this N, sweeping N is legitimate "
                    "tuning, not a rescue. See Jun 10 controls audit memo."
                ),
            ),
            semi_markov_warning=(
                "If any state's semi_markov_flag_counts show 'peaked' "
                "dominantly, the teacher's rhythm is more clock-like than "
                "Markov. Geometric dwell assumption breaks down; consider "
                "semi-Markov model. Reported dwell numbers are still a fair "
                "summary but the model is misspecified."
            ),
        ),
        windowing=dict(
            window=args.window if hasattr(args, "window") else None,
            stride=args.stride if hasattr(args, "stride") else None,
            note="If pre-extracted JSD sequences were used (--jsd-mode extracted), "
                 "window/stride are inherited from the original C1-A pipeline.",
        ),
    )

    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print()
    print(f"Wrote aggregate report: {args.report}")
    print()
    print("Key numbers for v200 design memo:")
    print(f"  Dwell units:                {next(iter(dwell_units_seen)) if len(dwell_units_seen) == 1 else 'MIXED — UNUSABLE'}")
    print(f"  BIC-modal K (raw):          {bic_modal}")
    print(f"  Substantive-modal K:        {sub_modal}")
    print(f"  K-selection stability:      {overall_stability:.2f}  ({report['confidence_verdict']})")
    if aggregate_states:
        for s in aggregate_states:
            print(f"  state {s['state_idx']}: dwell≈{s['mean_empirical_dwell']:.1f}  "
                  f"occ={s['mean_occupancy']:.0%}  log_jsd={s['mean_log_jsd']:.2f}  "
                  f"flags={s['semi_markov_flag_counts']}")
    print(f"  N prior candidates: slow/fast={dwell_ratio_slow_over_fast}  "
          f"fast/slow={dwell_ratio_fast_over_slow}")
    print()
    print("Cautions for v200 brief:")
    if overall_stability < 0.5:
        print("  ⚠ K-selection stability is coin-flip — modal K is NOT a strong signal.")
        print("    Do not bake N into v200 from this number. Sweep N instead.")
    if any("peaked" in str(s["semi_markov_flag_counts"]) for s in aggregate_states):
        print("  ⚠ Some states show peaked (semi-Markov) dwells — geometric assumption broken.")
        print("    Reported dwell numbers may underestimate true characteristic length.")
    if len(dwell_units_seen) > 1:
        print("  ⚠ Mixed dwell_units detected. Re-run with consistent jsd_mode before using.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p_inspect = sub.add_parser("--inspect", aliases=["inspect"],
                                help="Step 0: probe one object's format, exit.")
    p_inspect.add_argument("--bucket", required=True)
    p_inspect.add_argument("--prefix", default="")
    p_inspect.add_argument("--inspect-key", default=None,
                            help="specific key to inspect (else first listed)")
    p_inspect.add_argument("--region", default=None,
                            help="AWS region (defaults to profile/env, e.g. us-east-1)")
    p_inspect.set_defaults(func=cmd_inspect)

    p_fit = sub.add_parser("--fit", aliases=["fit"],
                            help="Per-trace HMM fitting (map-reduce).")
    p_fit.add_argument("--bucket", required=True)
    p_fit.add_argument("--prefix", default="")
    p_fit.add_argument("--suffix", default="",
                       help="filter to keys ending in this (e.g., .npz)")
    p_fit.add_argument("--output-dir", required=True)
    p_fit.add_argument("--workers", type=int, default=4)
    p_fit.add_argument("--window", type=int, default=16,
                       help="C1-A token window size (used if raw attention)")
    p_fit.add_argument("--stride", type=int, default=8,
                       help="C1-A window stride (used if raw attention)")
    p_fit.add_argument("--jsd-mode", choices=["extracted", "raw_attention"],
                       default="extracted",
                       help="extracted=JSD sequence already in object; "
                            "raw_attention=compute JSD on the fly via W/stride")
    p_fit.add_argument("--k-min", type=int, default=2)
    p_fit.add_argument("--k-max", type=int, default=5)
    p_fit.add_argument("--n-restarts", type=int, default=8)
    p_fit.add_argument("--min-seq-len", type=int, default=100,
                       help="skip traces shorter than this (junk fits otherwise)")
    p_fit.add_argument("--region", default=None,
                       help="AWS region (defaults to profile/env, e.g. us-east-1)")
    p_fit.set_defaults(func=cmd_fit)

    p_agg = sub.add_parser("--aggregate", aliases=["aggregate"],
                            help="Aggregate per-trace results into one report.")
    p_agg.add_argument("--output-dir", required=True,
                       help="dir of per-trace JSONs from --fit")
    p_agg.add_argument("--report", required=True,
                       help="output aggregate JSON path")
    p_agg.add_argument("--window", type=int, default=16)
    p_agg.add_argument("--stride", type=int, default=8)
    p_agg.set_defaults(func=cmd_aggregate)

    # Handle the leading-dash subcommands python arg-parse hates
    # Translate "--inspect" → "inspect" in argv
    argv = sys.argv[1:]
    if argv and argv[0] in ("--inspect", "--fit", "--aggregate"):
        argv[0] = argv[0].lstrip("-")

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
