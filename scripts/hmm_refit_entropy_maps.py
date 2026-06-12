"""HMM re-fit on teacher per-token ENTROPY maps (salvage, Jun 10).

Per Jun 10 spec: the canonical attention-window JSD data wasn't persisted
(C1-A pipeline computed JSD on the fly during training and never wrote
intermediates to S3). But 121 c0_entropy maps survive in
s3://mycelium-data/analysis/c0_entropy/maps/ — per-token entropy
sequences from the same teacher runs.

Entropy ≠ JSD: attention SHARPNESS, not attention CHANGE. Different
instrument. But if the teacher genuinely telegraphs between read and
compute states, that rhythm plausibly shows in entropy too (diffuse-
attention phases vs sharp-attention phases). Worth 20 CPU-min to
confirm the question closes today.

Disciplines from the JSD script preserved:
- Log-transform before Gaussian HMM (entropy is non-neg + bursty)
- 8 EM restarts per (trajectory, K), keep best LL
- BIC across K=2..5
- Dwell two ways (empirical Viterbi + implied 1/(1-p_ii))
- Geometric-vs-peaked semi-Markov flag
- Junk-state filter (≥5% occupancy)
- K-selection stability across restarts
- Min-seq-len guard
- Caveat field in every output JSON: "INSTRUMENT=entropy NOT JSD"

Sampling: 1 trajectory per file = 121 total. Subsampling is intentional
— full 6050 trajectories would be ~27hr CPU.

Usage:
  pip install boto3 hmmlearn numpy scipy

  python hmm_refit_entropy_maps.py --fit \\
      --bucket mycelium-data \\
      --prefix analysis/c0_entropy/maps/ \\
      --output-dir ./hmm_entropy_results \\
      --workers 4

  python hmm_refit_entropy_maps.py --aggregate \\
      --output-dir ./hmm_entropy_results \\
      --report hmm_entropy_aggregate.json
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np


CAVEAT = ("INSTRUMENT=entropy (per-token attention sharpness), NOT JSD "
          "(attention change). State count and dwell structure measured "
          "here may differ from the canonical JSD instrument. Use with "
          "stated caveat in v200 design memo.")


def fit_hmm_for_K(log_seq, K, n_restarts=8, seed_base=0):
    """Fit GaussianHMM at given K. Returns best fit + all restart BICs."""
    from hmmlearn.hmm import GaussianHMM

    X = log_seq.reshape(-1, 1)
    if len(X) < K * 5:
        return None

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

    fits.sort(key=lambda f: f["ll"], reverse=True)
    best = fits[0]
    best["K"] = K
    best["n_params"] = n_params
    best["N"] = N
    best["all_restart_bics"] = sorted([f["bic"] for f in fits])
    best["all_restart_lls"] = sorted([f["ll"] for f in fits], reverse=True)
    return best


def extract_dwell_stats(hmm, log_seq):
    X = log_seq.reshape(-1, 1)
    states = hmm.predict(X)
    K = hmm.n_components

    # Empirical dwell from Viterbi runs
    empirical = {k: [] for k in range(K)}
    cur, cur_len = states[0], 1
    for s in states[1:]:
        if s == cur:
            cur_len += 1
        else:
            empirical[cur].append(cur_len)
            cur, cur_len = s, 1
    empirical[cur].append(cur_len)

    trans = hmm.transmat_
    implied = {k: float(1.0 / max(1.0 - trans[k, k], 1e-9)) for k in range(K)}
    occupancy = {k: float((states == k).mean()) for k in range(K)}
    state_means = {k: float(hmm.means_[k][0]) for k in range(K)}

    # Geometric vs peaked check
    from collections import Counter
    semi_markov_flags = {}
    for k in range(K):
        dwells = np.array(empirical[k])
        if len(dwells) < 10:
            semi_markov_flags[k] = "insufficient_data"
            continue
        c = Counter(dwells.tolist())
        mode_dwell = max(c, key=c.get)
        mode_mass = c[mode_dwell] / len(dwells)
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


def process_file(args_tuple):
    """Pull one S3 file, take ONE trajectory's entropy sequence, fit HMMs."""
    (key, bucket, output_dir, k_range, n_restarts, min_seq_len,
     traj_idx, region) = args_tuple

    import boto3

    safe_name = key.replace("/", "_").replace(":", "_")
    out_path = Path(output_dir) / f"trace_{safe_name}_t{traj_idx}.json"
    if out_path.exists():
        return str(out_path)

    try:
        s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        problems = json.loads(body)
        if not isinstance(problems, list) or traj_idx >= len(problems):
            out_path.write_text(json.dumps(dict(
                key=key, status="bad_format_or_traj_oob",
                n_problems=len(problems) if isinstance(problems, list) else None,
                traj_idx=traj_idx,
            )))
            return str(out_path)

        prob = problems[traj_idx]
        entropy_seq = prob.get("entropy_map_weighted")
        if entropy_seq is None:
            entropy_seq = prob.get("entropy_map") or prob.get("entropy")
        if entropy_seq is None:
            out_path.write_text(json.dumps(dict(
                key=key, status="no_entropy_field",
                available_keys=list(prob.keys()),
            )))
            return str(out_path)

        entropy_seq = np.asarray(entropy_seq, dtype=np.float64).flatten()
        n_input = prob.get("n_input_tokens")
        n_decode = prob.get("n_decode_steps")

        if len(entropy_seq) < min_seq_len:
            out_path.write_text(json.dumps(dict(
                key=key, status="too_short", seq_len=int(len(entropy_seq)),
                min_required=min_seq_len, caveat=CAVEAT,
            )))
            return str(out_path)

        # Log-transform (entropy can be very small near sharp-attention boundaries;
        # the small + epsilon keeps log finite)
        log_seq = np.log(entropy_seq + 1e-9)

        fits = {}
        for K in k_range:
            fit = fit_hmm_for_K(log_seq, K, n_restarts=n_restarts)
            if fit is None:
                continue
            dwell = extract_dwell_stats(fit["hmm"], log_seq)
            # Note: dwell already contains a `state_means` dict (per-state log values).
            # Keep that and don't duplicate. Rename the flat list to state_means_flat.
            fits[K] = dict(
                K=K, ll=fit["ll"], bic=fit["bic"],
                n_params=fit["n_params"], N=fit["N"],
                trans_matrix=fit["hmm"].transmat_.tolist(),
                start_prob=fit["hmm"].startprob_.tolist(),
                state_means_flat=fit["hmm"].means_.flatten().tolist(),
                state_covars_flat=fit["hmm"].covars_.flatten().tolist(),
                all_restart_bics=fit["all_restart_bics"],
                all_restart_lls=fit["all_restart_lls"],
                **dwell,  # contributes state_means (per-state dict), occupancy, etc.
            )

        if not fits:
            out_path.write_text(json.dumps(dict(
                key=key, status="all_K_failed",
                seq_len=int(len(entropy_seq)), caveat=CAVEAT,
            )))
            return str(out_path)

        bic_K = min(fits, key=lambda k: fits[k]["bic"])
        substantive_K = None
        for K in sorted(fits):
            if all(v >= 0.05 for v in fits[K]["occupancy"].values()):
                substantive_K = K
        if substantive_K is None and 2 in fits:
            substantive_K = 2

        # K-selection stability
        k_selections_by_restart = []
        n_rest = len(fits[bic_K]["all_restart_bics"])
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

        out_path.write_text(json.dumps(dict(
            key=key,
            traj_idx=traj_idx,
            status="ok",
            n_input_tokens=n_input,
            n_decode_steps=n_decode,
            seq_len=int(len(entropy_seq)),
            entropy_min=float(entropy_seq.min()),
            entropy_max=float(entropy_seq.max()),
            entropy_mean=float(entropy_seq.mean()),
            entropy_std=float(entropy_seq.std()),
            log_entropy_mean=float(log_seq.mean()),
            log_entropy_std=float(log_seq.std()),
            bic_selected_K=int(bic_K),
            substantive_selected_K=int(substantive_K) if substantive_K else None,
            k_selections_across_restarts=k_selections_by_restart,
            k_selection_stability=float(k_selection_stability),
            fits=fits,
            instrument="entropy",
            caveat=CAVEAT,
        ), indent=2, default=str))
        return str(out_path)

    except Exception as e:
        out_path.write_text(json.dumps(dict(
            key=key, status="error", error=str(e),
            error_type=type(e).__name__, caveat=CAVEAT,
        )))
        return str(out_path)


def cmd_fit(args):
    import boto3

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    region = args.region

    print(f"Listing s3://{args.bucket}/{args.prefix}...")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])

    print(f"Found {len(keys)} JSON files. Output dir: {args.output_dir}")
    print(f"Settings: K={args.k_min}..{args.k_max}  n_restarts={args.n_restarts}  "
          f"min_seq_len={args.min_seq_len}  traj_idx={args.traj_idx}")
    print(f"INSTRUMENT: entropy (NOT JSD). Caveat applies to all outputs.")
    print()

    job_args = [
        (k, args.bucket, args.output_dir,
         list(range(args.k_min, args.k_max + 1)),
         args.n_restarts, args.min_seq_len, args.traj_idx, region)
        for k in keys
    ]

    if args.workers > 1:
        print(f"Using {args.workers} parallel workers")
        with mp.Pool(args.workers) as pool:
            for i, out_path in enumerate(pool.imap_unordered(process_file, job_args)):
                if (i+1) % 10 == 0 or i+1 == len(keys):
                    print(f"  [{i+1}/{len(keys)}]")
    else:
        for i, ja in enumerate(job_args):
            process_file(ja)
            if (i+1) % 10 == 0 or i+1 == len(keys):
                print(f"  [{i+1}/{len(keys)}]")

    print(f"Done. Now: --aggregate --output-dir {args.output_dir} --report ...")


def cmd_aggregate(args):
    out_dir = Path(args.output_dir)
    results = []
    for p in sorted(out_dir.glob("trace_*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except Exception:
            pass

    by_status = {}
    for r in results:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    print(f"Aggregating {len(results)} total results:")
    for s, n in sorted(by_status.items()):
        print(f"  {s:20s}: {n}")
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        print("No successful fits.")
        return 1

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
    overall_stability = sum(r.get("k_selection_stability", 0) for r in ok) / len(ok)
    target_K = sub_modal or bic_modal

    print(f"\nBIC-modal K (raw): {bic_modal}  distribution: {dict(sorted(bic_K_dist.items()))}")
    print(f"Substantive-modal K: {sub_modal}  distribution: {dict(sorted(sub_K_dist.items()))}")
    print(f"K-selection stability: {overall_stability:.2f}")
    print(f"  high≥75%: {stability_buckets['high']}  medium: {stability_buckets['medium']}  low<50%: {stability_buckets['low']}")

    state_dwell_means = {k: [] for k in range(target_K)}
    state_occ = {k: [] for k in range(target_K)}
    state_implied = {k: [] for k in range(target_K)}
    state_log_means = {k: [] for k in range(target_K)}
    state_semi = {k: {} for k in range(target_K)}
    for r in ok:
        fits = r["fits"]
        K = str(target_K) if str(target_K) in fits else target_K
        if K not in fits:
            continue
        fit = fits[K]
        for k in range(target_K):
            sk = str(k) if str(k) in fit["empirical_mean_dwells"] else k
            state_dwell_means[k].append(fit["empirical_mean_dwells"][sk])
            state_occ[k].append(fit["occupancy"][sk])
            state_implied[k].append(fit["implied_dwells"][sk])
            sm = fit["state_means"]
            sk_means = str(k) if str(k) in sm else k
            state_log_means[k].append(sm[sk_means])
            flag = fit["semi_markov_flags"][sk]
            simple = "peaked" if flag.startswith("peaked") else flag
            state_semi[k][simple] = state_semi[k].get(simple, 0) + 1

    aggregate_states = []
    for k in range(target_K):
        if not state_dwell_means[k]:
            continue
        aggregate_states.append(dict(
            state_idx=k,
            mean_log_entropy=float(np.mean(state_log_means[k])),
            mean_empirical_dwell=float(np.mean(state_dwell_means[k])),
            mean_implied_dwell=float(np.mean(state_implied[k])),
            mean_occupancy=float(np.mean(state_occ[k])),
            semi_markov_flag_counts=state_semi[k],
        ))
    aggregate_states.sort(key=lambda s: s["mean_log_entropy"])

    if len(aggregate_states) >= 2:
        slow = aggregate_states[0]["mean_empirical_dwell"]   # low-entropy (sharp attention)
        fast = aggregate_states[-1]["mean_empirical_dwell"]  # high-entropy (diffuse attention)
        ratio_slow_over_fast = slow / max(fast, 1e-9)
        ratio_fast_over_slow = fast / max(slow, 1e-9)
    else:
        slow = fast = ratio_slow_over_fast = ratio_fast_over_slow = None

    report = dict(
        instrument="entropy (NOT JSD)",
        caveat=CAVEAT,
        n_traces_total=len(results),
        n_traces_successful=len(ok),
        traces_by_status=by_status,
        bic_K_distribution=dict(sorted(bic_K_dist.items())),
        substantive_K_distribution=dict(sorted(sub_K_dist.items())),
        bic_modal_K=bic_modal,
        substantive_modal_K=sub_modal,
        k_selection_stability_overall=float(overall_stability),
        k_selection_stability_buckets=stability_buckets,
        confidence_verdict=(
            "high — modal K well-supported" if overall_stability >= 0.75
            else "medium" if overall_stability >= 0.5
            else "low — modal K essentially coin-flip"
        ),
        target_K_for_dwell_analysis=target_K,
        aggregate_states_sorted_by_motion=aggregate_states,
        dwell_summary=dict(
            sharp_attention_state_mean_dwell=slow,
            diffuse_attention_state_mean_dwell=fast,
            ratio_sharp_over_diffuse=ratio_slow_over_fast,
            ratio_diffuse_over_sharp=ratio_fast_over_slow,
        ),
        v200_implications=dict(
            note=(
                "Entropy traces measure attention SHARPNESS rhythm, not "
                "attention CHANGE. If teacher genuinely telegraphs read vs "
                "compute states, both signals plausibly carry it. But the "
                "absolute dwell numbers and ratio are measured in a "
                "different instrument; using them as v200's N is the "
                "transposition bet applied with an additional instrument-"
                "translation caveat. Sweep N at v1.1 regardless."
            ),
        ),
    )

    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote: {args.report}")
    print()
    print(f"Cautions:")
    if overall_stability < 0.5:
        print("  ⚠ stability coin-flip; modal K is NOT a strong signal")
    if any("peaked" in str(s["semi_markov_flag_counts"]) for s in aggregate_states):
        print("  ⚠ semi-Markov: some states show peaked dwells; geometric assumption broken")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p_fit = sub.add_parser("--fit", aliases=["fit"])
    p_fit.add_argument("--bucket", required=True)
    p_fit.add_argument("--prefix", required=True)
    p_fit.add_argument("--output-dir", required=True)
    p_fit.add_argument("--workers", type=int, default=4)
    p_fit.add_argument("--k-min", type=int, default=2)
    p_fit.add_argument("--k-max", type=int, default=5)
    p_fit.add_argument("--n-restarts", type=int, default=8)
    p_fit.add_argument("--min-seq-len", type=int, default=100)
    p_fit.add_argument("--traj-idx", type=int, default=0,
                       help="Which trajectory (of 50 per file) to use")
    p_fit.add_argument("--region", default="us-east-1")
    p_fit.set_defaults(func=cmd_fit)

    p_agg = sub.add_parser("--aggregate", aliases=["aggregate"])
    p_agg.add_argument("--output-dir", required=True)
    p_agg.add_argument("--report", required=True)
    p_agg.set_defaults(func=cmd_aggregate)

    argv = sys.argv[1:]
    if argv and argv[0] in ("--fit", "--aggregate"):
        argv[0] = argv[0].lstrip("-")
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
