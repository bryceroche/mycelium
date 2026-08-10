"""gen41_verdict.py — DOOR #41 PROMOTION (2026-08-10; the word given
and countersigned; the manifest write and PROMOTED are one atomic
act). Field-drop law: load the existing manifest, mutate consciously,
carry everything else. Riders as DATA: op-margin thinness; the
refold clause (successors inherit the REQUIREMENT, not the cure)."""
import json, hashlib, shutil
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]
man = json.load(open(".cache/GENERATION.json"))
prior_notes = man["notes"]
man["gen_id"] = "41"
man["date"]   = "2026-08-10"
man["parser_ckpt"] = ".cache/g41_onemass_refold.safetensors"
man["corpora"]["train"] = ".cache/form_mix3.jsonl"
man["scope"]["dup_cure"]["claimed"] = ("dup binding at deployment scope: scan 0/15 (nd0), 3/15 (nd1), 0/15 (nd4) "
  "misbind; row-grade 15/15 at nd0 AND nd4 on the fenced novel surface (door #41 battery)")
man["scope"]["dup_cure"]["mechanism"] = ("calibrated-rite fold on g41's OWN waist (fixture-min 1.534 / neg-max 0.710, "
  "theta 1.1220, holdout FPR 0.00000; control g23v5 exact, sixth consecutive deployment)")
man["scope"]["op_margin_thinness"] = {
  "data": "nd0 mul-dup fixture margins +0.74..+2.00 (gate lineage floor was +1.85..+3.11; #40's distinct-args arm +1.45..+2.76)",
  "reading": "the args=[a,a] vehicle supplies SUFFICIENCY; distinct-args mass bought extra depth. Decisions 5/5 correct.",
  "no_reader_may_infer": "deep op margins on mul-dup at low crowding — the next fire touching mul MEETS THIS WALL; maintenance note, not defect"}
man["scope"]["refold_clause"] = ("this checkpoint holds the dup cure because its fold was derived on ITS OWN waist and "
  "nothing has trained it since. REFOLD AFTER ANY TRAINING (the law's final form): any successor inherits the "
  "REQUIREMENT (scripts/calibrated_rite.py, control-assert first, always) — not the cure.")
man["regression_bars"]["bigtest"] = 1254
man["regression_bars"]["alg4test"] = 405
man["regression_bars"]["stress_certified_lies"] = 0
man["hashes"]["parser"] = sha(".cache/g41_onemass_refold.safetensors")
man["hashes"]["train"]  = sha(".cache/form_mix3.jsonl")
man["notes"] = ("2026-08-10 GEN-41 PROMOTED (door #41, the one-mass fire; the word given and countersigned): "
  "g41_onemass_refold — crowded mul-dup sliver 2,700 @ 8x beside size 3,000 (one vehicle, two goods: formation AND "
  "op-balance; the distinct-args counterweight retired by the interference fingerprint), dup-familiar 8,233 @ 3x, "
  "h_dup frozen in-fire, fold re-derived post-fire by the calibrated rite (control sixth-exact). SHEET: scan 0/3/-/0 "
  "FULL SCOPE (first both-goods arm); row-grade 15/15 both ends; bigtest 1254; alg4 405; sentinels 20/20 x2; frontier "
  "DEPLOYED 40/40; STRESS CERTIFIED LIES ZERO (gate baseline was 1 — the claim's own currency IMPROVED). Explicit "
  "lines: op-margin thinness (scope as data), size -5 vs record 1259, stress churn resolved as noise. ENTOURAGE-41 "
  "OWED: specialist remine, centroids rebuild, mouth/watcher recal, registry re-audition, frontier stands LAST "
  "(post-dressing, per the rider — 40/40 before dressing does not guarantee after). | " + prior_notes)
shutil.copy(".cache/GENERATION.json", ".cache/GENERATION.json.pre41")
json.dump(man, open(".cache/GENERATION.json","w"), indent=1)
print("PROMOTED — GEN-41 (manifest written; entourage owed)")
