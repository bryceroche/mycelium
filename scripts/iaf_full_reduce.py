"""iaf_full_reduce.py — THE FULL 117-CHUNK TELEGRAPH (2026-08-27, word
given; Bryce's lambda-mapreduce recommendation honored: 10GB map fn).
Fan out all chunks to mycelium-iaf-telegraph, reduce locally: per-head
switch rates + dwell log-histograms + A-B-A vs null + synchrony at
W=2/5/15 + per-level and per-type strata. Stats only, corpus stays
analysis-only.
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import boto3

s3 = boto3.client('s3')
lam = boto3.client('lambda', region_name='us-west-2',
                   config=boto3.session.Config(read_timeout=660,
                                               max_pool_connections=40))
keys = []
pg = s3.get_paginator('list_objects_v2')
for page in pg.paginate(Bucket='mycelium-data', Prefix='iaf_extraction/chunked/'):
    for o in page.get('Contents', []):
        if o['Key'].endswith('.json'):
            keys.append(o['Key'])
print(f"[full] {len(keys)} chunks", flush=True)

def run(k):
    try:
        r = lam.invoke(FunctionName='mycelium-iaf-telegraph',
                       Payload=json.dumps({'key': k}))
        out = json.loads(r['Payload'].read())
        if 'errorMessage' in out:
            return ('err', k, out['errorMessage'][:100])
        return ('ok', k, out)
    except Exception as e:
        return ('err', k, str(e)[:100])

results = []
with ThreadPoolExecutor(max_workers=30) as ex:
    for i, res in enumerate(ex.map(run, keys)):
        results.append(res)
        if (i + 1) % 20 == 0:
            print(f"[full] {i+1}/{len(keys)}", flush=True)
oks = [r[2] for r in results if r[0] == 'ok']
errs = [r for r in results if r[0] == 'err']
print(f"[full] ok {len(oks)}, err {len(errs)}", flush=True)
for e in errs[:5]: print("[full] ERR", e[1], e[2], flush=True)

H = {}; SY = {}
nrec = 0
for out in oks:
    nrec += out['nrec']
    for h, st in out['heads'].items():
        A = H.setdefault(h, {'sw': 0, 'steps': 0, 'ret': 0, 'retn': 0.0,
                             'dw': [0]*12, 'anch': 0, 'gens': 0,
                             'lv': {}, 'ty': {}})
        for f in ('sw', 'steps', 'ret', 'retn', 'anch', 'gens'):
            A[f] += st[f]
        A['dw'] = [a + b for a, b in zip(A['dw'], st['dw'])]
        for d, agg in (('lv', A['lv']), ('ty', A['ty'])):
            for k2, v in st[d].items():
                e = agg.setdefault(k2, [0, 0, 0])
                for i in range(3): e[i] += v[i]
    for w, v in out['sync'].items():
        e = SY.setdefault(w, [0.0, 0.0, 0])
        for i in range(3): e[i] += v[i]

json.dump({'heads': H, 'sync': SY, 'nrec': nrec},
          open('.cache/iaf_full_stats.json', 'w'))
print(f"[full] === {nrec} generations across {len(oks)} chunks ===", flush=True)
import re
heads = sorted(H, key=lambda k: (int(re.match(r'L(\d+)', k).group(1)),
                                 int(re.search(r'H(\d+)', k).group(1))))
print("[full] head | sw/1k | anchors/gen | ret/step obs:null | dwell hist (log2 buckets 1,2,4..)")
for h in heads:
    A = H[h]
    print(f"[full] {h:7s}| {1000*A['sw']/max(A['steps'],1):6.1f} "
          f"| {A['anch']/max(A['gens'],1):6.1f} "
          f"| {A['ret']/max(A['steps'],1):.4f}:{A['retn']/max(A['steps'],1):.4f} "
          f"| {A['dw'][:8]}", flush=True)
print("[full] per-LEVEL switch rates (sw/1k), fast keyers vs lockers:")
FAST = [h for h in heads if 1000*H[h]['sw']/max(H[h]['steps'],1) > 100]
LOCK = [h for h in heads if 1000*H[h]['sw']/max(H[h]['steps'],1) <= 20]
for grp, name in ((FAST, 'keyers'), (LOCK, 'lockers')):
    for lvl in sorted({k for h in grp for k in H[h]['lv']}):
        sw = sum(H[h]['lv'].get(lvl, [0,0,0])[0] for h in grp)
        stp = sum(H[h]['lv'].get(lvl, [0,0,0])[1] for h in grp)
        if stp > 1000:
            print(f"[full]   {name:7s} {lvl:9s}: {1000*sw/stp:6.1f}", flush=True)
for w in sorted(SY, key=int):
    o, n, c = SY[w]
    print(f"[full] SYNCHRONY W={w}: obs {o/max(c,1):.3f} vs null {n/max(c,1):.3f} "
          f"({o/max(n,1e-9):.2f}x, {c} pair-gens)", flush=True)
