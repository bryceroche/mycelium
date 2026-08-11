"""p233_read.py — door #43's P-233: ordinal-graded conversions from the
banked 233 (frozen indices; the meter law's form)."""
import json, re
import numpy as np
base=json.load(open('.cache/miss_census_gen41.json'))
import os
new=json.load(open(os.environ.get('P233_NEW','.cache/miss_census_g44.json')))
rows=[json.loads(l) for l in open('.cache/algebra_nl_bigtest.jsonl')]
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
om=set(base["miss_idx"]); nm=set(new["miss_idx"])
conv=om-nm
o=[i for i in om if ORD.search(rows[i]["text"])]; no=[i for i in om if not ORD.search(rows[i]["text"])]
co=[i for i in conv if i in set(o)]; cno=[i for i in conv if i in set(no)]
ro=len(co)/max(len(o),1); rno=len(cno)/max(len(no),1)
print(f"[P233] converts {len(conv)} (regressions {len(nm-om)}) of {len(om)}")
print(f"[P233] ordinal band: {len(co)}/{len(o)} = {ro:.3f}   non-ordinal: {len(cno)}/{len(no)} = {rno:.3f}")
print(f"[P233] ratio = {ro/max(rno,1e-9):.2f}  (prediction >=2.0)")
json.dump({"converts":sorted(conv),"rate_ord":ro,"rate_nonord":rno},open('.cache/p233_read.json','w'))
