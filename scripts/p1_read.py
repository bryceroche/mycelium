"""p1_read.py — door #42's P1: conversions from the banked 233, split
by the FROZEN length partition (the meter law's lawful form)."""
import json
import numpy as np
base=json.load(open('.cache/miss_census_gen41.json'))
new=json.load(open('.cache/miss_census_g42.json'))
z=np.load('.cache/phase1_alg_states_bigtest.npz'); L=z["tokmask"].sum(1)
old_miss=set(base["miss_idx"]); new_miss=set(new["miss_idx"])
conv=sorted(old_miss-new_miss); regress=sorted(new_miss-old_miss)
lg=[i for i in old_miss if L[i]>189]; sh=[i for i in old_miss if L[i]<=189]
clg=[i for i in conv if L[i]>189]; csh=[i for i in conv if L[i]<=189]
rl=len(clg)/max(len(lg),1); rs=len(csh)/max(len(sh),1)
print(f"[P1] converts {len(conv)} (regressions {len(regress)}) of {len(old_miss)}")
print(f"[P1] long band (> 189): {len(clg)}/{len(lg)} = {rl:.3f}   short band: {len(csh)}/{len(sh)} = {rs:.3f}")
print(f"[P1] ratio long:short = {rl/max(rs,1e-9):.2f}  (prediction >=2.0; flat = length-a-correlate)")
json.dump({"converts":conv,"regressions":regress,"rate_long":rl,"rate_short":rs},
          open('.cache/p1_read.json','w'))
