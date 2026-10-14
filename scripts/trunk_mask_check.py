"""trunk_mask_check.py (2026-10-14) -- zero-GPU (DEV=CPU): cosine of L0..L3 trunk states under
the DEPLOYED precompute regime (phase1_algebra_head.do_precompute: no attn_mask, padded to T_ALG=256 with id 0)
vs bidirectional-unpadded vs Llama-native causal-unpadded, on the first 4 pm35c rows."""
import os, sys, json
os.environ["DEV"]="CPU"
sys.path.insert(0,'.')
import numpy as np
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG
tok=Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
rows=[json.loads(l) for _,l in zip(range(4),open('.cache/form_mix_pm35c.jsonl'))]
e=tok.encode(rows[0]['text']); print("first ids", e.ids[:3], tok.id_to_token(e.ids[0]))
class H: pass
h=H(); sd=load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(h,n_layers=4,sd=sd,cfg=LLAMA_3_2_1B_CFG); del sd
T=256
def run(ids, mask):
    x=h.llama_embed[Tensor(ids[None],dtype=dtypes.int)]
    outs=[]
    for L in h.llama_layers:
        x=L(x,h.llama_rope_cos,h.llama_rope_sin,attn_mask=mask); outs.append(x.numpy()[0])
    return outs
for r in rows:
    ids=np.array(tok.encode(r['text']).ids,np.int32); n=len(ids)
    pad=np.zeros(T,np.int32); pad[:n]=ids
    dep=run(pad,None)                                    # deployed: no mask, padded
    bid=run(ids,None)                                    # bidirectional, unpadded
    cm=np.triu(np.full((n,n),-1e9,np.float32),1)
    cau=run(ids,Tensor(cm[None,None]))                   # native: causal, unpadded
    for li in range(4):
        def cos(a,b): return float(np.mean(np.sum(a*b,1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1)))
        print(f"n={n} L{li}: cos(deployed,causal)={cos(dep[li][:n],cau[li]):.3f}  cos(bidir,causal)={cos(bid[li],cau[li]):.3f}  cos(deployed,bidir)={cos(dep[li][:n],bid[li]):.3f}")
