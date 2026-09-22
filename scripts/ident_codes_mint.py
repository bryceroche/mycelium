"""ident_codes_mint.py -- THE IDENTITY TABLE + THE IDENTITY PHASOR MINT (2026-09-22, THE BUS
REGISTER organ; word given on the mixed-bucket decomposition's reading: ledger 2026-09-22 12:50).

(1) .cache/ident_codes512.npz -- IDT (vocab, 2P) float16: every Llama-3.2-1B token id's INPUT
    EMBEDDING (frozen, context-free) projected by ONE FIXED orthonormal map (2048 -> 512, a seeded
    Gaussian's QR; never trained, stored beside the table) and CENTRED on the vocabulary mean and unit-normalised per row, so the
    identity of a token is a DIRECTION in the bus's own 512-d (256 complex planes) space. Row 0
    (the pad id in the staged arrays) is zeroed: pads never enter a pooled identity even before
    the token mask does its job. This is the identity stream of the registered DUAL-STREAM port
    (ledger 2026-09-22 THE TWIN-SPLIT CENSUS): context-free, breath-free, never summed into the
    waist -- the bus register carries it beside the role.
(2) .cache/bindbus_codes512ri.npz -- the rcue codebook (bindbus_codes512r.npz) plus theta_id
    (P,), a seventh role offset minted the same way as the others (uniform random phase per
    plane, its own documented seed); every old key byte-identical (checked below), so every config
    that does not opt into BIND_CODES=<this file> is bit-identical.
usage: ident_codes_mint.py [weights.safetensors] [codes_in] [codes_out] [ident_out]
"""
import json
import struct
import sys

import numpy as np

W = sys.argv[1] if len(sys.argv) > 1 else ".cache/llama-3.2-1b-weights/model.safetensors"
CODES_IN = sys.argv[2] if len(sys.argv) > 2 else ".cache/bindbus_codes512r.npz"
CODES_OUT = sys.argv[3] if len(sys.argv) > 3 else ".cache/bindbus_codes512ri.npz"
IDENT_OUT = sys.argv[4] if len(sys.argv) > 4 else ".cache/ident_codes512.npz"
D_BUS = 512
SEED_PROJ = 0x1d5e   # the fixed projection's seed (documented; reproducible)
SEED_THETA = 0x1d5f  # theta_id's seed (not used by any other theta_* mint in this tree)

# --- the embedding, read straight from the safetensors header (no model build) ---
with open(W, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    meta = hdr["model.embed_tokens.weight"]
    assert meta["dtype"] == "BF16", meta
    a, b = meta["data_offsets"]
    f.seek(8 + n + a)
    raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
V, H = meta["shape"]
# bf16 -> f32: the 16 bits are the top half of the float32 pattern
E = (raw.astype(np.uint32) << 16).view(np.float32).reshape(V, H)
# CENTRED: the embedding carries a shared offset (random-pair cosine 0.12 uncentred); an identity is a
# direction RELATIVE to the vocabulary's mean, so unrelated identities dot to ~0 and a match is a match.
E = E - E[1:].mean(0, keepdims=True)

rng = np.random.default_rng(SEED_PROJ)
Q, _ = np.linalg.qr(rng.standard_normal((H, D_BUS)).astype(np.float64))
P_fixed = Q.astype(np.float32)                      # (H, D_BUS), orthonormal columns
IDT = E @ P_fixed                                   # (V, D_BUS)
norms = np.linalg.norm(IDT, axis=1, keepdims=True)
IDT = IDT / np.maximum(norms, 1e-6)
IDT[0] = 0.0                                        # the pad id: no identity
np.savez(IDENT_OUT, IDT=IDT.astype(np.float16), P_fixed=P_fixed,
         seed=np.int64(SEED_PROJ), vocab=np.int64(V), embed_dim=np.int64(H), d_bus=np.int64(D_BUS))

# --- the codebook: theta_id appended, old keys byte-identical ---
z = np.load(CODES_IN)
out = {k: z[k] for k in z.files}
P = out["theta_rcue"].shape[0]
out["theta_id"] = np.random.default_rng(SEED_THETA).uniform(0.0, 2 * np.pi, P).astype(out["theta_rcue"].dtype)
np.savez(CODES_OUT, **out)
z2 = np.load(CODES_OUT)
for k in z.files:
    assert np.array_equal(z[k], z2[k]), f"[ident-mint] FAILED: {k} changed"
assert z2["theta_id"].shape == (P,)
i2 = np.load(IDENT_OUT)["IDT"]
# a sanity read: same-word tokens are identical by construction; a few cosines for the record
def cos(a_, b_):
    return float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9))
print(f"[ident-mint] {IDENT_OUT}: IDT {i2.shape} {i2.dtype}; row-norm mean "
      f"{np.linalg.norm(i2[1:].astype(np.float32), axis=1).mean():.4f}; "
      f"random-pair cosine mean {np.mean([cos(i2[i].astype(np.float32), i2[j].astype(np.float32)) for i, j in rng.integers(1, V, (200, 2))]):.4f}")
print(f"[ident-mint] {CODES_OUT}: theta_id minted ({P} planes, seed {SEED_THETA:#x}) min "
      f"{z2['theta_id'].min():.4f} max {z2['theta_id'].max():.4f}; old keys ({', '.join(z.files)}) "
      f"byte-identical to {CODES_IN}")
