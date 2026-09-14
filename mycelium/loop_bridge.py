"""loop_bridge.py — THE BRIDGE between the two mediums (2026-09-14, word
given; blog "One map, two charts").

The machine has two loops on two mediums: the NL loop on TOKENS and the
math loop on SLOTS. They share the breath clock, the rotation bus, the
atlas (one map, two charts — mycelium/step_atlas.Atlas) and THIS object.

Two lanes:
  THE FORWARD LANE is the waist: the trunk's reading (2048d) carried
  down to the 512d space the slots read (the head's waist projection —
  the waist IS the token medium the NL loop breathes on).
  THE RETURN LANE is the cross attention itself: each breath's
  slots<-tokens attention (B, LT, T), head-mean, is the ONLY
  correspondence between the mediums that exists, so it is what
  carries CERTIFICATES back and forth as dynamic masks.

A Certificate is born in one medium (the solver's MUC contradiction and
the fingerpost's view-instability locus are born on slots; the claims are
born on the matrix itself) and is PROJECTED into the other by the bridge:
Bridge.to_tokens / Bridge.to_slots. Bridge.spotlight is the wheel's bias
(the token-side projection of a slot certificate, per flagged slot); it
replaces the two private copies that lived in the head (_wheel_turn) and
the step trainer (wheel_bias), shown equal to both before they were
removed (scripts/bridge_identity_check.py). The in-graph roads (the T2
claim bias, the melt, the nogood, the token loop's within-sentence mask)
live here as pure functions of tensors so no loop builds a cross-medium
or within-medium mask by hand.

All of it is CONDITIONING / STRUCTURE — never a supervised target
(Goodhart fence); hard -inf never (A0's grave): masks are additive
biases or {0,1} carries the consumer applies.
"""
import numpy as np


class Certificate:
    """A certificate born in one medium. kind: "muc" (the solver's
    unsat core), "locus" (view instability), "claims", ... . slots:
    (B, LT) {0,1} or None; tokens: (B, T) {0,1}/mass or None; nogoods:
    [(b, j, field, idx)] refuted choices; breath: the breath it was
    born at (applied at the next)."""

    def __init__(self, kind, slots=None, tokens=None, nogoods=None, breath=None):
        self.kind = str(kind)
        self.slots = None if slots is None else np.asarray(slots, np.float32)
        self.tokens = None if tokens is None else np.asarray(tokens, np.float32)
        self.nogoods = list(nogoods or [])
        self.breath = breath
        assert self.slots is not None or self.tokens is not None or self.nogoods, \
            "an empty certificate certifies nothing"
        # the medium it was BORN in is fixed at birth (a projection never changes it)
        self.born = "slot" if self.slots is not None else "token"

    def rows_flagged(self):
        s = self.slots if self.slots is not None else self.tokens
        return (s > 0).any(1) if s is not None else np.zeros(0, bool)


class Bridge:
    """THE RETURN LANE of one breath: fat (B, LT, T) = the head-mean
    slots<-tokens attention; sent (B, T) sentence ids; tokmask (B, T)
    real-token mask (None = all real)."""

    def __init__(self, fat, sent, tokmask=None):
        self.fat = np.asarray(fat, np.float32)
        self.sent = np.asarray(sent)
        assert self.fat.ndim == 3 and self.sent.shape == (self.fat.shape[0], self.fat.shape[2]), \
            (self.fat.shape, self.sent.shape)
        self.B, self.LT, self.T = (int(x) for x in self.fat.shape)
        self.tokmask = (np.ones((self.B, self.T), np.float32) if tokmask is None
                        else np.asarray(tokmask, np.float32).reshape(self.B, self.T))

    # --- the correspondence ----------------------------------------------
    def source_token(self, b, j):
        """The token slot j reads most (argmax of its attention), clamped."""
        return min(int(self.fat[b, j].argmax()), self.T - 1)

    def source_sentence(self, b, j):
        return int(self.sent[b, self.source_token(b, j)])

    # --- slots -> tokens ---------------------------------------------------
    def to_tokens(self, slots, how="sentence"):
        """Project a slot certificate (B, LT) onto the tokens (B, T).
        "sentence": 1 on every real token of the source sentences of the
        flagged slots (the spotlight's carry, union over the row's flagged
        slots); "mass": the attention mass the flagged slots put on each
        token (slots @ fat)."""
        S = np.asarray(slots, np.float32).reshape(self.B, self.LT)
        if how == "mass":
            return np.einsum("bj,bjt->bt", S, self.fat) * self.tokmask
        assert how == "sentence", how
        out = np.zeros((self.B, self.T), np.float32)
        for b in range(self.B):
            js = np.where(S[b] > 0)[0]
            if len(js) == 0:
                continue
            want = {self.source_sentence(b, int(j)) for j in js}
            out[b] = np.isin(self.sent[b], list(want)).astype(np.float32) * self.tokmask[b]
        return out

    def spotlight(self, slots, beta, mode="union"):
        """THE WHEEL'S BIAS (B, 1, LT, T): for each flagged slot j of row
        b, +beta on the tokens of the wanted sentences — the union of the
        row's flagged slots' source sentences ("union") or slot j's own
        ("own"); unflagged slots and rows stay 0. Identical to the
        constructions that lived in _wheel_turn / wheel_bias."""
        S = np.asarray(slots, np.float32).reshape(self.B, self.LT)
        bias = np.zeros((self.B, 1, self.LT, self.T), np.float32)
        for b in range(self.B):
            js = [int(j) for j in np.where(S[b] > 0)[0]]
            if not js:
                continue
            sents = {j: self.source_sentence(b, j) for j in js}
            for j in js:
                want = set(sents.values()) if mode == "union" else {sents[j]}
                bias[b, 0, j, :] = np.where(np.isin(self.sent[b], list(want)), beta, 0.0).astype(np.float32)
        return bias

    # --- tokens -> slots ---------------------------------------------------
    def to_slots(self, tokens, how="mass"):
        """Project a token certificate (B, T) onto the slots (B, LT).
        "mass": each slot's attention mass on the flagged tokens (fat @
        tokens); "argmax": 1 where the slot's source token is flagged."""
        Tk = np.asarray(tokens, np.float32).reshape(self.B, self.T) * self.tokmask
        if how == "mass":
            return np.einsum("bjt,bt->bj", self.fat, Tk)
        assert how == "argmax", how
        out = np.zeros((self.B, self.LT), np.float32)
        for b in range(self.B):
            for j in range(self.LT):
                out[b, j] = Tk[b, self.source_token(b, j)] > 0
        return out

    # --- the crossing ------------------------------------------------------
    def project(self, cert, to_tokens="sentence", to_slots="mass"):
        """Fill the certificate's other side through the bridge (the side
        it was born on is never overwritten). Returns the certificate."""
        if cert.slots is not None and cert.tokens is None:
            cert.tokens = self.to_tokens(cert.slots, to_tokens)
        elif cert.tokens is not None and cert.slots is None:
            cert.slots = self.to_slots(cert.tokens, to_slots)
        return cert


# ---------------------------------------------------------------------------
# THE IN-GRAPH ROADS (tinygrad tensors; pure functions, no head globals).
# ---------------------------------------------------------------------------
def claims_bias(prev_at, melted, B, LT, beta, tau):
    """T2 — THE CLAIM MASK (2026-09-13): prev_at (B, LT, T) the last
    breath's head-mean slots<-tokens attention; melted (B, LT) or None.
    Returns (B, 1, LT, T): -beta where the token is claimed (> tau) by
    ANOTHER, unmelted slot."""
    C = (prev_at > tau).float()
    if melted is not None:
        C = C * (1.0 - melted.reshape(B, LT, 1))
    anyc = C.sum(1, keepdim=True)                     # (B, 1, T)
    other = ((anyc - C) > 0).float()                  # claimed by someone else
    return (other * -beta).reshape(B, 1, LT, -1)


def melt(cur, m, amp, noise, gate=1.0):
    """THE MELT (2026-09-13, the targeted whip): cur (B, LT, HW) the state
    entering a breath; m (B, LT) 1.0 on the slots to melt; noise (B, LT,
    HW) seeded unit noise; gate = the content-plane gate (clock planes 0)
    or 1.0. On melted slots the content becomes amp x ||slot|| x unit
    noise (amp 0 -> zeros); the clock planes and other slots untouched."""
    B, LT, HW = (int(x) for x in cur.shape)
    n = noise * gate
    nn = (n * n).sum(-1, keepdim=True).sqrt() + 1e-6
    nc = (cur * cur).sum(-1, keepdim=True).sqrt()
    fresh = n / nn * nc * amp                         # the melted content
    mm = m.reshape(B, LT, 1)
    return cur + mm * (fresh - cur * gate)            # content replaced; clock (gate=0) kept


def nogood_apply(onp, nogoods):
    """THE NOGOOD: refuted choices are not re-committed — mask them out of
    the head logits in place. nogoods: [(b, j, field, idx)]."""
    for b, j, f, idx in nogoods:
        onp[f][b, j, idx] = -1e9
    return onp


def same_sentence(sent, tokmask, B, T):
    """The token loop's WITHIN-MEDIUM mask (B, 1, T, T): 1 where query and
    key tokens share a sentence and the key is a real token."""
    same = (sent.reshape(B, 1, T, 1) == sent.reshape(B, 1, 1, T)).float()
    return same * tokmask.reshape(B, 1, 1, T)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    B, LT, T = 3, 5, 12
    fat = rng.random((B, LT, T)).astype(np.float32); fat /= fat.sum(-1, keepdims=True)
    sent = np.repeat(np.arange(4), 3)[None].repeat(B, 0); sent[1] = np.repeat([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], 1)
    tokmask = np.ones((B, T), np.float32); tokmask[2, 10:] = 0
    br = Bridge(fat, sent, tokmask)
    slots = np.zeros((B, LT), np.float32); slots[0, 1] = 1; slots[0, 3] = 1; slots[2, 4] = 1
    # spotlight == the wheel's construction, both modes
    for mode in ("union", "own"):
        ref = np.zeros((B, 1, LT, T), np.float32)
        for b in range(B):
            js = [j for j in range(LT) if slots[b, j] > 0]
            sents = {j: int(sent[b, min(int(fat[b, j].argmax()), T - 1)]) for j in js}
            for j in js:
                want = set(sents.values()) if mode == "union" else {sents[j]}
                ref[b, 0, j, :] = np.where(np.isin(sent[b], list(want)), 2.5, 0.0)
        assert np.array_equal(br.spotlight(slots, 2.5, mode), ref), mode
    # to_tokens("sentence") is the union spotlight's carry, masked by real tokens
    tk = br.to_tokens(slots)
    sp = br.spotlight(slots, 1.0, "union")
    for b in range(B):
        js = np.where(slots[b] > 0)[0]
        exp = (sp[b, 0, js[0]] if len(js) else np.zeros(T)) * tokmask[b]
        assert np.array_equal(tk[b], exp), b
    assert tk[1].sum() == 0 and tk[2, 10:].sum() == 0
    # mass both ways: adjoint through the same matrix
    m_t = br.to_tokens(slots, "mass"); m_s = br.to_slots(tk, "mass")
    assert np.allclose(m_t, np.einsum("bj,bjt->bt", slots, fat) * tokmask)
    assert np.allclose(m_s, np.einsum("bjt,bt->bj", fat, tk * tokmask))
    assert np.allclose((slots * m_s).sum(), (m_t * tk).sum())           # <S, F t> == <F^T S, t>
    a_s = br.to_slots(tk, "argmax")
    assert a_s[0, 1] == 1 and a_s[0, 3] == 1 and a_s[1].sum() == 0
    # the crossing fills the missing side only
    c = br.project(Certificate("muc", slots=slots, breath=2))
    assert c.born == "slot" and np.array_equal(c.tokens, tk) and np.array_equal(c.slots, slots)
    c2 = br.project(Certificate("locus", tokens=tk))
    assert c2.born == "token" and np.allclose(c2.slots, m_s)
    try:
        Certificate("empty"); raise AssertionError("empty certificate accepted")
    except AssertionError as e:
        assert "certifies nothing" in str(e)
    # the in-graph roads on CPU tensors
    from tinygrad import Tensor
    pa = Tensor(rng.random((B, LT, T)).astype(np.float32)); me = Tensor(slots)
    cb = claims_bias(pa, me, B, LT, 2.0, 0.5).numpy()
    C = (pa.numpy() > 0.5) * (1 - slots[:, :, None]); oth = ((C.sum(1, keepdims=True) - C) > 0)
    assert np.array_equal(cb.reshape(B, LT, T), oth * -2.0)
    cur = Tensor(rng.standard_normal((B, LT, 8)).astype(np.float32)); nz = Tensor(rng.standard_normal((B, LT, 8)).astype(np.float32))
    g = Tensor(np.array([1, 1, 1, 1, 1, 1, 0, 0], np.float32))
    out = melt(cur, me, 0.5, nz, g).numpy(); cn = cur.numpy()
    assert np.allclose(out[slots == 0], cn[slots == 0]) and np.allclose(out[slots > 0][:, 6:], cn[slots > 0][:, 6:])   # untouched slots; clock kept
    assert not np.allclose(out[slots > 0][:, :6], cn[slots > 0][:, :6])
    onp = {"res": np.zeros((B, LT, 4), np.float32)}; nogood_apply(onp, [(0, 1, "res", 2)]); assert onp["res"][0, 1, 2] == -1e9 and onp["res"].sum() == -1e9
    ss = same_sentence(Tensor(sent.astype(np.int32)), Tensor(tokmask), B, T).numpy()
    assert ss.shape == (B, 1, T, T) and ss[0, 0, 0, 1] == 1 and ss[0, 0, 0, 3] == 0 and ss[2, 0, 9, 10] == 0
    print("[loop_bridge] self-test PASS: spotlight == the wheel's construction (union/own), to_tokens carries the "
          "sentences, mass projections are adjoint through fat, argmax carry, project fills the missing side only, "
          "claims/melt/nogood/same_sentence roads on tensors")
