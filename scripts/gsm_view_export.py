"""AG3 part 2, THE VIEW-1 STATES EXPORT (ledger 2026-11-04): the mbr8 reader's word states for the view texts. GPU (DEV=PCI+AMD) in the chain.

A wrapper around scripts/distill_reader.py --gsm (which is edited by nobody here): it injects a FAKE `gsm_trunk_states` module into
sys.modules whose texts() serves the VIEW texts (.cache/gsmb_data_v1.json) and whose OUT_IDX is the view index (.cache/gsmb_trunk_index_v1.json,
written by gsm_view_data.py with the same uniq/align layout), then runpy-runs distill_reader.py as __main__ (works before and after its
__main__ guard). The export's own assertion then re-checks that the layout it computes equals the view index (a misaligned item fails loudly).
Run with the reader env: DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_TAG=mbr2x4.
usage: gsm_view_export.py OUT.npy   env AG3_DATA / AG3_IDX override the data / index (smoke); in smoke the index is (re)written by this
wrapper from the same procedure, so the check is then trivial.
"""
import os, runpy, sys, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import json
import gsm_trunk_states as REAL

DATA = os.environ.get("AG3_DATA", ".cache/gsmb_data_v1.json"); IDX = os.environ.get("AG3_IDX", ".cache/gsmb_trunk_index_v1.json")
out = sys.argv[1]
assert out.endswith("_v1.npy") or os.environ.get("AG3_SMOKE_OK") == "1", "view-1 states must be named *_v1.npy"
assert os.path.abspath(out) != os.path.abspath(".cache/gsmb_mbr8_states.npy")


def texts():
    Dt = json.load(open(DATA))
    return Dt, {sp: [it["text"] for it in its] for sp, its in Dt.items()}


fake = types.ModuleType("gsm_trunk_states")
fake.texts, fake.align, fake.OUT_IDX = texts, REAL.align, IDX
sys.modules["gsm_trunk_states"] = fake
sys.argv = ["distill_reader.py", "--gsm", out]
runpy.run_path("scripts/distill_reader.py", run_name="__main__")
