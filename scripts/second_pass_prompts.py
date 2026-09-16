"""second_pass_prompts.py — after the first pass: rebuild prompts for the REFUSED
rows only, under the loosened silver rulebook (protocol v2 in the packet).
Writes .cache/sonnet_top300/pass2_prompt_NN.txt (10 rows each) + prints counts."""
import json, glob
pk = json.load(open(".cache/annotation_packet_top300.json")); by_id = {r["id"]: r for r in pk["rows"]}
refused = []
for f in sorted(glob.glob(".cache/sonnet_top300/returns_*.jsonl")):
    for l in open(f):
        try: o = json.loads(l)
        except Exception: continue
        if "refuse" in o and o.get("id") in by_id: refused.append(by_id[o["id"]])
seen = set(); refused = [r for r in refused if not (r["id"] in seen or seen.add(r["id"]))]
ex_txt = "\n\n".join(f"EXAMPLE {k + 1}\ntext: {e['text']}\nannotation: " + json.dumps({"n_vars": e["n_vars"], "query_var": e["query_var"], "factors": e["factors"], "mentions": e["mentions"]}) for k, e in enumerate(pk["examples"]))
head = pk["protocol"] + "\n\nWORKED EXAMPLES (synthetic rows; every span and mention is an EXACT substring copied from the text):\n\n" + ex_txt + "\n\nRETURN one JSON object per line, exactly: {\"id\": <id>, \"n_vars\": int, \"query_var\": int, \"factors\": [...], \"mentions\": {...}}. No prose, no code fences. These rows were refused in a first pass under stricter rules (a 300 cap, one division); under THESE rules most should now be annotatable. If a row still cannot be annotated (a decimal value, a value above 9999, an operation outside add/sub/mul/div/fdiv, or knowledge the text does not name), return {\"id\": <id>, \"refuse\": \"<reason>\"}.\n"
B = 10; n = 0
for b in range(0, len(refused), B):
    chunk = refused[b:b + B]
    body = "\n\n".join(f"ROW id={r['id']}\ntext: {r['text']}\nanswer (the key your graph must force; the arithmetic in it is NOT to be copied): {r['answer_field'].split('####')[-1].strip()}" for r in chunk)
    open(f".cache/sonnet_top300/pass2_prompt_{b // B + 1:02d}.txt", "w").write(head + "\nROWS\n\n" + body + "\n"); n += 1
print(f"[pass2] {len(refused)} refused rows -> {n} prompt files (.cache/sonnet_top300/pass2_prompt_NN.txt)")
