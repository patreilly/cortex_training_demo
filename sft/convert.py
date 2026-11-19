"""Convert Cortex Agent teacher traces into Qwen chat-format SFT examples.

Pipeline per trace:
  1. keep only traces that PASSED the benchmark scorer (teacher_scores.json)
  2. hold out ~15% of case ids (deterministic hash) for student evaluation
  3. rebuild the conversation: system (agent instructions) -> user -> assistant
     (thinking + tool calls) -> tool results -> ... -> final assistant answer
  4. pseudonymize identifiers with values from reserved, never-issued namespaces
  5. render with the Qwen chat template, tokenize, and build next-token labels
     where only assistant tokens are trained (everything else = -100)
  6. leak scan: fail if any real identifier survives in the training text

Outputs (sft/data/):
  train.jsonl     {"case_id","intent","input_ids","labels","n_tokens","n_trained"}
  holdout_ids.json, stats.json, preview.txt (one rendered example)
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IGNORE = -100

# Reserved namespaces: production never issues these ranges, so a surrogate can
# never collide with a current or future real identifier.
FAKE_CUSTOMER_RANGE = range(9000, 10000)   # C9000-C9999
FAKE_TICKET_RANGE = range(80000, 90000)    # T80000-T89999
FAKE_BRANDS = [
    "Quillfern", "Brasswick", "Tamberlyn", "Orrowind", "Velmonte", "Cindrel",
    "Halvorne", "Pexmoor", "Lumbrecht", "Saxonby", "Drevalle", "Ostrava",
    "Kelderan", "Myrtlebank", "Fenwhistle", "Arkenvale", "Corvallon", "Zephrine",
]

CUST_RE = re.compile(r"\bC(\d{4})\b")
TICK_RE = re.compile(r"\bT(\d{5})\b")
CSM_RE = re.compile(r"\bcsm(\d+)@northwind\.example\b")
ACV_RE = re.compile(r'("annual_contract_value\\?"\s*:\s*)(\d+)')

# Blocks the platform renders itself; the orchestrator never emits them.
DROP_BLOCKS = {"table", "chart", "suggested_queries"}


def load_spec():
    spec = yaml.safe_load(open(ROOT / "agent" / "SUPPORT_AGENT_TEACHER.agent.yaml"))
    instr = spec["instructions"]
    system = instr["orchestration"].strip() + "\n\nResponse style:\n" + instr["response"].strip()
    tools = []
    # Search/analyst tools have no input_schema in the spec; the platform injects
    # these parameters, inferred here from the teacher's calls.
    inferred = {
        "KB_SEARCH": {"query": "string", "limit": "integer", "columns": "array",
                      "analytical_search": "boolean", "num_chunks_per_doc": "integer"},
        "SUPPORT_ANALYTICS": {"pruning_question": "string"},
    }
    for t in spec["tools"]:
        ts = t["tool_spec"]
        params = ts.get("input_schema") or {"type": "object", "properties": {
            k: {"type": v} for k, v in inferred.get(ts["name"], {}).items()}}
        tools.append({"type": "function", "function": {
            "name": ts["name"], "description": ts.get("description", ""), "parameters": params}})
    # Platform built-ins the teacher also calls; schemas inferred from traces.
    builtins = {
        "system_execute_sql": ("Executes SQL generated against a semantic model.",
                               {"semantic_model": "string", "sql": "string"}),
        "data_to_chart": ("Renders a Vega-Lite chart from the latest SQL result.", {"chart": "string"}),
        "server_skill": ("Loads platform instructions for a named skill.", {"skill_name": "string"}),
    }
    for name, (desc, props) in builtins.items():
        tools.append({"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": {k: {"type": v} for k, v in props.items()}}}})
    return system, tools


class Pseudonymizer:
    """Consistent within one trace, independent across traces."""

    def __init__(self, real_brands, seed):
        self.rng = random.Random(seed)
        self.cust, self.tick, self.brand = {}, {}, {}
        self.real_brands = real_brands
        self.brand_re = re.compile(r"\b(" + "|".join(map(re.escape, sorted(real_brands))) + r")\b")
        self.fake_brands = [b for b in FAKE_BRANDS if b not in real_brands]

    def _pick(self, used, space):
        while True:
            v = self.rng.choice(space)
            if v not in used:
                return v

    def __call__(self, text):
        def cust(m):
            k = m.group(1)
            if k not in self.cust:
                self.cust[k] = f"{self._pick(set(self.cust.values()), FAKE_CUSTOMER_RANGE):04d}"
            return "C" + self.cust[k]

        def tick(m):
            k = m.group(1)
            if k not in self.tick:
                self.tick[k] = f"{self._pick(set(self.tick.values()), FAKE_TICKET_RANGE):05d}"
            return "T" + self.tick[k]

        def brand(m):
            k = m.group(1)
            if k not in self.brand:
                self.brand[k] = self._pick(set(self.brand.values()), self.fake_brands)
            return self.brand[k]

        text = CUST_RE.sub(cust, text)
        text = TICK_RE.sub(tick, text)
        text = self.brand_re.sub(brand, text)
        text = CSM_RE.sub(lambda m: f"csm{self.rng.randint(100, 999)}@support.example", text)
        # Coarsen contract values to the nearest 10k so exact amounts don't carry over.
        text = ACV_RE.sub(lambda m: m.group(1) + str(round(int(m.group(2)), -4)), text)
        # Company names embed the customer number ("Acme Foods 212"); re-sync it
        # with the fake customer id so names and ids stay coherent.
        for real, fake in self.cust.items():
            text = re.sub(rf"(\b(?:{'|'.join(map(re.escape, self.fake_brands))}) \w+ ){int(real)}\b",
                          rf"\g<1>{int(fake)}", text)
        return text


def tool_result_text(tr):
    parts = []
    for c in tr.get("content", []):
        if c.get("type") == "json":
            j = c["json"]
            # Function results arrive as a JSON string under "result".
            parts.append(j["result"] if isinstance(j.get("result"), str) else json.dumps(j, ensure_ascii=False))
        elif c.get("type") == "text":
            parts.append(c.get("text", ""))
    return "\n".join(parts)


def build_messages(trace, system):
    """Map the agent's final response content into chat messages."""
    content = trace["final_response"]["content"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": trace["request_text"]}]
    cur = None  # assistant message under construction

    def flush():
        nonlocal cur
        if cur is not None:
            msgs.append(cur)
            cur = None

    for block in content:
        t = block.get("type")
        if t in DROP_BLOCKS:
            continue
        if t == "thinking":
            if cur is not None and (cur.get("tool_calls") or cur["content"]):
                flush()
            cur = cur or {"role": "assistant", "content": "", "reasoning_content": ""}
            cur["reasoning_content"] = (cur["reasoning_content"] + "\n" + block["thinking"]["text"]).strip()
        elif t == "tool_use":
            tu = block["tool_use"]
            cur = cur or {"role": "assistant", "content": "", "reasoning_content": ""}
            cur.setdefault("tool_calls", []).append(
                {"type": "function", "function": {"name": tu["name"], "arguments": tu.get("input", {})}})
        elif t == "tool_result":
            flush()
            msgs.append({"role": "tool", "content": tool_result_text(block["tool_result"])})
        elif t == "text":
            if cur is not None and cur.get("tool_calls"):
                flush()
            cur = cur or {"role": "assistant", "content": "", "reasoning_content": ""}
            cur["content"] = (cur["content"] + block["text"]).strip()
        else:
            raise ValueError(f"unknown block type {t}")
    flush()
    if msgs[-1]["role"] != "assistant" or not msgs[-1]["content"]:
        raise ValueError("trace does not end with an assistant answer")
    return msgs


def pseudonymize_messages(msgs, pz):
    out = []
    for m in msgs:
        m = dict(m)
        if m["role"] != "system":
            for k in ("content", "reasoning_content"):
                if m.get(k):
                    m[k] = pz(m[k])
            if m.get("tool_calls"):
                m["tool_calls"] = json.loads(pz(json.dumps(m["tool_calls"], ensure_ascii=False)))
        out.append(m)
    return out


def assistant_labels(ids, tok, mask_empty_think=True):
    """HF-style labels (labels[i] = ids[i]; the server shifts). Only assistant spans count.

    Verified with sft/probe_labels.py: pre-shifted labels give loss ~15.6 on the
    server, unshifted give ~0.28, so the server applies the shift itself.

    mask_empty_think: the teacher often calls a follow-up tool without reasoning,
    which renders as an empty "<think>\\n\\n</think>\\n\\n". Training on that marker
    taught the student to skip reasoning and call a tool on every turn, so the
    marker is excluded from the loss; the tool call after it is still trained.
    """
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    header = tok.encode("assistant\n", add_special_tokens=False)
    empty_think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    train = [False] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] == im_start and ids[i + 1:i + 1 + len(header)] == header:
            j = start = i + 1 + len(header)
            while j < len(ids) and ids[j] != im_end:
                train[j] = True
                j += 1
            if j < len(ids):
                train[j] = True  # learn to emit <|im_end|>
            if mask_empty_think and ids[start:start + len(empty_think)] == empty_think:
                for k in range(start, start + len(empty_think)):
                    train[k] = False
            i = j
        i += 1
    return [tid if t else IGNORE for tid, t in zip(ids, train)]


def per_turn_variants(msgs):
    """One message list per assistant turn, ending at that turn.

    Earlier assistant turns lose their reasoning: the serving endpoint rejects
    reasoning_content on prior turns, so this is the history the model will see.
    """
    out = []
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = [{kk: vv for kk, vv in p.items() if kk != "reasoning_content"} if p["role"] == "assistant" else p
                  for p in msgs[:k]]
        out.append(prefix + [m])
    return out


def last_span_only(labels):
    """Keep only the final contiguous trained span (the target turn)."""
    end = max((i for i, l in enumerate(labels) if l != IGNORE), default=-1)
    if end < 0:
        return labels
    start = end
    while start > 0 and labels[start - 1] != IGNORE:
        start -= 1
    return [l if start <= i <= end else IGNORE for i, l in enumerate(labels)]


def leak_scan(text, real_ids, real_brands):
    hits = []
    for m in CUST_RE.finditer(text):
        if int(m.group(1)) not in FAKE_CUSTOMER_RANGE:
            hits.append(m.group(0))
    for m in TICK_RE.finditer(text):
        if int(m.group(1)) not in FAKE_TICKET_RANGE:
            hits.append(m.group(0))
    hits += [b for b in real_brands if re.search(rf"\b{re.escape(b)}\b", text)]
    hits += [i for i in real_ids if i in text]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--holdout-pct", type=int, default=15)
    ap.add_argument("--no-pseudonymize", action="store_true")
    ap.add_argument("--drop-thinking", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    ap.add_argument("--traces", default=str(ROOT / "traces" / "raw" / "full.jsonl"),
                    help="trace file from run_traces.py (raw/pilot.jsonl for a pilot run)")
    ap.add_argument("--per-turn", action="store_true",
                    help="one example per assistant turn; prior assistant turns carry no reasoning "
                         "(matches what the serving endpoint accepts) and only the last turn is trained")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    system, tools = load_spec()
    scores = {r["CASE_ID"]: r for r in json.load(open(HERE / "teacher_scores.json"))}
    customers = json.load(open(HERE / "real_customers.json"))
    real_brands = {c["COMPANY_NAME"].split()[0] for c in customers}
    real_names = [c["COMPANY_NAME"] for c in customers]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stats, holdout, lengths = Counter(), [], []
    preview_done = False
    with open(out_dir / "train.jsonl", "w") as fout:
        for line in open(args.traces):
            tr = json.loads(line)
            cid = tr["request_id"]
            stats["traces"] += 1
            sc = scores.get(cid)
            if sc is None:
                stats["skip_not_scored"] += 1   # adversarial-intent requests are never trained on
                continue
            if int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 100 < args.holdout_pct:
                holdout.append(cid)
                stats["holdout"] += 1
                continue
            if not sc["PASS"]:
                stats["skip_failed_score"] += 1
                continue
            if not isinstance(tr.get("final_response"), dict) or tr.get("errors"):
                stats["skip_error"] += 1
                continue
            if any(b.get("type") == "tool_use" and b["tool_use"]["name"] == "read"
                   for b in tr["final_response"]["content"]):
                stats["skip_read_tool"] += 1
                continue
            try:
                msgs = build_messages(tr, system)
            except ValueError:
                stats["skip_malformed"] += 1
                continue
            if args.drop_thinking:
                for m in msgs:
                    m.pop("reasoning_content", None)
            if not args.no_pseudonymize:
                pz = Pseudonymizer(real_brands, seed=hashlib.sha1(("pz" + cid).encode()).hexdigest())
                msgs = pseudonymize_messages(msgs, pz)
                convo = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs[1:])
                hits = leak_scan(convo, [], real_brands) + [n for n in real_names if n in convo]
                if hits:
                    raise SystemExit(f"LEAK in {cid}: {sorted(set(hits))[:10]}")
            variants = per_turn_variants(msgs) if args.per_turn else [msgs]
            for k, vmsgs in enumerate(variants):
                text = tok.apply_chat_template(vmsgs, tools=tools, tokenize=False)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) > args.max_seq_len:
                    stats["skip_too_long"] += 1
                    continue
                labels = assistant_labels(ids, tok, mask_empty_think=not args.drop_thinking)
                if args.per_turn:
                    labels = last_span_only(labels)
                n_tr = sum(l != IGNORE for l in labels)
                if n_tr == 0:
                    stats["skip_nothing_trained"] += 1
                    continue
                lengths.append(len(ids))
                stats["train"] += 1
                stats[f"intent:{sc['INTENT']}"] += 1
                fout.write(json.dumps({"case_id": cid, "turn": k, "intent": sc["INTENT"], "input_ids": ids,
                                       "labels": labels, "n_tokens": len(ids), "n_trained": n_tr}) + "\n")
                if not preview_done:
                    (out_dir / "preview.txt").write_text(text)
                    preview_done = True

    lengths.sort()
    pct = lambda p: lengths[min(len(lengths) - 1, int(p / 100 * len(lengths)))] if lengths else 0
    summary = {**dict(stats), "len_p50": pct(50), "len_p90": pct(90), "len_p99": pct(99),
               "len_max": lengths[-1] if lengths else 0, "total_tokens": sum(lengths),
               "pseudonymized": not args.no_pseudonymize, "model": args.model}
    json.dump(sorted(holdout), open(out_dir / "holdout_ids.json", "w"), indent=1)
    json.dump(summary, open(out_dir / "stats.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
