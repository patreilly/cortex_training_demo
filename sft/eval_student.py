"""Offline next-turn evaluation of a student orchestrator on held-out teacher traces.

The fine-tuned model can't be an agent's orchestration model yet, so this
replays each held-out trace turn by turn: the student sees the teacher's
conversation up to an assistant turn and must produce that turn. We score:
  kind_ok   - tool call vs final answer matches the teacher
  tools_ok  - same set of tool names
  ids_ok    - same ticket/customer id arguments on GET_TICKET/LOOKUP_CUSTOMER/CREATE_ESCALATION
Final answers are written out for an LLM judge (see judge step in this file).

  python sft/eval_student.py --label base     --source base
  python sft/eval_student.py --label student  --source checkpoint --job-id <id> --checkpoint <ckpt>
"""

import os
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cortex_training import CortexTrainingClient
from transformers import AutoTokenizer

import convert as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CALL_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
ID_TOOLS = {"GET_TICKET", "LOOKUP_CUSTOMER", "CREATE_ESCALATION"}
ID_KEYS = {"p_ticket_id", "p_customer_id", "p_team"}


def parse_turn(text):
    body = text.split("</think>", 1)[-1]
    calls = [{"name": n, "arguments": {k: v.strip() for k, v in PARAM_RE.findall(p)}}
             for n, p in CALL_RE.findall(body)]
    answer = re.sub(r"<tool_call>.*?</tool_call>", "", body, flags=re.S).replace("<|im_end|>", "").strip()
    return calls, answer


def ids_of(calls):
    return sorted((c["name"], k, str(v).upper()) for c in calls if c["name"] in ID_TOOLS
                  for k, v in c["arguments"].items() if k in ID_KEYS)


def build_cases(holdout, scores, system, traces):
    cases = []
    for line in open(traces):
        tr = json.loads(line)
        cid = tr["request_id"]
        if cid not in holdout or not isinstance(tr.get("final_response"), dict) or tr.get("errors"):
            continue
        try:
            msgs = C.build_messages(tr, system)
        except ValueError:
            continue
        for k, m in enumerate(msgs):
            if m["role"] == "assistant":
                cases.append({"case_id": cid, "intent": scores[cid]["INTENT"], "turn": k,
                              "teacher_pass": scores[cid]["PASS"], "prefix": msgs[:k], "target": m})
    return cases


def deploy(client, args):
    if args.source == "base":
        source = {"type": "base_model", "model": args.model}
    else:
        source = {"type": "checkpoint", "job_id": args.job_id, "checkpoint_id": args.checkpoint}
    spec = {"deployment": {"class": "EVALUATION", "min_replicas": 1, "gpus_per_replica": 1, "gpu_type": "H200"},
            "source": source, "inference": {"max_seq_len": args.max_seq_len}}
    meta = client.create_model_deployment(spec)
    print("deployment", json.dumps(meta), flush=True)
    ready = client.wait_for_model_deployment(meta["deployment"])
    print("ready", json.dumps(ready), flush=True)
    return ready


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--source", choices=["base", "checkpoint"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--job-id")
    ap.add_argument("--checkpoint")
    ap.add_argument("--deployment-id", help="reuse a READY deployment instead of creating one")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep", action="store_true", help="don't drop the deployment")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--traces", default=str(ROOT / "traces" / "raw" / "full.jsonl"),
                    help="trace file from run_traces.py (raw/pilot.jsonl for a pilot run)")
    ap.add_argument("--split", choices=["holdout", "train"], default="holdout",
                    help="train = memorization check that the adapter is actually applied")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    system, tools = C.load_spec()
    scores = {r["CASE_ID"]: r for r in json.load(open(HERE / "teacher_scores.json"))}
    if args.split == "train":
        holdout = {json.loads(l)["case_id"] for l in open(Path(args.data_dir) / "train.jsonl")}
    else:
        holdout = set(json.load(open(Path(args.data_dir) / "holdout_ids.json")))
    cases = build_cases(holdout, scores, system, args.traces)
    if args.limit:
        cases = cases[:args.limit]
    print(f"{len(cases)} assistant turns from {len({c['case_id'] for c in cases})} held-out cases", flush=True)

    client = CortexTrainingClient.from_configs(os.environ.get("SNOWFLAKE_CONNECTION_NAME", "default"), database="CT_DEMO_DB", schema="CORTEX_TRAINING")
    dep = {"deployment": args.deployment_id} if args.deployment_id else deploy(client, args)
    dep_id = dep["deployment"]
    model_field = dep.get("model") or dep.get("served_model_name") or dep_id
    url = f"{client.model_deployment_base_url}/chat/completions"

    def to_openai(prefix):
        """Training messages -> OpenAI chat format (string args, tool_call ids)."""
        out, pending = [], []
        for m in prefix:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                # The endpoint rejects reasoning_content on prior turns (400 unsupported_parameter).
                if m.get("tool_calls"):
                    msg["tool_calls"] = []
                    for c in m["tool_calls"]:
                        cid = f"call_{len(out)}_{len(msg['tool_calls'])}"
                        pending.append(cid)
                        msg["tool_calls"].append({"id": cid, "type": "function", "function": {
                            "name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])}})
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_x",
                            "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def run(case):
        body = {"model": model_field, "messages": to_openai(case["prefix"]), "tools": tools,
                "max_completion_tokens": 2048, "temperature": 0.0}
        t0 = time.time()
        for attempt in range(3):
            try:
                r = client._send("POST", url, json=body).json()
                break
            except Exception as e:  # noqa: BLE001 - record and move on after retries
                err = str(e)[:300]
                time.sleep(2 * (attempt + 1))
        else:
            return {**{k: case[k] for k in ("case_id", "intent", "turn", "teacher_pass")}, "error": err}
        msg = r["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                a = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                a = {}
            calls.append({"name": c["function"]["name"], "arguments": a})
        answer = (msg.get("content") or "").split("</think>")[-1].strip()
        tgt = case["target"]
        t_calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                   for c in tgt.get("tool_calls", [])]
        kind_ok = bool(calls) == bool(t_calls)
        tools_ok = sorted({c["name"] for c in calls}) == sorted({c["name"] for c in t_calls})
        return {"case_id": case["case_id"], "intent": case["intent"], "turn": case["turn"],
                "teacher_pass": case["teacher_pass"], "is_final": not t_calls,
                "kind_ok": kind_ok, "tools_ok": tools_ok, "ids_ok": ids_of(calls) == ids_of(t_calls),
                "student_tools": [c["name"] for c in calls], "teacher_tools": [c["name"] for c in t_calls],
                "student_answer": answer if not calls else None,
                "teacher_answer": tgt["content"] if not t_calls else None,
                "user_request": case["prefix"][1]["content"],
                "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                "latency_s": round(time.time() - t0, 2)}

    out_path = HERE / "eval" / f"{args.label}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    try:
        with ThreadPoolExecutor(args.workers) as pool, open(out_path, "w") as f:
            for n, res in enumerate(pool.map(run, cases), 1):
                f.write(json.dumps(res) + "\n")
                if n % 50 == 0:
                    print(f"{n}/{len(cases)}", flush=True)
    finally:
        if not args.keep and not args.deployment_id:
            client.drop_model_deployment(dep_id)
            print(f"dropped {dep_id}", flush=True)

    rows = [json.loads(l) for l in open(out_path)]
    ok = [r for r in rows if "error" not in r]
    summ = lambda key, rs: round(100 * sum(r[key] for r in rs) / max(1, len(rs)), 1)
    print(json.dumps({"label": args.label, "turns": len(rows), "errors": len(rows) - len(ok),
                      "kind_ok": summ("kind_ok", ok), "tools_ok": summ("tools_ok", ok), "ids_ok": summ("ids_ok", ok),
                      "avg_completion_tokens": round(sum(r["completion_tokens"] or 0 for r in ok) / max(1, len(ok)), 1)},
                     indent=1))


if __name__ == "__main__":
    main()
