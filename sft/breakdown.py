"""Split next-turn eval results into tool turns vs answer turns, overall and per intent.

  python sft/breakdown.py base_qwen35_4b_v2 student_qwen35_4b_v3 student_qwen35_4b_v4
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent / "eval"


def pct(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def summarize(label):
    rows = [json.loads(l) for l in open(EVAL / f"{label}.jsonl")]
    rows = [r for r in rows if "kind_ok" in r]
    tool = [r for r in rows if r["teacher_tools"]]
    ans = [r for r in rows if not r["teacher_tools"]]
    out = {"label": label, "turns": len(rows),
           "kind": pct([r["kind_ok"] for r in rows]),
           "tools": pct([r["tools_ok"] for r in rows]),
           "ids": pct([r["ids_ok"] for r in rows]),
           "tool_turns_right_tool": pct([r["tools_ok"] for r in tool]),
           "tool_turns_called_any": pct([bool(r["student_tools"]) for r in tool]),
           "answer_turns_answered": pct([not r["student_tools"] for r in ans]),
           "n_tool": len(tool), "n_answer": len(ans),
           "avg_completion_tokens": round(sum(r.get("completion_tokens") or 0 for r in rows) / len(rows), 1)}
    by = defaultdict(list)
    for r in rows:
        by[r["intent"]].append(r["kind_ok"] and r["tools_ok"])
    out["intent_turn_ok"] = {k: pct(v) for k, v in sorted(by.items())}
    return out


if __name__ == "__main__":
    print(json.dumps([summarize(l) for l in sys.argv[1:]], indent=1))
