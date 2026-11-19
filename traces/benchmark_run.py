"""Run any Cortex Agent over the benchmark cases and record results in AGENT_BENCHMARK_RUNS.

Usage:
  python benchmark_run.py --agent CT_DEMO_DB.CORTEX_TRAINING.SUPPORT_AGENT_TEACHER --run-name teacher_v1
  python benchmark_run.py ... --case-set adversarial
  python benchmark_run.py ... --per-intent 20      # stratified sample for quick comparisons

Same cases + same tools for every agent, so runs are directly comparable. Re-running a run name
skips cases already recorded locally (resumable).

What this benchmark measures, and how the student was evaluated
---------------------------------------------------------------
This script runs end to end against a live Cortex Agent, and in this demo that agent is the
teacher (SUPPORT_AGENT_TEACHER, orchestrated by Claude Sonnet 4.5):

1. Cases come from BENCHMARK_CASES, which has two sets:
     trace_requests  in-distribution support requests, like those used to generate teacher traces
     adversarial     prompt injection, instruction override, policy edge cases, etc.
   Each case carries ground truth: required/allowed tools, forbidden escalation teams, facts the
   answer must mention, and the expected behavior (answer / clarify / decline / escalate_or_confirm).
2. This script sends every case to the agent through the agent:run REST API and saves the full
   trace (tool calls, tool results, final answer), latency, and orchestration token usage.
3. score_run.py then grades each trace: deterministic SQL checks against the ground truth plus an
   LLM judge from a different model family. A case passes only if every check passes.

The teacher's scores set the quality bar, and the teacher's passing traces become the student's
training data (see sft/convert.py).

The fine-tuned student can't be selected as a Cortex Agent orchestration model yet, so it was
not run through this script. Instead, sft/eval_student.py evaluates it offline by next-turn
replay. It deploys the model with Cortex Training, takes held-out teacher traces (cases never
seen in training), feeds the student the teacher's conversation up to each assistant turn, and
compares the student's turn with the teacher's: right kind of turn (tool call vs. answer), same
tools, same ticket/customer IDs, with final answers sent to an LLM judge. The base model goes
through the same replay as a baseline. This measures single-turn agreement with the teacher, not
end-to-end agent behavior. Once fine-tuned models can orchestrate agents, a student agent can be
passed to --agent here and scored with score_run.py like the teacher.
"""
import argparse
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_client import connect, run_agent
from run_traces import summarize

OUT_DIR = pathlib.Path(__file__).parent / "benchmark_runs"
OUT_DIR.mkdir(exist_ok=True)


def orchestration_usage(final):
    """Extract token usage for the orchestration model from a final `response` event.

    Returns (model_name, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens), all
    None if the response is missing or reports no orchestration usage. Only the orchestration role
    is counted because that is the cost a student model would replace.
    """
    usage = ((final or {}).get("metadata") or {}).get("usage") or {}
    for u in usage.get("tokens_consumed", []):
        if u.get("model_role") == "orchestration":
            inp, out = u.get("input_tokens", {}), u.get("output_tokens", {})
            return u.get("model_name"), inp.get("total"), inp.get("cache_read"), inp.get("cache_write"), out.get("total")
    return None, None, None, None, None


def load_cases(cur, case_set, per_intent):
    """Load benchmark cases as dicts, optionally filtered to one case set.

    If per_intent > 0, take the first N cases per (case set, intent), ordered by CASE_ID, so a
    small sample still covers every intent and is the same sample for every agent.
    """
    sql = "SELECT CASE_ID, CASE_SET, INTENT, REQUEST_TEXT FROM CT_DEMO_DB.CORTEX_TRAINING.BENCHMARK_CASES"
    params = []
    if case_set != "all":
        sql += " WHERE CASE_SET = %s"
        params.append(case_set)
    if per_intent:
        sql += " QUALIFY ROW_NUMBER() OVER (PARTITION BY CASE_SET, INTENT ORDER BY CASE_ID) <= %s"
        params.append(per_intent)
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def run_case(conn, host, agent, case, attempts=2):
    """Send one benchmark case to the agent and return a JSON-serializable result record.

    Retries once if the run produced no final response or emitted an error event, so transient
    service failures are not scored as model failures. The record keeps the full trace for scoring.
    """
    for _ in range(attempts):
        res = run_agent(conn, host, case["REQUEST_TEXT"], agent=tuple(agent.split(".")))
        final, thinking, errors, kept = summarize(res["events"])
        if final is not None and not errors:
            break
    model, tin, cread, cwrite, tout = orchestration_usage(final)
    return {
        "case_id": case["CASE_ID"], "case_set": case["CASE_SET"], "intent": case["INTENT"],
        "request_text": case["REQUEST_TEXT"], "status_code": res["status_code"], "http_error": res.get("error"),
        "elapsed_s": round(res["elapsed_s"], 2), "errors": errors, "thinking": thinking,
        "final_response": final, "events": kept, "orchestration_model": model,
        "orch_input_tokens": tin, "orch_cache_read_tokens": cread, "orch_cache_write_tokens": cwrite,
        "orch_output_tokens": tout,
    }


def upload(conn, path, run_name, agent):
    """Load the local JSONL results into AGENT_BENCHMARK_RUNS, replacing any rows for run_name.

    The whole record is stored in TRACE (VARIANT) for score_run.py; HAS_ERROR marks runs with an
    HTTP error, an error event, or no final answer.
    """
    cur = conn.cursor()
    cur.execute("USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING")
    cur.execute("CREATE TEMPORARY STAGE IF NOT EXISTS BENCH_UPLOAD")
    cur.execute("CREATE TEMPORARY FILE FORMAT IF NOT EXISTS BENCH_JSON TYPE = JSON")
    cur.execute(f"PUT 'file://{path}' @BENCH_UPLOAD OVERWRITE = TRUE AUTO_COMPRESS = TRUE")
    cur.execute("DELETE FROM AGENT_BENCHMARK_RUNS WHERE RUN_NAME = %s", (run_name,))
    cur.execute(
        "INSERT INTO AGENT_BENCHMARK_RUNS (RUN_NAME, AGENT, ORCHESTRATION_MODEL, CASE_ID, CASE_SET, STATUS_CODE, "
        " ELAPSED_S, HAS_ERROR, ORCH_INPUT_TOKENS, ORCH_CACHE_READ_TOKENS, ORCH_CACHE_WRITE_TOKENS, ORCH_OUTPUT_TOKENS, TRACE) "
        "SELECT %s, %s, $1:orchestration_model, $1:case_id, $1:case_set, $1:status_code, $1:elapsed_s, "
        "  ARRAY_SIZE($1:errors) > 0 OR $1:status_code <> 200 OR $1:final_response IS NULL, "
        "  $1:orch_input_tokens, $1:orch_cache_read_tokens, $1:orch_cache_write_tokens, $1:orch_output_tokens, $1 "
        "FROM @BENCH_UPLOAD (FILE_FORMAT => 'BENCH_JSON', PATTERN => %s)",
        (run_name, agent, f".*{path.name}.*"),
    )
    cur.execute("SELECT COUNT(*), COUNT_IF(HAS_ERROR) FROM AGENT_BENCHMARK_RUNS WHERE RUN_NAME = %s", (run_name,))
    print(f"AGENT_BENCHMARK_RUNS[{run_name}] rows, errors:", cur.fetchone())


def main():
    """Parse arguments, run all pending cases in parallel, and upload the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="DB.SCHEMA.AGENT")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--case-set", default="all", choices=["all", "trace_requests", "adversarial"])
    ap.add_argument("--per-intent", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # Resume support: keep successful records from a previous attempt at this run name.
    path = OUT_DIR / f"{args.run_name}.jsonl"
    done = set()
    if path.exists():
        good = [json.loads(l) for l in path.open() if l.strip()]
        good = [r for r in good if r.get("final_response") is not None and not r.get("errors")]
        with path.open("w") as f:  # drop failed records so they are retried
            for r in good:
                f.write(json.dumps(r, default=str) + "\n")
        done = {r["case_id"] for r in good}
    conn, host = connect()
    cases = [c for c in load_cases(conn.cursor(), args.case_set, args.per_intent) if c["CASE_ID"] not in done]
    print(f"{len(cases)} cases to run for {args.agent} -> {path}")

    # Results are written as they finish (flushed per line), so an interrupted run loses little.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a") as out:
        futures = {pool.submit(run_case, conn, host, args.agent, c): c for c in cases}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                out.write(json.dumps(fut.result(), default=str) + "\n")
                out.flush()
            except Exception as exc:
                print(f"FAIL {futures[fut]['CASE_ID']}: {exc}")
            if i % 25 == 0:
                print(f"progress: {i}/{len(cases)}")
    upload(conn, path, args.run_name, args.agent)


if __name__ == "__main__":
    main()
