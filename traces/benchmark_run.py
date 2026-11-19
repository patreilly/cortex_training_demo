"""Run any Cortex Agent over the benchmark cases and record results in AGENT_BENCHMARK_RUNS.

Usage:
  python benchmark_run.py --agent CT_DEMO_DB.CORTEX_TRAINING.SUPPORT_AGENT_TEACHER --run-name teacher_v1
  python benchmark_run.py --agent DB.SCHEMA.SUPPORT_AGENT_STUDENT --run-name qwen8b_lora_v1 --case-set adversarial
  python benchmark_run.py ... --per-intent 20      # stratified sample for quick comparisons

Same cases + same tools for every agent, so runs are directly comparable. Re-running a run name
skips cases already recorded locally (resumable).
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
    usage = ((final or {}).get("metadata") or {}).get("usage") or {}
    for u in usage.get("tokens_consumed", []):
        if u.get("model_role") == "orchestration":
            inp, out = u.get("input_tokens", {}), u.get("output_tokens", {})
            return u.get("model_name"), inp.get("total"), inp.get("cache_read"), inp.get("cache_write"), out.get("total")
    return None, None, None, None, None


def load_cases(cur, case_set, per_intent):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="DB.SCHEMA.AGENT")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--case-set", default="all", choices=["all", "trace_requests", "adversarial"])
    ap.add_argument("--per-intent", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

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
