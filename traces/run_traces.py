"""Generate teacher-agent traces for TRACE_REQUESTS.

Usage:
  python run_traces.py --pilot 10        # N requests per intent
  python run_traces.py --all             # every request not yet traced

Each trace keeps the full final `response` event (tool calls, tool results, text),
the concatenated thinking, every non-delta event, latency, and any error. Raw traces
are kept intact so the SFT format can be re-derived later without re-running the agent.
"""
import argparse
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_client import AGENT, connect, run_agent

OUT_DIR = pathlib.Path(__file__).parent / "raw"
OUT_DIR.mkdir(exist_ok=True)
AGENT_FQN = ".".join(AGENT)
DELTA_EVENTS = {"response.text.delta", "response.thinking.delta"}

_lock = threading.Lock()


def summarize(events):
    final = next((e["data"] for e in reversed(events) if e["event"] == "response"), None)
    thinking = "".join(
        e["data"].get("text", "") for e in events
        if e["event"] == "response.thinking.delta" and isinstance(e["data"], dict)
    )
    errors = [e["data"] for e in events if e["event"] == "error"]
    kept = [e for e in events if e["event"] not in DELTA_EVENTS]
    return final, thinking, errors, kept


def trace_one(conn, host, row, attempts=2):
    for _ in range(attempts):
        result = run_agent(conn, host, row["REQUEST_TEXT"])
        final, thinking, errors, kept = summarize(result["events"])
        if final is not None and not errors:
            break
    return {
        "request_id": row["REQUEST_ID"],
        "intent": row["INTENT"],
        "request_text": row["REQUEST_TEXT"],
        "agent": AGENT_FQN,
        "status_code": result["status_code"],
        "http_error": result.get("error"),
        "elapsed_s": round(result["elapsed_s"], 2),
        "errors": errors,
        "thinking": thinking,
        "final_response": final,
        "events": kept,
    }


def load_requests(cur, pilot, done_ids):
    if pilot:
        cur.execute(
            "SELECT REQUEST_ID, INTENT, REQUEST_TEXT FROM CT_DEMO_DB.CORTEX_TRAINING.TRACE_REQUESTS "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY INTENT ORDER BY REQUEST_ID) <= %s",
            (pilot,),
        )
    else:
        cur.execute("SELECT REQUEST_ID, INTENT, REQUEST_TEXT FROM CT_DEMO_DB.CORTEX_TRAINING.TRACE_REQUESTS ORDER BY REQUEST_ID")
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [r for r in rows if r["REQUEST_ID"] not in done_ids]


def already_done(path):
    """Return ids with a successful trace, and rewrite the file without failed records so they are retried."""
    if not path.exists():
        return set()
    good = [json.loads(l) for l in path.open() if l.strip()]
    good = [r for r in good if r.get("final_response") is not None and not r.get("errors")]
    with path.open("w") as f:
        for r in good:
            f.write(json.dumps(r, default=str) + "\n")
    return {r["request_id"] for r in good}


def upload(conn, path):
    cur = conn.cursor()
    cur.execute("USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS AGENT_TRACES ("
        " REQUEST_ID STRING, INTENT STRING, REQUEST_TEXT STRING, AGENT STRING, STATUS_CODE NUMBER,"
        " ELAPSED_S FLOAT, HAS_ERROR BOOLEAN, TRACE VARIANT, LOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())"
    )
    cur.execute("CREATE TEMPORARY STAGE IF NOT EXISTS TRACE_UPLOAD")
    cur.execute("CREATE TEMPORARY FILE FORMAT IF NOT EXISTS TRACE_JSON TYPE = JSON")
    cur.execute(f"PUT 'file://{path}' @TRACE_UPLOAD OVERWRITE = TRUE AUTO_COMPRESS = TRUE")
    cur.execute(
        "DELETE FROM AGENT_TRACES WHERE REQUEST_ID IN ("
        " SELECT $1:request_id::STRING FROM @TRACE_UPLOAD (FILE_FORMAT => 'TRACE_JSON'))"
    )
    cur.execute(
        "INSERT INTO AGENT_TRACES (REQUEST_ID, INTENT, REQUEST_TEXT, AGENT, STATUS_CODE, ELAPSED_S, HAS_ERROR, TRACE) "
        "SELECT $1:request_id, $1:intent, $1:request_text, $1:agent, $1:status_code, $1:elapsed_s, "
        "  ARRAY_SIZE($1:errors) > 0 OR $1:status_code <> 200 OR $1:final_response IS NULL, $1 "
        "FROM @TRACE_UPLOAD (FILE_FORMAT => 'TRACE_JSON')"
    )
    cur.execute("SELECT COUNT(*), COUNT_IF(HAS_ERROR) FROM AGENT_TRACES")
    print("AGENT_TRACES rows, errors:", cur.fetchone())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not args.pilot and not args.all:
        ap.error("pass --pilot N or --all")

    out_path = OUT_DIR / ("pilot.jsonl" if args.pilot else "full.jsonl")
    conn, host = connect()
    todo = load_requests(conn.cursor(), args.pilot, already_done(out_path))
    print(f"{len(todo)} requests to trace -> {out_path}")

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a") as out:
        futures = {pool.submit(trace_one, conn, host, row): row for row in todo}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # network errors etc.; retried on next invocation
                failed += 1
                print(f"FAIL {row['REQUEST_ID']}: {exc}")
                continue
            with _lock:
                out.write(json.dumps(rec, default=str) + "\n")
                out.flush()
            if rec["errors"] or rec["final_response"] is None:
                failed += 1
            else:
                ok += 1
            if (ok + failed) % 25 == 0:
                print(f"progress: {ok} ok, {failed} failed")
    print(f"done: {ok} ok, {failed} failed")
    upload(conn, out_path)


if __name__ == "__main__":
    main()
