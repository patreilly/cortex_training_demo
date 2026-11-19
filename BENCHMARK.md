# Support-agent benchmark

Compare any Cortex Agent orchestration model (frontier, base Qwen, fine-tuned) on the same cases, tools, and scoring.

## What's recorded
| Object (`CT_DEMO_DB.CORTEX_TRAINING`) | Contents |
| --- | --- |
| `BENCHMARK_CASES` | 2,045 cases: 1,900 standard requests (9 intents) + 145 hand-labelled adversarial (6 attack types). Expectations come from rules and source data, not from any model's output. |
| `ADVERSARIAL_CASES` | Adversarial cases with the expected behavior written out. |
| `AGENT_TRACES` | Raw teacher traces used for fine-tuning (full SSE events, tool calls/results, thinking, token usage). |
| `AGENT_BENCHMARK_RUNS` | One row per (run, case): full trace, latency, orchestration tokens incl. cache read/write. |
| `AGENT_BENCHMARK_SCORES` | One row per (run, case): deterministic checks + LLM-judge verdict, `PASS`. |
| `CORTEX_TRAINING_BENCH_V1` | Zero-copy snapshot of all data as of 2026-11-03. |

Raw JSONL copies are kept in `traces/raw/` and `traces/benchmark_runs/`.

## Benchmark a new model
1. Copy `agent/SUPPORT_AGENT_TEACHER.agent.yaml`, change only `models.orchestration`, deploy as a new agent:
   `cortex agent-studio agent-deploy --file-path <new>.agent.yaml --fqn CT_DEMO_DB.CORTEX_TRAINING.<NAME> -c $SNOWFLAKE_CONNECTION_NAME`
2. Run: `python traces/benchmark_run.py --agent CT_DEMO_DB.CORTEX_TRAINING.<NAME> --run-name <run>`
   (`--per-intent 20` for a ~300-case quick comparison; `--case-set adversarial` for adversarial only)
3. Score: `python traces/score_run.py --run-name <run>` (judge defaults to `mistral-large2`, a different family than the teacher)
4. Compare runs:
```sql
SELECT s.RUN_NAME, s.CASE_SET, COUNT(*) n, ROUND(AVG(IFF(s.PASS,1,0))*100,1) pass_pct,
       COUNT_IF(s.POLICY_VIOLATION) policy_violations, ROUND(AVG(r.ELAPSED_S),1) avg_s,
       ROUND(AVG(r.ORCH_INPUT_TOKENS)) avg_in_tok, ROUND(AVG(r.ORCH_OUTPUT_TOKENS)) avg_out_tok
FROM CT_DEMO_DB.CORTEX_TRAINING.AGENT_BENCHMARK_SCORES s
JOIN CT_DEMO_DB.CORTEX_TRAINING.AGENT_BENCHMARK_RUNS r USING (RUN_NAME, CASE_ID)
GROUP BY 1, 2 ORDER BY 2, 1;
```

## Caveats
- Tool UDFs read the live `CORTEX_TRAINING` tables. Don't modify TICKETS/CUSTOMERS/KB_ARTICLES between runs; use `CORTEX_TRAINING_BENCH_V1` as the reference copy. `CREATE_ESCALATION` writes to `ESCALATIONS`, which no read tool depends on.
- The Cortex Agents platform system prompt (~30k tokens) is not returned in traces, so offline replay against models outside Cortex Agents is approximate.
- LLM-judge verdicts are a second opinion; spot-check failures before trusting a pass-rate difference under ~3 points.
- Run `sql/08_import_teacher_run.sql` after the full trace run so the teacher's 1,900 standard cases count as run `teacher_sonnet45_v1` without re-running.
