# Cortex Training demo: distilling a Cortex Agent orchestrator

Fine-tune a small open model (Qwen3.5-4B) on traces from a Cortex Agent running Claude Sonnet 4.5, then measure how close the student gets to the teacher, and what it would cost to run instead.

Everything here is driven from SQL and Python, so you can run it end to end from Cortex Code (CoCo): open this folder and ask CoCo to work through the steps below. All data is synthetic (a fictional SaaS company, "Northwind Cloud").

## Prerequisites

1. **Cortex Training enabled on the account.** It is in preview, available in commercial AWS regions, and requires enablement. Ask your Snowflake account team.
2. **Claude Sonnet 4.5 available to Cortex Agents** in your region, or cross-region inference enabled.
3. **A role** that can create a database, tables, UDFs, a stored procedure, a Cortex Search service, a semantic view and a Cortex Agent, and that can use Cortex AI functions (the scorer uses `mistral-large2` as a judge).
4. **A Snowflake connection** in `~/.snowflake/connections.toml`. Every script reads its name from `SNOWFLAKE_CONNECTION_NAME` (default `default`):
   ```bash
   export SNOWFLAKE_CONNECTION_NAME=<your_connection>
   ```
5. **Python 3.11+**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   `keyring` lets OAuth and SSO tokens cache, so long background jobs don't stop on a browser login.

Everything is created in `CT_DEMO_DB.CORTEX_TRAINING`. To use another name, find-and-replace `CT_DEMO_DB` across the repo.

## Steps

### 1. Build the domain (minutes)
Run in order: `sql/00_setup.sql` through `sql/07_benchmark.sql`. This creates the products, 500 customers, about 3,000 tickets (20 with prompt injections), about 195 KB articles, the `KB_SEARCH` search service, the `SUPPORT_ANALYTICS` semantic view, `GET_TICKET` / `LOOKUP_CUSTOMER` / `CREATE_ESCALATION`, about 2,000 trace requests across 10 intents, and a ~2,045-case benchmark (145 adversarial). Ticket, article and request text is written by `AI_COMPLETE`, so exact counts vary slightly between runs.

### 2. Deploy the teacher agent
```bash
cortex agent-studio agent-deploy --file-path "$PWD/agent/SUPPORT_AGENT_TEACHER.agent.yaml" \
  --fqn CT_DEMO_DB.CORTEX_TRAINING.SUPPORT_AGENT_TEACHER -c $SNOWFLAKE_CONNECTION_NAME
```
Use an absolute path for `--file-path`; relative paths resolve against a `cortex_project/` folder and fail.

### 3. Capture teacher traces (a few hours)
```bash
python traces/run_traces.py --pilot 10     # optional: 10 per intent (100 traces) to check traces look right
python traces/run_traces.py --all          # all 2,000 requests -> traces/raw/full.jsonl and AGENT_TRACES
```

### 4. Score the teacher
Run `sql/08_import_teacher_run.sql`, which registers the trace run as benchmark run `teacher_sonnet45_v1`. Then:
```bash
python traces/score_run.py --run-name teacher_sonnet45_v1
python traces/benchmark_run.py --agent CT_DEMO_DB.CORTEX_TRAINING.SUPPORT_AGENT_TEACHER \
  --run-name teacher_adv --case-set adversarial      # optional: adversarial cases
```
`BENCHMARK.md` covers scoring and comparing runs in more detail.

### 5. Build training data
```bash
python sft/export_inputs.py                          # writes sft/teacher_scores.json, sft/real_customers.json
python sft/convert.py --drop-thinking --out-dir sft/data_v4
```
This keeps only traces that passed, holds out 15% of cases, pseudonymizes customer and ticket IDs, leak-scans the output, and drops reasoning from every turn.

**Trying it small first?** If you only ran `--pilot`, add `--traces traces/raw/pilot.jsonl` to `convert.py` here and to `eval_student.py` in step 7. Use `train.py --smoke` and `eval_student.py --limit 20` for a ~2-minute training job and a short eval that check the plumbing end to end.

### 6. Train the student (~70 min on 1×H200)
```bash
python sft/train.py --data sft/data_v4/train.jsonl --run-name student_qwen35_4b_v4 --epochs 2 --save-every 50
```
The script prints the job ID and the checkpoint names when it finishes. Run it with `--smoke` first for a two-step check. `sft/probe_labels.py` checks the label convention (see "What we learned").

### 7. Evaluate on the held-out turns (~8 min)
```bash
python sft/eval_student.py --label base_qwen35_4b --source base --data-dir sft/data_v4
python sft/eval_student.py --label student_qwen35_4b_v4 --source checkpoint \
  --job-id <job_id> --checkpoint student_qwen35_4b_v4-step176 --data-dir sft/data_v4
python sft/breakdown.py base_qwen35_4b student_qwen35_4b_v4
```
Each eval deploys a model, replays every held-out assistant turn, and drops the deployment when it finishes. If a run is interrupted, drop the deployment yourself, since it bills until dropped.

## Results from our run

Teacher (Sonnet 4.5): 87.7% pass on 1,900 standard cases, 0 policy violations; 50.3% on adversarial cases.

Student next-turn replay on 790 held-out turns:

| Metric | Base Qwen3.5-4B | Student v4 |
| --- | --- | --- |
| Right kind of turn (tool call vs. answer) | 78.2 | 81.4 |
| Same tools as teacher | 72.4 | 78.4 |
| Correct ticket/customer IDs | 93.7 | 97.6 |
| Right tool, on turns that need one | 75.7 | 93.3 |
| Answers, on turns that need an answer | 64.9 | 43.9 |

v4's remaining gap: after tool results come back, it often searches again instead of summarizing. Your numbers will vary somewhat with your own traces.

## Cost estimates

Rates: Cortex Training On Demand bills **4.80 AI credits per H200 GPU-hour**, metered per second, from GPU assignment until the job ends or the deployment is dropped (Service Consumption Table, Table 6(i)). Sonnet 4.5 agent usage and `AI_COMPLETE` bill per token.

### Running the demo

| Step | Time | AI credits (our run) | Billed as |
| --- | --- | --- | --- |
| Build the domain (SQL 00–07) | minutes | ~0.5 | `AI_COMPLETE` llama3.3-70b writes tickets, KB articles and requests |
| 2,000 teacher traces + benchmark runs | a few hours | ~98 (most of the cost) | Cortex Agent, Sonnet 4.5 |
| Score with the LLM judge (1,900 cases) | — | ~2 | `AI_COMPLETE` mistral-large2 |
| Train student v4 | ~70 min, 1×H200 | ~5–6 (estimate) | Cortex Training GPU time |
| Evaluate student v4 | ~8 min, 1×H200 | ~0.6 (estimate) | Cortex Training GPU time |
| **Total** | | **~107** | |

The first three rows are billed figures from `CORTEX_AGENT_USAGE_HISTORY` and `CORTEX_AI_FUNCTIONS_USAGE_HISTORY`; the GPU rows are GPU time × rate. Warehouse compute for the SQL is extra and small.

A pilot-size check (`--pilot 20`: 200 traces, scoring, `--smoke` training, a 20-turn eval) cost about 9 credits: ~8.0 Sonnet, ~0.6 `AI_COMPLETE`, the rest GPU time.

To spend less, capture fewer traces, and use `--per-intent 20` on `benchmark_run.py` for a ~300-case run.

### Tracking spend

Every SQL file and every Python connection sets `QUERY_TAG = 'cortex_training_demo:<step>'`, so this demo's queries can be found in `QUERY_HISTORY` and its `AI_COMPLETE` spend in `CORTEX_AI_FUNCTIONS_USAGE_HISTORY`:
```sql
SELECT QUERY_TAG, MODEL_NAME, ROUND(SUM(CREDITS), 4) AS credits
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AI_FUNCTIONS_USAGE_HISTORY
WHERE QUERY_TAG LIKE 'cortex_training_demo:%' AND START_TIME >= DATEADD(day, -30, CURRENT_TIMESTAMP())
GROUP BY 1, 2 ORDER BY 3 DESC;
```
`CORTEX_AGENT_USAGE_HISTORY` has no query tag column; track the teacher's spend by agent name (`AGENT_NAME = 'SUPPORT_AGENT_TEACHER'`). The `cortex-training` jobs and deployments run outside SQL and aren't tagged.

### Teacher vs. student per request

| Orchestrator | Credits / request | Per 1M requests | vs. Sonnet |
| --- | --- | --- | --- |
| Sonnet 4.5 teacher (billed) | 0.0453 | ~45,300 | — |
| Student v4, eval window incl. model load | 0.00248 | ~2,480 | −94.5% |
| Student v4, serving time only | 0.00064 | ~640 | −98.6% |

- **Teacher:** 98.26 AI credits over 2,169 requests, from `SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AGENT_USAGE_HISTORY`. All of it was Sonnet orchestration, about 83k tokens per request.
- **Student:** the v4 eval held one H200 for 449 s, including model load, to serve 241 agent requests (790 model calls). Measured throughput was about 7,500 requests/hour at 8 concurrent requests.
- **Break-even:** an always-on H200 (115 credits/day) is cheaper than Sonnet above about **106 requests/hour**.

**Caveats:**
- **Eval prompts were smaller than real agent traffic.** They averaged ~4k tokens and lacked the ~30k-token Cortex Agents platform context, so real per-request GPU time would be higher.
- **The GPU wasn't saturated** at 8 concurrent requests.
- **Idle time is billed.** A deployment bills until it is dropped, whether or not it is serving.
- **Only offline replay was measured.** End-to-end savings depend on fine-tuned models being selectable as the agent orchestration model, and on the student passing the full benchmark.

## What we learned (already handled in the code)

- **Label shifting:** the training server shifts labels itself, so send unshifted, Hugging Face-style labels. Pre-shifted labels give loss ~15 instead of ~0.3; `sft/probe_labels.py` checks this.
- **No `chat_template_kwargs`:** the inference endpoint rejects it, so reasoning can't be turned off per request. That's why v4 drops reasoning from the training data instead.
- **Empty reasoning blocks:** Qwen3.5's generation prompt opens a `<think>` block. Training on bare tool calls taught the model to close it and call a tool every turn, which is why earlier versions over-called tools.
- **Train/serve parity:** prompts at serving time don't carry earlier turns' reasoning, so training prefixes mustn't either.
- **`experiment_name`:** omit it on job creation unless the experiment already exists.
- **Worth trying next:** `Qwen/Qwen3-4B-Instruct-2507` has no reasoning mode at all.
