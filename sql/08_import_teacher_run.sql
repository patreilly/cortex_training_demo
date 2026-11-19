ALTER SESSION SET QUERY_TAG = 'cortex_training_demo:08_import_teacher_run';
-- Register the teacher's trace-generation run as a benchmark run so the 1,900 standard cases
-- don't have to be re-run. Run after run_traces.py --all has uploaded to AGENT_TRACES.
USE SCHEMA CT_DEMO_DB.CORTEX_TRAINING;

DELETE FROM AGENT_BENCHMARK_RUNS WHERE RUN_NAME = 'teacher_sonnet45_v1';

INSERT INTO AGENT_BENCHMARK_RUNS (RUN_NAME, AGENT, ORCHESTRATION_MODEL, CASE_ID, CASE_SET, STATUS_CODE, ELAPSED_S, HAS_ERROR,
    ORCH_INPUT_TOKENS, ORCH_CACHE_READ_TOKENS, ORCH_CACHE_WRITE_TOKENS, ORCH_OUTPUT_TOKENS, TRACE, RUN_AT)
SELECT 'teacher_sonnet45_v1', t.AGENT, u.value:model_name::STRING, t.REQUEST_ID, 'trace_requests', t.STATUS_CODE, t.ELAPSED_S, t.HAS_ERROR,
    u.value:input_tokens:total, u.value:input_tokens:cache_read, u.value:input_tokens:cache_write, u.value:output_tokens:total,
    t.TRACE, t.LOADED_AT
FROM AGENT_TRACES t,
    LATERAL FLATTEN(input => t.TRACE:final_response:metadata:usage:tokens_consumed, outer => TRUE) u
WHERE COALESCE(u.value:model_role::STRING, 'orchestration') = 'orchestration'
  AND t.REQUEST_ID IN (SELECT CASE_ID FROM BENCHMARK_CASES WHERE CASE_SET = 'trace_requests');
