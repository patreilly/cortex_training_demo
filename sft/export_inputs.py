"""Export the two inputs convert.py needs, after the teacher run has been scored.

  teacher_scores.json   - PASS/INTENT per trace case (only PASS traces become training data)
  real_customers.json   - customer ids and names, used to pseudonymize and leak-scan the data

  python sft/export_inputs.py [--run-name teacher_sonnet45_v1]
"""

import argparse
import json
import os
from pathlib import Path

import snowflake.connector

HERE = Path(__file__).resolve().parent
SCHEMA = "CT_DEMO_DB.CORTEX_TRAINING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION_NAME", "default"))
    ap.add_argument("--run-name", default="teacher_sonnet45_v1")
    args = ap.parse_args()

    cur = snowflake.connector.connect(connection_name=args.connection,
                                      session_parameters={"QUERY_TAG": "cortex_training_demo:export_inputs"}
                                      ).cursor(snowflake.connector.DictCursor)
    cur.execute(f"""SELECT s.CASE_ID, c.INTENT, s.PASS
                    FROM {SCHEMA}.AGENT_BENCHMARK_SCORES s
                    JOIN {SCHEMA}.BENCHMARK_CASES c USING (CASE_ID)
                    WHERE s.RUN_NAME = %s ORDER BY s.CASE_ID""", (args.run_name,))
    scores = cur.fetchall()
    if not scores:
        raise SystemExit(f"no scores for run {args.run_name}: run sql/08 and traces/score_run.py first")
    json.dump(scores, open(HERE / "teacher_scores.json", "w"), indent=1)

    cur.execute(f"SELECT CUSTOMER_ID, COMPANY_NAME FROM {SCHEMA}.CUSTOMERS ORDER BY CUSTOMER_ID")
    customers = cur.fetchall()
    json.dump(customers, open(HERE / "real_customers.json", "w"), indent=1)
    print(f"wrote {len(scores)} scores, {len(customers)} customers")


if __name__ == "__main__":
    main()
