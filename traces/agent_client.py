"""Run a Cortex Agent via the agent:run REST API and capture the full SSE event stream as a trace."""
import os
import json
import sys
import threading
import time

import requests
import snowflake.connector

CONNECTION = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "default")
AGENT = ("CT_DEMO_DB", "CORTEX_TRAINING", "SUPPORT_AGENT_TEACHER")
KEEPALIVE_S = 600
# Every query this demo runs is tagged so it can be found in QUERY_HISTORY and
# CORTEX_AI_FUNCTIONS_USAGE_HISTORY. Agent spend is tracked by agent name instead.
QUERY_TAG = "cortex_training_demo:" + os.path.splitext(os.path.basename(sys.argv[0]))[0]


class Session:
    """Thread-safe Snowflake session that keeps its REST token fresh for long runs.

    A cheap query every KEEPALIVE_S lets the connector renew the session token; on an
    auth failure the connection is rebuilt.
    """

    def __init__(self, connection_name=CONNECTION):
        """Connect using a named connection from the Snowflake connections config."""
        self._name = connection_name
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        """Open a new connection and record the REST host and refresh time."""
        self.conn = snowflake.connector.connect(connection_name=self._name, client_session_keep_alive=True,
                                                session_parameters={"QUERY_TAG": QUERY_TAG})
        # Underscores in the account locator are not valid in TLS hostnames.
        self.host = self.conn.host.replace("_", "-")
        self._last_refresh = time.time()

    def token(self, force=False):
        """Return a valid REST session token.

        force=True reconnects unconditionally (used after a 401/403); otherwise the session is
        pinged at most every KEEPALIVE_S seconds and rebuilt if the ping fails.
        """
        with self._lock:
            if force:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self._connect()
            elif time.time() - self._last_refresh > KEEPALIVE_S:
                try:
                    self.conn.cursor().execute("SELECT 1")
                except snowflake.connector.errors.ProgrammingError:
                    self._connect()
                self._last_refresh = time.time()
            return self.conn.rest.token

    def cursor(self):
        """Return a cursor on a connection whose token has just been checked."""
        self.token()
        return self.conn.cursor()


def connect():
    """Create a Session and return it with the REST host to call."""
    session = Session()
    return session, session.host


def _post(session, url, body, timeout, force_refresh=False):
    """POST to the agent API with session-token auth and return the streaming response."""
    headers = {
        "Authorization": f'Snowflake Token="{session.token(force=force_refresh)}"',
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    return requests.post(url, headers=headers, json=body, stream=True, timeout=timeout)


def run_agent(session, host, question, agent=AGENT, timeout=300):
    """Ask an agent a single question and collect every SSE event it streams back.

    Returns {"status_code", "events", "elapsed_s"} plus "error" on a non-200 response. Each event
    is {"event": <SSE event type>, "data": <parsed JSON or raw string>}.
    """
    db, schema, name = agent
    url = f"https://{host}/api/v2/databases/{db}/schemas/{schema}/agents/{name}:run"
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}]}
    started = time.time()
    events = []
    resp = _post(session, url, body, timeout)
    # An expired token shows up as 401/403; reconnect once and retry.
    if resp.status_code in (401, 403):
        resp.close()
        resp = _post(session, url, body, timeout, force_refresh=True)
    with resp:
        if resp.status_code != 200:
            return {"status_code": resp.status_code, "error": resp.text, "events": [], "elapsed_s": time.time() - started}
        # SSE framing: an "event:" line names the type of the "data:" line that follows.
        event_type = None
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw.startswith("event:"):
                event_type = raw[len("event:"):].strip()
            elif raw.startswith("data:"):
                data = raw[len("data:"):].strip()
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = data
                events.append({"event": event_type, "data": payload})
    return {"status_code": 200, "events": events, "elapsed_s": time.time() - started}


if __name__ == "__main__":
    session, host = connect()
    result = run_agent(session, host, sys.argv[1])
    print(json.dumps({"status_code": result["status_code"], "error": result.get("error"),
                      "elapsed_s": round(result["elapsed_s"], 1),
                      "event_types": [e["event"] for e in result["events"]]}, indent=2))
    if result["events"]:
        print(json.dumps(result["events"][-3:], indent=2)[:4000])
