"""Bounded stdio transport; invoked by trusted host configuration, never a shell."""
import json
import os
import signal
import sys
from dataclasses import asdict

from . import IsolatedHarnessBackend, RunCancelled, Settings, TaskDefinition, TaskRunner, json_result


def main():
    session_id = None
    usage = None

    def send(kind, **payload):
        print(json.dumps({"type": kind, **payload}, allow_nan=False), flush=True)

    def started(value):
        nonlocal session_id
        session_id = value
        send("started", session_id=value)

    def measured(value):
        nonlocal usage
        usage = value
        send("usage", usage=value)

    def cancel(_signal, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise RunCancelled()

    signal.signal(signal.SIGTERM, cancel)
    try:
        line = sys.stdin.readline(256 * 1024 + 2)
        if len(line.encode()) > 256 * 1024 or not line.endswith("\n"):
            raise ValueError("Bridge input exceeds its budget")
        value = json.loads(line)
        settings = Settings(**value["settings"], api_key=os.environ.get(value["settings"]["api_key_env"]))
        task = value["task"]
        validator = json_result(task["resultSchema"])
        policy = task["systemPrompt"] + "\nOutput only JSON matching this schema:\n" + json.dumps(task["resultSchema"])
        definition = TaskDefinition(policy, json.dumps(value["input"], ensure_ascii=False, allow_nan=False), validator)
        run = TaskRunner(IsolatedHarnessBackend(settings)).run(definition,
            on_started=started, on_usage=measured, on_event=lambda event: send("event", event=event))
        send("result", result=asdict(run))
    except RunCancelled:
        send("result", result=dict(session_id=session_id, outcome="cancelled", result=None,
                                   error="Host cancelled the isolated task", usage=usage))
    except Exception:
        send("error", error="Isolated task failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
