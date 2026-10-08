import json
import time

from codex_agent.model_router import ModelRouter, TelemetryRecorder, estimate_tokens


def test_routes_by_intent_and_escalates():
    router = ModelRouter()
    assert router.route("validate schema rules").model == "rules"
    assert router.route("summarize this result").model == "fast"
    assert router.route("implement a Python operator").model == "coder"
    assert router.route("debug MLIR lowering compiler kernel").model == "deep"
    assert router.route("implement a Python operator", failure_stage=1).model == "deep"
    assert router.route("validate schema rules", failure_stage=1).model == "fast"


def test_telemetry_span_writes_jsonl(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    recorder = TelemetryRecorder(path, pricing={"coder": (0.01, 0.02)})
    with recorder.span("coder", task="implement", input_tokens=100) as span:
        time.sleep(0.001)
        span.finish(output_tokens=50)
    events = recorder.read()
    assert len(events) == 1
    event = events[0]
    assert event.total_tokens == 150
    assert event.cost_usd == 0.002
    assert event.latency_ms > 0
    assert json.loads(path.read_text())["model"] == "coder"


def test_estimate_tokens_is_stable():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
