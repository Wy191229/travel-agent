import json
import threading
import time

STATS_LOCK = threading.Lock()

STATS = {
    "request_count": 0,
    "success_count": 0,
    "error_count": 0,
    "llm_calls": 0,
    "llm_latency_ms_total": 0,
    "tool_calls": {},
    "tool_errors": {},
    "tool_latency_ms_total": {},
    "last_error": None,
}


def log_event(event: str, **fields):
    payload = {
        "event": event,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def record_chat_started():
    with STATS_LOCK:
        STATS["request_count"] += 1


def record_chat_success():
    with STATS_LOCK:
        STATS["success_count"] += 1


def record_chat_error(trace_id: str, session_id: str, error_type: str, error: str):
    with STATS_LOCK:
        STATS["error_count"] += 1
        STATS["last_error"] = {
            "trace_id": trace_id,
            "session_id": session_id,
            "error_type": error_type,
            "error": error,
        }


def record_llm_latency(latency_ms: int):
    with STATS_LOCK:
        STATS["llm_calls"] += 1
        STATS["llm_latency_ms_total"] += latency_ms


def record_tool_result(tool_name: str, latency_ms: int, success: bool):
    with STATS_LOCK:
        STATS["tool_calls"][tool_name] = STATS["tool_calls"].get(tool_name, 0) + 1
        STATS["tool_latency_ms_total"][tool_name] = (
            STATS["tool_latency_ms_total"].get(tool_name, 0) + latency_ms
        )
        if not success:
            STATS["tool_errors"][tool_name] = STATS["tool_errors"].get(tool_name, 0) + 1


def build_stats_snapshot():
    with STATS_LOCK:
        request_count = STATS["request_count"]
        success_count = STATS["success_count"]
        llm_calls = STATS["llm_calls"]

        return {
            "request_count": request_count,
            "success_count": success_count,
            "error_count": STATS["error_count"],
            "success_rate": round(success_count / request_count, 4) if request_count else 0,
            "llm_calls": llm_calls,
            "avg_llm_latency_ms": round(STATS["llm_latency_ms_total"] / llm_calls, 2) if llm_calls else 0,
            "tool_calls": dict(STATS["tool_calls"]),
            "tool_errors": dict(STATS["tool_errors"]),
            "tool_latency_ms_total": dict(STATS["tool_latency_ms_total"]),
            "last_error": STATS["last_error"],
        }
