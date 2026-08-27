import json
import os
import time
from pathlib import Path

import requests

API_BASE = os.getenv("AGENT_API_BASE", "http://127.0.0.1:8000")

AUTO_CLEAR_SESSIONS = os.getenv("EVAL_AUTO_CLEAR_SESSIONS", "1") != "0"


def clear_eval_session(session_id):
    if not session_id:
        return

    api_key = globals().get("API_KEY") or os.getenv("AGENT_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        response = requests.post(
            f"{API_BASE}/clear_session",
            headers=headers,
            json={"session_id": session_id},
            timeout=10,
        )
        if response.status_code >= 400:
            print(
                f"  WARN clear_session failed session_id={session_id} "
                f"status={response.status_code} body={response.text[:120]}"
            )
    except Exception as exc:
        print(f"  WARN clear_session failed session_id={session_id}: {exc}")


API_KEY = (
    os.getenv("AGENT_API_KEY")
    or os.getenv("APP_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "120"))

BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "cases.jsonl"
REPORTS_DIR = BASE_DIR / "reports"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}


def load_cases():
    cases = []
    with CASES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def clear_sessions(cases):
    seen = set()
    for case in cases:
        session_id = case["session_id"]
        if session_id in seen:
            continue
        seen.add(session_id)
        try:
            requests.post(
                f"{API_BASE}/clear_session",
                headers=HEADERS,
                json={"session_id": session_id},
                timeout=15,
            )
        except Exception as exc:
            print(f"  WARN clear_session failed session_id={session_id}: {exc}")


def call_agent(case):
    started = time.time()
    response = requests.post(
        f"{API_BASE}/chat",
        headers=HEADERS,
        json={
            "session_id": case["session_id"],
            "message": case["message"],
        },
        timeout=TIMEOUT,
    )
    latency_ms = int((time.time() - started) * 1000)

    try:
        data = response.json()
    except Exception:
        data = {"answer": response.text, "trace_id": None}

    return response.status_code, data, latency_ms

def judge_case(case, status_code, data):
    answer = str(data.get("answer", ""))

    must_include = case.get("must_include", [])
    any_include = case.get("any_include", case.get("expect_keywords", []))
    must_not_include = case.get("must_not_include", [])

    missing_must = [keyword for keyword in must_include if keyword not in answer]
    any_hits = [keyword for keyword in any_include if keyword in answer]
    forbidden_hits = [keyword for keyword in must_not_include if keyword in answer]

    reasons = []
    if status_code != 200:
        reasons.append(f"status_code={status_code}")
    if missing_must:
        reasons.append("missing_must=" + ",".join(missing_must))
    if any_include and not any_hits:
        reasons.append("no_any_include_hit")
    if forbidden_hits:
        reasons.append("forbidden_hits=" + ",".join(forbidden_hits))

    return {
        "passed": not reasons,
        "reasons": reasons,
        "keyword_hits": any_hits,
        "answer": answer,
        "trace_id": data.get("trace_id"),
    }

def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    index = int(round((len(values) - 1) * p / 100))
    return values[index]


def summarize_by_category(results):
    summary = {}
    for item in results:
        category = item["category"]
        bucket = summary.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if item["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    for bucket in summary.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0

    return summary


def write_badcases(results, report_path):
    badcases = [item for item in results if not item["passed"]]
    badcase_path = report_path.with_name(report_path.stem.replace("eval_report", "badcases") + ".md")

    lines = ["# Agent Eval Badcases", ""]
    if not badcases:
        lines.append("本次评测无失败用例。")
    else:
        for item in badcases:
            lines.extend([
                f"## {item['id']}",
                "",
                f"- category: {item['category']}",
                f"- latency_ms: {item['latency_ms']}",
                f"- trace_id: {item['trace_id']}",
                f"- reasons: {', '.join(item['reasons'])}",
                "",
                "### 用户输入",
                "",
                item["message"],
                "",
                "### 实际回答",
                "",
                item["answer"],
                "",
            ])

    badcase_path.write_text("\n".join(lines), encoding="utf-8")
    return badcase_path


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    clear_sessions(cases)

    results = []
    cleared_sessions = set()

    for index, case in enumerate(cases, 1):
        session_id = case.get("session_id", "")
        if AUTO_CLEAR_SESSIONS and session_id and session_id not in cleared_sessions:
            clear_eval_session(session_id)
            cleared_sessions.add(session_id)
        print(f"[{index}/{len(cases)}] {case['id']} ...")
        status_code, data, latency_ms = call_agent(case)
        judged = judge_case(case, status_code, data)

        result = {
            "id": case["id"],
            "category": case["category"],
            "message": case["message"],
            "status_code": status_code,
            "latency_ms": latency_ms,
            "passed": judged["passed"],
            "reasons": judged.get("reasons", []),
            "keyword_hits": judged["keyword_hits"],
            "trace_id": judged["trace_id"],
            "answer": judged["answer"],
        }
        results.append(result)

        state = "PASS" if result["passed"] else "FAIL"
        print(f"  {state} latency={latency_ms}ms trace_id={result['trace_id']}")
        if result["reasons"]:
            print("  reasons=" + "; ".join(result["reasons"]))

    latencies = [item["latency_ms"] for item in results]
    passed_count = sum(1 for item in results if item["passed"])
    total_count = len(results)

    summary = {
        "total": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": round(passed_count / total_count, 4) if total_count else 0,
        "avg_latency_ms": round(sum(latencies) / total_count, 2) if total_count else 0,
        "p95_latency_ms": percentile(latencies, 95),
        "by_category": summarize_by_category(results),
        "results": results,
    }

    report_path = REPORTS_DIR / f"eval_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    badcase_path = write_badcases(results, report_path)

    print("")
    print(json.dumps({
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "pass_rate": summary["pass_rate"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "p95_latency_ms": summary["p95_latency_ms"],
        "report_path": str(report_path),
        "badcase_path": str(badcase_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
