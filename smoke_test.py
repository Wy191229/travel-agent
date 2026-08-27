import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1")
API_KEY = os.getenv("AGENT_API_KEY") or os.getenv("APP_API_KEY") or os.getenv("API_KEY")

if not API_KEY:
    print("FAIL: .env 中没有 AGENT_API_KEY")
    sys.exit(1)


def check(name, message, keywords):
    print(f"\n=== {name} ===")
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY,
            },
            json={
                "session_id": f"smoke-{int(time.time())}-{name}",
                "message": message,
            },
            timeout=90,
        )
        print("HTTP", response.status_code)

        if response.status_code != 200:
            print(response.text)
            return False

        data = response.json()
        answer = data.get("answer", "")
        print(answer[:500])

        missing = [word for word in keywords if word not in answer]
        if missing:
            print("FAIL: 缺少关键词", missing)
            return False

        print("PASS")
        return True

    except Exception as exc:
        print("FAIL:", exc)
        return False


def main():
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    if health.status_code != 200:
        print("FAIL: /health 不正常", health.text)
        sys.exit(1)

    cases = [
        (
            "beijing_weather_attraction",
            "请查询北京今天的天气，然后推荐一个合适的旅游景点。",
            ["北京", "天气", "推荐"],
        ),
        (
            "beijing_route",
            "请帮我规划从北京站到故宫博物院的公共交通路线。",
            ["路线", "北京站", "故宫"],
        ),
        (
            "shanghai_route",
            "请帮我规划从上海站到东方明珠的公共交通路线。",
            ["路线", "上海", "东方明珠"],
        ),
        (
            "chengdu_route",
            "请帮我规划从成都东站到宽窄巷子的公共交通路线。",
            ["路线", "成都东站", "宽窄巷子"],
        ),
        (
            "memory",
            "我现在在成都东站，想去宽窄巷子。下一句我只说“怎么坐车”，你要理解我的上下文。",
            ["成都", "宽窄巷子"],
        ),
    ]

    passed = 0
    for name, message, keywords in cases:
        if check(name, message, keywords):
            passed += 1

    print(f"\nRESULT: {passed}/{len(cases)} passed")

    if passed != len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
