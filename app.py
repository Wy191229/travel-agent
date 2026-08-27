import uuid
from observability import (
    build_stats_snapshot,
    log_event,
    record_chat_error,
    record_chat_started,
    record_chat_success,
    record_llm_latency,
    record_tool_result,
)
import time
import os
import re
import logging
import requests
import memory as memory_store
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient
from rag import search_travel_guide, format_search_results
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("travel-agent")


def log_amap_error(service: str, data: dict):
    logger.warning(
        "AMAP_ERROR service=%s status=%s info=%s infocode=%s",
        service,
        data.get("status"),
        data.get("info"),
        data.get("infocode"),
    )


def log_tool_exception(service: str, exc: Exception):
    logger.exception("TOOL_EXCEPTION service=%s error=%s", service, exc)

load_dotenv()
memory_store.init_memory_db()
app = FastAPI(title="Travel Agent")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ClearSessionRequest(BaseModel):
    session_id: str
session_memory = {}
MAX_MEMORY_MESSAGES = 8


def get_service_api_key() -> str:
    return (
        os.getenv("AGENT_API_KEY")
        or os.getenv("APP_API_KEY")
        or os.getenv("API_KEY")
        or ""
    )

SYSTEM_PROMPT = """
你是一个智能旅行助手。

你可以使用以下工具：
1. get_weather(city="城市名")：查询城市天气
2. get_attraction(city="城市名", weather="天气情况")：根据天气推荐景点
3. plan_route(city="城市名", origin="起点", destination="终点", mode="walking")：规划从起点到终点的路线，mode 可选 walking、driving、transit。walking 表示步行，driving 表示驾车，transit 表示公共交通/公交/地铁。
你必须严格按照以下格式输出：

Thought: 你的思考
Action: 工具调用

可用 Action 示例：
Action: get_weather(city="北京")
Action: get_attraction(city="北京", weather="晴天")
Action: Finish[最终回答]
Action: plan_route(city="北京", origin="北京站", destination="故宫博物院", mode="walking")
Action: plan_route(city="北京", origin="北京站", destination="故宫博物院", mode="transit")
路线规划规则：
- 用户提到“公共交通、公交、地铁、坐车、怎么过去、不想步行”时，必须调用 plan_route，并使用 mode="transit"。
- 用户提到“步行、走路”时，使用 mode="walking"。
- 用户提到“开车、驾车、打车”时，使用 mode="driving"。
- 涉及天气、景点、路线等实时信息时，必须先调用工具，不能凭常识编造路线、天气或交通方案。
规则：
- 如果还需要信息，调用工具
- 如果已经可以回答，必须使用 Action: Finish[最终回答]
- 每轮只能输出一个 Action
- 不要自己编写 Observation，Observation 只能由系统工具返回
- 不要输出其他格式
- 最终回答必须完整保留天气工具 Observation 中的天气、气温、湿度、风向、风力、发布时间，不要省略。
"""

SYSTEM_PROMPT += """
路线规划边界规则：
- 调用 plan_route 前，必须确认用户输入或当前会话上下文中同时存在明确的起点和终点。
- 缺少起点时，直接询问用户“请提供出发地”，不要默认使用北京站、上海站、成都东站等地点。
- 缺少终点时，直接询问用户“请提供目的地”，并复述已知起点。
- 用户给出起点或终点但地点可能不存在时，必须保留用户原始地点名称，说明无法确认该地点，请用户确认；不要替换成其他地点。
- 不允许把用户没有明确提供的地点写入 plan_route 参数。
"""

SYSTEM_PROMPT += """
缺失参数追问规则：
- 如果用户已经提供起点但没有提供终点，最终回答必须包含这个已知起点，例如“已知你从成都东站出发，请提供目的地”。
- 如果用户已经提供终点但没有提供起点，最终回答必须包含这个已知终点，例如“请提供前往故宫的出发地”。
- 缺少路线规划参数时，不要调用 plan_route，直接用 Finish[...] 追问缺失信息。
"""

SYSTEM_PROMPT += """
旅行攻略 RAG 工具：
- search_travel_guide(city="城市名", query="检索问题")：从本地旅行攻略知识库中检索城市景点、游玩建议、预约提醒、适合天气等信息。
- 当用户询问景点游玩建议、攻略、适合什么天气、是否需要预约、上午/下午/雨天/晴天去哪时，应优先调用 search_travel_guide 获取知识库信息。
- 如果问题同时涉及实时天气和攻略建议，应先调用 get_weather，再调用 search_travel_guide，最后综合两类信息回答。
- search_travel_guide 返回的是知识库检索结果，最终回答需要自然整合，不要逐字堆砌相似度。
"""


def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content

def get_weather(city: str) -> str:
    city_adcodes = {
        "北京": "110000",
        "上海": "310000",
        "天津": "120000",
        "重庆": "500000",
        "杭州": "330100",
        "广州": "440100",
        "深圳": "440300",
        "成都": "510100",
        "南京": "320100",
        "武汉": "420100",
        "西安": "610100",
    }

    fallback_weather = {
        "北京": "北京: 晴，气温 26-34 摄氏度，空气质量良好，适合户外游览，但中午较热。",
        "上海": "上海: 多云，气温 27-33 摄氏度，湿度较高，适合室内外结合游览。",
        "杭州": "杭州: 阴，气温 25-31 摄氏度，适合西湖周边轻松游览。",
        "广州": "广州: 阵雨，气温 26-32 摄氏度，建议选择室内景点或带伞出行。",
        "深圳": "深圳: 多云，气温 27-32 摄氏度，适合海边和城市休闲游。",
    }

    amap_key = os.getenv("AMAP_API_KEY")
    adcode = city_adcodes.get(city)

    if not amap_key or not adcode:
        return fallback_weather.get(
            city,
            f"{city}: 暂未配置该城市的天气编码，请补充 adcode 后查询真实天气。"
        )

    try:
        response = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={
                "key": amap_key,
                "city": adcode,
                "extensions": "base",
                "output": "JSON",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "1" or not data.get("lives"):
            return fallback_weather.get(city, f"{city}: 天气查询失败，返回信息：{data}")

        live = data["lives"][0]
        return (
            f"{live.get('city', city)}: {live.get('weather')}，"
            f"气温 {live.get('temperature')} 摄氏度，"
            f"湿度 {live.get('humidity')}%，"
            f"{live.get('winddirection')}风 {live.get('windpower')} 级，"
            f"发布时间 {live.get('reporttime')}。"
        )
    except Exception as exc:
        return fallback_weather.get(
            city,
            f"{city}: 天气接口暂时不可用，原因是 {exc}。"
        )

def get_attraction(city: str, weather: str) -> str:
    city_adcodes = {
        "北京": "110000",
        "上海": "310000",
        "天津": "120000",
        "重庆": "500000",
        "杭州": "330100",
        "广州": "440100",
        "深圳": "440300",
        "成都": "510100",
        "南京": "320100",
        "武汉": "420100",
        "西安": "610100",
    }

    local_attractions = {
        "北京": (
            "推荐景点：颐和园。理由：晴天适合游览古典园林，昆明湖和长廊能提供较好的游览体验；"
            "如果气温较高，可以避开正午，选择上午或傍晚游览。"
        ),
        "上海": "推荐景点：上海博物馆或外滩。理由：室内外都方便安排。",
        "杭州": "推荐景点：西湖。理由：适合湖边步行、拍照和轻松游览。",
        "广州": "推荐景点：广东省博物馆。理由：天气不稳定时更适合室内景点。",
        "深圳": "推荐景点：深圳湾公园。理由：适合海边散步，也方便灵活调整行程。",
    }

    amap_key = os.getenv("AMAP_API_KEY")
    adcode = city_adcodes.get(city)

    if not amap_key or not adcode:
        return local_attractions.get(
            city,
            f"暂未配置 {city} 的景点搜索信息，建议选择当地知名、交通便利、可根据天气灵活调整的景点。"
        )

    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": amap_key,
                "keywords": "景点",
                "types": "110000",
                "city": adcode,
                "citylimit": "true",
                "offset": 5,
                "page": 1,
                "extensions": "all",
                "output": "JSON",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "1" or not data.get("pois"):
            return local_attractions.get(city, f"{city}: 景点搜索失败，返回信息：{data}")

        pois = data["pois"][:5]
        lines = []
        for index, poi in enumerate(pois, start=1):
            name = poi.get("name", "")
            address = poi.get("address", "")
            poi_type = poi.get("type", "")
            rating = ""
            biz_ext = poi.get("biz_ext")
            if isinstance(biz_ext, dict):
                rating = biz_ext.get("rating", "")

            rating_text = f"，评分 {rating}" if rating else ""
            lines.append(
                f"{index}. {name}，类型：{poi_type}，地址：{address}{rating_text}"
            )

        return (
            f"根据高德 POI 搜索，{city} 可选景点如下：\n"
            + "\n".join(lines)
            + f"\n当前天气信息：{weather}\n"
            "请结合天气、温度、舒适度和游览便利性，从中推荐一个最合适的景点。"
        )
    except Exception as exc:
        return local_attractions.get(
            city,
            f"联网搜索景点失败：{exc}。建议在{city}选择当地知名、交通便利、可根据天气灵活调整的景点。"
        )
CITY_ADCODE_CACHE = {
    "北京": "110000",
    "北京市": "110000",
    "上海": "310000",
    "上海市": "310000",
    "广州": "440100",
    "广州市": "440100",
    "深圳": "440300",
    "深圳市": "440300",
    "杭州": "330100",
    "杭州市": "330100",
    "成都": "510100",
    "成都市": "510100",
    "天津": "120000",
    "天津市": "120000",
    "重庆": "500000",
    "重庆市": "500000",
    "南京": "320100",
    "南京市": "320100",
    "武汉": "420100",
    "武汉市": "420100",
    "西安": "610100",
    "西安市": "610100",
}


PLACE_ALIASES = {
    "故宫": "故宫午门",
    "故宫博物院": "故宫午门",
    "紫禁城": "故宫午门",
    "颐和园": "颐和园东宫门",
    "圆明园": "圆明园南门",
    "天坛": "天坛公园南门",
    "上海迪士尼": "上海迪士尼乐园主入口",
    "迪士尼": "上海迪士尼乐园主入口",
}


def resolve_city_adcode(city: str) -> str:
    city = (city or "").strip()
    if not city:
        return city

    if city in CITY_ADCODE_CACHE:
        return CITY_ADCODE_CACHE[city]

    amap_key = os.getenv("AMAP_API_KEY")
    if not amap_key:
        return city

    try:
        response = requests.get(
            "https://restapi.amap.com/v3/config/district",
            params={
                "key": amap_key,
                "keywords": city,
                "subdistrict": 0,
                "extensions": "base",
                "output": "JSON",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return city

    districts = data.get("districts") or []
    if not districts:
        return city

    adcode = districts[0].get("adcode")
    name = districts[0].get("name")

    if adcode:
        CITY_ADCODE_CACHE[city] = adcode
        if name:
            CITY_ADCODE_CACHE[name] = adcode
        return adcode

    return city


def resolve_place_location(city: str, keyword: str):
    amap_key = os.getenv("AMAP_API_KEY")
    if not amap_key:
        raise ValueError("未配置 AMAP_API_KEY")

    adcode = resolve_city_adcode(city)
    search_keyword = PLACE_ALIASES.get(keyword, keyword)

    response = requests.get(
        "https://restapi.amap.com/v3/place/text",
        params={
            "key": amap_key,
            "keywords": search_keyword,
            "city": adcode,
            "citylimit": "true",
            "offset": 1,
            "page": 1,
            "extensions": "base",
            "output": "JSON",
        },
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "1" or not data.get("pois"):
        raise ValueError(f"没有找到地点：{keyword}")

    poi = data["pois"][0]
    return {
        "name": poi.get("name", search_keyword),
        "address": poi.get("address", ""),
        "location": poi.get("location", ""),
    }

def plan_transit_route(city: str, origin_place: dict, destination_place: dict) -> str:
    amap_key = os.getenv("AMAP_API_KEY")
    if not amap_key:
        return "未配置 AMAP_API_KEY，无法进行公共交通路线规划。"

    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/transit/integrated",
            params={
                "key": amap_key,
                "origin": origin_place["location"],
                "destination": destination_place["location"],
                "city": resolve_city_adcode(city),
                "strategy": 0,
                "extensions": "base",
                "output": "JSON",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log_tool_exception("amap_transit", exc)
        return f"公共交通路线规划失败：无法连接高德路线服务，原因：{exc}"

    if data.get("status") != "1":
        log_amap_error("amap_transit", data)
        return f"公共交通路线规划失败：{data.get('info', '未知错误')}"

    route = data.get("route") or {}
    transits = route.get("transits") or []
    if not transits:
        return "公共交通路线规划失败：没有找到可用的公交或地铁方案。"

    plan = min(
    transits,
    key=lambda item: int(float(item.get("walking_distance") or 999999))
)

    def to_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    duration_min = round(to_int(plan.get("duration")) / 60)
    walking_m = to_int(plan.get("walking_distance"))
    cost = plan.get("cost")

    lines = [
        f"公共交通路线：{origin_place['name']} -> {destination_place['name']}",
        f"预计耗时：约 {duration_min} 分钟",
        f"步行距离：约 {walking_m} 米",
    ]

    if cost:
        lines.append(f"预计费用：约 {cost} 元")

    lines.append("主要换乘步骤：")

    step_index = 1
    for segment in plan.get("segments", []):
        walking = segment.get("walking") or {}
        walking_distance = to_int(walking.get("distance"))
        if walking_distance:
            lines.append(f"{step_index}. 步行约 {walking_distance} 米")
            step_index += 1

        bus = segment.get("bus") or {}
        buslines = bus.get("buslines") or []
        if buslines:
            busline = buslines[0]
            name = busline.get("name", "公交/地铁")
            departure = busline.get("departure_stop", {}).get("name", "上车站")
            arrival = busline.get("arrival_stop", {}).get("name", "下车站")
            via_num = busline.get("via_num", "若干")
            lines.append(f"{step_index}. 乘坐 {name}：{departure} -> {arrival}，经过 {via_num} 站")
            step_index += 1

    if origin_place.get("address"):
        lines.append(f"起点地址：{origin_place['address']}")
    if destination_place.get("address"):
        lines.append(f"终点地址：{destination_place['address']}")

    return "\n".join(lines)

def plan_route(city: str, origin: str, destination: str, mode: str = "walking") -> str:
    amap_key = os.getenv("AMAP_API_KEY")
    if not amap_key:
        return "未配置 AMAP_API_KEY，无法进行路线规划。"

    try:
        origin_place = resolve_place_location(city, origin)
        destination_place = resolve_place_location(city, destination)
    except Exception as exc:
        return f"地点解析失败：{exc}"

    mode = (mode or "walking").lower()
    if mode in ["transit", "bus", "subway", "public"]:
        return plan_transit_route(city, origin_place, destination_place)
    if mode == "driving":
        url = "https://restapi.amap.com/v3/direction/driving"
        mode_label = "驾车"
    else:
        url = "https://restapi.amap.com/v3/direction/walking"
        mode = "walking"
        mode_label = "步行"

    try:
        response = requests.get(
            url,
            params={
                "key": amap_key,
                "origin": origin_place["location"],
                "destination": destination_place["location"],
                "output": "JSON",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return f"路线规划失败：无法连接高德路线服务，原因：{exc}"

    if data.get("status") != "1":
        return f"路线规划失败：{data.get('info', '未知错误')}"

    route = data.get("route") or {}
    paths = route.get("paths") or []
    if not paths:
        return "路线规划失败：没有找到可用路线。"

    path_data = paths[0]

    def to_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    distance_m = to_int(path_data.get("distance"))
    duration_s = to_int(path_data.get("duration"))

    distance_text = f"{distance_m / 1000:.2f} 公里" if distance_m else "未知"
    duration_text = f"{round(duration_s / 60)} 分钟" if duration_s else "未知"

    lines = [
        f"路线规划：{origin_place['name']} -> {destination_place['name']}",
        f"出行方式：{mode_label}",
        f"距离：约 {distance_text}",
        f"耗时：约 {duration_text}",
    ]

    if mode == "walking" and distance_m >= 3000:
        lines.append("提示：步行距离较长，不太建议全程步行；如果是游客，更建议打车或换成公共交通。")
    elif mode == "walking" and distance_m >= 1500:
        lines.append("提示：步行距离中等，建议穿舒适鞋子，并根据天气安排休息。")

    steps = path_data.get("steps") or []
    instructions = []

    for step in steps:
        instruction = step.get("instruction")
        step_distance = to_int(step.get("distance"))

        if not instruction:
            continue

        if step_distance:
            instructions.append(f"{instruction}（约{step_distance}米）")
        else:
            instructions.append(instruction)

    if instructions:
        lines.append("主要步骤：")

        if len(instructions) <= 10:
            for index, instruction in enumerate(instructions, 1):
                lines.append(f"{index}. {instruction}")
        else:
            shown_steps = instructions[:5]
            tail_steps = instructions[-3:]

            for index, instruction in enumerate(shown_steps, 1):
                lines.append(f"{index}. {instruction}")

            lines.append(f"... 中间省略 {len(instructions) - 8} 步 ...")

            start_index = len(instructions) - len(tail_steps) + 1
            for offset, instruction in enumerate(tail_steps):
                lines.append(f"{start_index + offset}. {instruction}")
    else:
        lines.append("主要步骤：高德返回了路线距离和耗时，但没有返回详细分步导航。")

    if origin_place.get("address"):
        lines.append(f"起点地址：{origin_place['address']}")
    if destination_place.get("address"):
        lines.append(f"终点地址：{destination_place['address']}")

    return "\n".join(lines)




def search_travel_guide_tool(city: str, query: str) -> str:
    try:
        results = search_travel_guide(query=query, city=city, top_k=3)
        return format_search_results(results)
    except Exception as exc:
        return f"旅行攻略检索失败：{exc}"


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
    "plan_route": plan_route,
    "search_travel_guide": search_travel_guide_tool,
}

MAX_TOOL_CALLS_PER_REQUEST = 6

def make_tool_call_key(tool_name: str, kwargs: dict):
    canonical_args = tuple(sorted((str(key), str(value)) for key, value in kwargs.items()))
    return (tool_name, canonical_args)


def should_retry_tool_result(result: str) -> bool:
    text = str(result).lower()
    retry_markers = [
        "timeout", "timed out", "超时", "无法连接", "连接失败",
        "connection", "temporarily", "临时", "429", "502", "503", "504",
    ]
    return any(marker in text for marker in retry_markers)


def call_tool(tool_name: str, kwargs: dict, called_tools: set,trace_id: str) -> str:
    if tool_name not in available_tools:
        return f"错误：未定义的工具 {tool_name}"

    call_key = make_tool_call_key(tool_name, kwargs)
    if call_key in called_tools:
        return f"工具调用被跳过：本轮已经用相同参数调用过 {tool_name}，请基于已有 Observation 总结回答，或更换必要工具。"

    called_tools.add(call_key)
    safe_kwargs = {
        key: ("***" if "key" in key.lower() or "token" in key.lower() else value)
        for key, value in kwargs.items()
    }

    last_result = ""
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        start_time = time.time()
        try:
            result = available_tools[tool_name](**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)
            record_tool_result(tool_name, latency_ms, True)
            log_event("tool_completed", trace_id=trace_id, tool=tool_name, attempt=attempt, latency_ms=latency_ms, args=safe_kwargs)
            if attempt < max_attempts and should_retry_tool_result(result):
                last_result = result
                time.sleep(0.5 * attempt)
                continue

            return result
        except Exception as exc:
            latency_ms = int((time.time() - start_time) * 1000)
            last_result = f"工具执行失败：{type(exc).__name__}: {exc}"
            record_tool_result(tool_name, latency_ms, False)
            log_event("tool_failed", trace_id=trace_id, tool=tool_name, attempt=attempt, latency_ms=latency_ms, args=safe_kwargs, error=str(exc))
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)

    return last_result or "工具执行失败：未知错误。"

def parse_action(action_str: str):
    if action_str.startswith("Finish"):
        final_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
        if final_match:
            return "finish", final_match.group(1).strip()
        return "finish", action_str

    tool_match = re.search(r"(\w+)\((.*)\)", action_str)
    if not tool_match:
        return "error", "Action 格式不正确。"

    tool_name = tool_match.group(1)
    args_str = tool_match.group(2)

    kwargs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str))
    if not kwargs:
        kwargs = dict(re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str))

    return "tool", (tool_name, kwargs)

def build_message_with_memory(session_id: str, message: str) -> str:
    history = memory_store.get_recent_messages(session_id, MAX_MEMORY_MESSAGES)

    if not history:
        return message

    history_text = []
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            history_text.append(f"用户：{content}")
        elif role == "assistant":
            history_text.append(f"助手：{content}")

    return (
        "以下是当前会话的最近对话历史，请结合上下文回答用户的新问题。\n\n"
        + "\n".join(history_text)
        + "\n\n用户的新问题："
        + message
    )


def save_memory(session_id: str, user_message: str, agent_answer: str):
    memory_store.save_turn(session_id, user_message, agent_answer)
def run_agent(user_input: str,trace_id: str) -> str:
    prompt_history = [f"用户输入: {user_input}"]
    called_tools = set()
    tool_call_count = 0

    for step in range(5):
        prompt = "\n".join(prompt_history)
        llm_start = time.time()
        llm_output = call_llm(prompt)
        llm_latency_ms = int((time.time() - llm_start) * 1000)
        record_llm_latency(llm_latency_ms)
        log_event("llm_completed", trace_id=trace_id, step=step + 1, latency_ms=llm_latency_ms)
#        print(f"\n=== STEP {step + 1} LLM OUTPUT ===")
#       print(llm_output)

        prompt_history.append(llm_output)

        finish_match = re.search(r"Finish\[(.*)\]", llm_output, re.DOTALL)
        if finish_match:
            return finish_match.group(1).strip()

        action_match = re.search(r"Action:\s*([^\n]+)", llm_output)
        if not action_match:
            observation = "错误：没有解析到 Action，请严格使用 Thought 和 Action 格式。"
        #    print("\n=== OBSERVATION ===")
         #   print(observation)
            prompt_history.append(f"Observation: {observation}")
            continue

        action_str = action_match.group(1).strip()
        action_type, payload = parse_action(action_str)

        if action_type == "finish":
            return payload

        if action_type == "error":
            observation = payload
 #           print("\n=== OBSERVATION ===")
 #           print(observation)
            prompt_history.append(f"Observation: {observation}")
            continue

        tool_name, kwargs = payload

        if tool_call_count >= MAX_TOOL_CALLS_PER_REQUEST:
            observation = "工具调用次数已达到上限，请基于已有信息总结回答；如果信息不足，请向用户说明还缺少哪些条件。"
        else:
            tool_call_count += 1
            observation = call_tool(tool_name, kwargs, called_tools,trace_id) 
   # print("\n=== OBSERVATION ===")
       # print(observation)
        prompt_history.append(f"Observation: {observation}")

    return "任务没有在最大循环次数内完成，请换一种问法再试。"


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/admin/stats")
def admin_stats(x_api_key: str = Header(default="")):
    expected_key = get_service_api_key()
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return build_stats_snapshot()


def answer_route_missing_destination(message: str):
    text = (message or "").strip()
    if not text:
        return None

    route_keywords = ["规划路线", "路线", "怎么走", "怎么坐车", "怎么过去", "如何过去"]
    if not any(keyword in text for keyword in route_keywords):
        return None

    origin = ""
    origin_patterns = [
        r"(?:我现在在|我在|当前位置在)([^到去，。,.、\s]+)",
        r"从([^到去，。,.、\s]+)",
    ]

    for pattern in origin_patterns:
        match = re.search(pattern, text)
        if match:
            origin = match.group(1).strip()
            break

    if not origin:
        return None

    destination_match = re.search(r"(?:到|去|前往)([^，。,.、\s？?]+)", text)
    if destination_match:
        destination = destination_match.group(1).strip()
        generic_words = {"哪里", "哪儿", "哪", "目的地"}
        if destination and destination not in generic_words:
            return None

    return f"已知你从{origin}出发，请提供目的地，我再为你规划路线。"



def detect_rag_context_city(message: str):
    text = (message or "").strip()
    if not text:
        return ""

    route_keywords = ["路线", "怎么走", "怎么坐车", "公共交通", "驾车", "步行"]
    if re.search(r"从.+到", text) or any(keyword in text for keyword in route_keywords):
        return ""

    rag_keywords = [
        "适合去哪",
        "去哪玩",
        "去哪里玩",
        "哪里玩",
        "推荐景点",
        "旅游景点",
        "攻略",
        "需要注意",
        "下雨",
        "雨天",
        "晴天",
        "上午适合",
        "高温",
    ]

    if not any(keyword in text for keyword in rag_keywords):
        return ""

    for city in ["北京", "上海", "成都", "广州", "杭州", "深圳"]:
        if city in text:
            return city

    return ""


def build_message_with_rag_context(message: str, trace_id: str):
    city = detect_rag_context_city(message)
    if not city:
        return message

    guide_text = call_tool(
        "search_travel_guide",
        {"city": city, "query": message},
        set(),
        trace_id,
    )

    bad_markers = ["没有检索到", "失败", "错误", "未定义"]
    if not guide_text or any(marker in guide_text for marker in bad_markers):
        return message

    return (
        f"{message}\n\n"
        "【旅行攻略知识库检索结果】\n"
        f"{guide_text}\n\n"
        "回答要求：必须优先参考上面的旅行攻略知识库；"
        "如果推荐景点，最终答案至少包含检索结果中出现的一个景点名；"
        "不要输出相似度。"
    )


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(default="")):
    trace_id = uuid.uuid4().hex[:12]
    expected_key = get_service_api_key()
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        record_chat_started()
        log_event("chat_started", trace_id=trace_id, session_id=req.session_id)
        direct_answer = answer_route_missing_destination(req.message)
        if direct_answer:
            save_memory(req.session_id, req.message, direct_answer)
            record_chat_success()
            log_event("chat_completed", trace_id=trace_id, session_id=req.session_id)
            return {"answer": direct_answer, "session_id": req.session_id, "trace_id": trace_id}

        message_for_agent = build_message_with_rag_context(req.message, trace_id)
        message_with_memory = build_message_with_memory(req.session_id, message_for_agent)
        answer = run_agent(message_with_memory, trace_id)
        save_memory(req.session_id, req.message, answer)
        record_chat_success()
        log_event("chat_completed", trace_id=trace_id, session_id=req.session_id)
        return {"answer": answer, "session_id": req.session_id, "trace_id": trace_id}
    except Exception as exc:
        record_chat_error(trace_id, req.session_id, type(exc).__name__, str(exc))
        log_event("chat_failed", trace_id=trace_id, session_id=req.session_id, error_type=type(exc).__name__, error=str(exc))
        return {
            "answer": "抱歉，智能体执行过程中遇到临时错误。请稍后重试，或换一种问法。",
            "session_id": req.session_id,
            "trace_id": trace_id,
        }
@app.post("/clear_session")
def clear_session(req: ClearSessionRequest, x_api_key: str = Header(default="")):
    expected_key = get_service_api_key()
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    memory_store.clear_session(req.session_id)
    return {"status": "ok", "session_id": req.session_id}
