# Travel Agent

基于 FastAPI + ReAct + RAG 的旅行智能体项目。

项目支持天气查询、景点推荐、POI 搜索、行政区解析、步行 / 驾车 / 公交路线规划、多轮会话记忆、旅行攻略 RAG 检索、API Key 鉴权、结构化日志、运行统计、Docker 部署和回归评测。

## 1. 项目简介

Travel Agent 是一个面向旅行场景的 AI Agent 后端服务。

它不是简单的聊天机器人，而是通过 ReAct 思路让大模型在多轮推理过程中主动选择工具，包括天气查询、景点推荐、POI 搜索、路线规划和旅行攻略 RAG 检索。

项目已部署在阿里云 ECS 上，支持：

- FastAPI 对外提供 RESTful API
- Nginx 反向代理
- systemd 服务守护
- Docker Compose 旁路部署
- SQLite 持久化会话记忆
- RAG 本地知识库检索
- 结构化日志和 trace_id 链路追踪
- `/admin/stats` 运行统计
- smoke test 与 eval 回归评测

当前 Docker v1 版本已经完成旁路部署，并连续通过 24 条回归评测用例。

## 2. 核心能力

- 基于 ReAct 思路实现 Agent 推理流程，支持 Thought / Action / Observation / Finish。
- 接入高德天气、POI 搜索、行政区查询和路径规划接口。
- 支持天气查询、景点推荐、步行路线、驾车路线和公交路线规划。
- 支持 SQLite 持久化多轮会话记忆，服务重启后上下文仍可恢复。
- 支持旅行攻略 RAG，能够检索城市攻略、景点介绍和出行建议。
- 支持 API Key 鉴权、健康检查、会话清理和运行统计。
- 支持 trace_id、结构化日志、工具调用耗时统计。
- 支持 systemd 部署和 Docker Compose 旁路部署。
- 提供 smoke test 和 eval 回归评测，形成部署验收闭环。

## 3. 架构说明

```mermaid
flowchart TD
    User[用户/前端] --> Nginx[Nginx 反向代理]
    Nginx --> FastAPI[FastAPI API 服务]

    FastAPI --> Auth[API Key 鉴权]
    FastAPI --> Memory[SQLite 会话记忆]
    FastAPI --> Agent[ReAct Agent]

    Agent --> LLM[qwen-plus]
    Agent --> Weather[高德天气]
    Agent --> POI[高德 POI 搜索]
    Agent --> Route[高德路线规划]
    Agent --> RAG[旅行攻略 RAG]

    RAG --> Knowledge[knowledge/*.md]
    RAG --> Embedding[text-embedding-v4]
    RAG --> VectorDB[SQLite 向量存储]

    FastAPI --> Logs[结构化日志/trace_id]
    FastAPI --> Stats[/admin/stats]
```

## 4. 技术栈

- Python 3.10
- FastAPI
- Uvicorn
- SQLite
- Docker / Docker Compose
- Nginx
- systemd
- OpenAI-compatible API
- qwen-plus
- text-embedding-v4
- 高德开放平台 API
- ReAct Agent
- RAG
- Markdown 知识库
- curl smoke test
- 自定义 eval 回归评测

## 5. 项目目录说明

```text
/opt/travel-agent
├── app.py                         # FastAPI 主服务与 Agent 编排逻辑
├── rag.py                         # RAG 构建与检索逻辑
├── memory.py                      # SQLite 会话记忆模块
├── observability.py               # 运行统计与可观测性模块
├── requirements.txt               # Python 依赖
├── Dockerfile                     # Docker 镜像构建文件
├── docker-compose.yml             # Docker Compose 配置
├── .dockerignore                  # Docker 构建忽略文件
├── .env                           # 环境变量文件，不应提交
├── knowledge/                     # RAG 知识库
│   ├── beijing.md
│   ├── shanghai.md
│   └── chengdu.md
├── rag_data/                      # RAG 向量数据
├── docker_data/                   # Docker 版独立运行数据
│   └── memory.db
├── eval/
│   ├── run_eval.py                # 回归评测脚本
│   ├── cases.jsonl                # 评测用例
│   └── reports/                   # 评测报告与 badcase
└── scripts/
    └── docker-smoke-test.sh       # smoke test 脚本
```

## 6. 环境变量

项目根目录需要 `.env` 文件。

示例：

```bash
LLM_API_KEY=你的通义千问API Key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024

AMAP_API_KEY=你的高德API Key

API_KEY=你的服务访问密钥
AGENT_API_KEY=你的服务访问密钥
```

说明：

- `LLM_API_KEY`：大模型 API Key。
- `LLM_BASE_URL`：OpenAI-compatible API 地址。
- `LLM_MODEL`：Agent 推理模型。
- `EMBEDDING_MODEL`：RAG 向量化模型。
- `EMBEDDING_DIMENSIONS`：Embedding 向量维度。
- `AMAP_API_KEY`：高德开放平台 Key。
- `API_KEY` / `AGENT_API_KEY`：服务接口鉴权密钥。

注意：`.env` 不能提交到 Git 仓库。

## 7. 接口说明

### 7.1 GET /health

健康检查。

```bash
curl http://127.0.0.1:8000/health
```

返回：

```json
{"status":"ok"}
```

### 7.2 POST /chat

智能体对话接口。

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 你的服务访问密钥" \
  -d '{"session_id":"demo-1","message":"上海下雨适合去哪玩？"}'
```

示例返回：

```json
{
  "answer": "上海当前天气为阴天，适合选择东方明珠、上海博物馆等室内或遮蔽性较好的景点。",
  "session_id": "demo-1",
  "trace_id": "xxxxxx"
}
```

### 7.3 POST /clear_session

清理指定会话记忆。

```bash
curl -X POST http://127.0.0.1:8000/clear_session \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 你的服务访问密钥" \
  -d '{"session_id":"demo-1"}'
```

示例返回：

```json
{
  "status": "ok",
  "session_id": "demo-1"
}
```

### 7.4 GET /admin/stats

查看运行统计。

```bash
curl http://127.0.0.1:8000/admin/stats \
  -H "X-API-Key: 你的服务访问密钥"
```

返回字段包括：

- `request_count`
- `success_count`
- `error_count`
- `success_rate`
- `llm_calls`
- `avg_llm_latency_ms`
- `tool_calls`
- `tool_errors`
- `tool_latency_ms_total`
- `last_error`

## 8. systemd 部署

当前正式服务通过 systemd 管理，监听：

```text
127.0.0.1:8000
```

常用命令：

```bash
sudo systemctl status travel-agent --no-pager
sudo systemctl restart travel-agent
sudo journalctl -u travel-agent -n 100 --no-pager
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

服务状态正常时返回：

```json
{"status":"ok"}
```

## 9. Docker 旁路部署

Docker 版服务通过 Docker Compose 管理，监听：

```text
127.0.0.1:8001
```

当前部署策略：

```text
systemd 正式版：127.0.0.1:8000
Docker 旁路版：127.0.0.1:8001
```

Docker 版先作为旁路验证环境，不直接替换正式服务。

### 9.1 构建镜像

```bash
docker compose build
```

### 9.2 启动服务

```bash
docker compose up -d
```

### 9.3 查看状态

```bash
docker compose ps
docker compose logs --tail=120
```

正常情况下可以看到：

```text
travel-agent-docker   Up ... (healthy)   127.0.0.1:8001->8000/tcp
```

### 9.4 停止服务

```bash
docker compose down
```

### 9.5 Docker 版健康检查

```bash
curl http://127.0.0.1:8001/health
```

返回：

```json
{"status":"ok"}
```

## 10. Docker 数据挂载说明

Docker Compose 挂载关系：

```text
.env                  -> /app/.env:ro
knowledge/            -> /app/knowledge:ro
rag_data/             -> /app/rag_data
docker_data/memory.db -> /app/memory.db
```

说明：

- `.env` 只读挂载，避免容器内修改密钥。
- `knowledge/` 只读挂载，作为 RAG 知识库。
- `rag_data/` 持久化 RAG 向量数据。
- `docker_data/memory.db` 是 Docker 版独立会话数据库，避免和 systemd 版本同时写同一个 SQLite 文件。

## 11. Smoke Test

项目提供 smoke test 脚本：

```bash
scripts/docker-smoke-test.sh
```

脚本会检查：

- `/health`
- `/clear_session`
- `/chat`
- `/admin/stats`

测试 Docker 旁路版：

```bash
bash scripts/docker-smoke-test.sh
```

测试 systemd 正式版：

```bash
AGENT_API_BASE=http://127.0.0.1:8000 bash scripts/docker-smoke-test.sh
```

成功时输出：

```text
SMOKE TEST PASSED
```

## 12. 回归评测

评测脚本：

```bash
eval/run_eval.py
```

评测 systemd 正式版：

```bash
python3 eval/run_eval.py
```

评测 Docker 旁路版：

```bash
AGENT_API_BASE=http://127.0.0.1:8001 python3 eval/run_eval.py
```

评测覆盖：

- 天气查询
- 天气 + 景点推荐
- 公交路线规划
- 多轮上下文
- 城市切换
- 缺少起点
- 缺少目的地
- 非法地点
- 无关问题
- Prompt 注入
- RAG 攻略问答

当前 Docker v1 基线：

```text
total: 24
passed: 24
failed: 0
pass_rate: 1.0
```

基线文件：

```text
eval/reports/baseline_eval_docker_v1.json
eval/reports/baseline_badcases_docker_v1.md
```

## 13. RAG 说明

RAG 知识库目录：

```text
knowledge/
```

当前包含：

```text
beijing.md
shanghai.md
chengdu.md
```

构建 RAG 索引：

```bash
python3 rag.py build
```

检索测试：

```bash
python3 rag.py search --city 北京 晴天适合去哪
python3 rag.py search --city 上海 下雨适合去哪
python3 rag.py search --city 成都 上午适合看什么
```

RAG 流程：

```text
Markdown 攻略文档
-> 文档切分
-> Embedding 向量化
-> SQLite 本地向量存储
-> Query 向量化
-> 余弦相似度 Top-K 检索
-> 注入 Agent 上下文
-> 生成最终回答
```

## 14. 可观测性

系统支持 trace_id 和结构化日志。

一次 `/chat` 请求会记录：

- `chat_started`
- `llm_completed`
- `tool_completed`
- `chat_completed`
- `chat_failed`

示例日志字段：

```text
trace_id
session_id
tool
attempt
latency_ms
args
```

查看 systemd 日志：

```bash
sudo journalctl -u travel-agent -n 100 --no-pager
```

查看 Docker 日志：

```bash
docker compose logs --tail=120
```

## 15. 常用验收命令

```bash
# systemd 正式版 smoke test
AGENT_API_BASE=http://127.0.0.1:8000 bash scripts/docker-smoke-test.sh

# Docker 旁路版 smoke test
bash scripts/docker-smoke-test.sh

# systemd 正式版回归评测
python3 eval/run_eval.py

# Docker 旁路版回归评测
AGENT_API_BASE=http://127.0.0.1:8001 python3 eval/run_eval.py

# 查看 Docker 状态
docker compose ps
docker compose logs --tail=120

# 查看 systemd 状态
sudo systemctl status travel-agent --no-pager
sudo journalctl -u travel-agent -n 100 --no-pager
```

## 16. 当前验证结果

Docker 旁路版 smoke test：

```text
SMOKE TEST PASSED
```

systemd 正式版 smoke test：

```text
SMOKE TEST PASSED
```

Docker 旁路版连续两次回归评测：

```text
total: 24
passed: 24
failed: 0
pass_rate: 1.0
```

说明当前 Docker 版具备可复现部署能力，并且与 systemd 正式版核心能力一致。

## 17. 项目亮点

- 不只是简单聊天机器人，而是具备工具调用、记忆、RAG 和部署闭环的工程化 Agent。
- 使用 ReAct 实现多步推理和工具编排。
- 使用 SQLite 实现服务重启后的上下文恢复。
- 使用 RAG 提升旅行攻略类回答的事实稳定性。
- 使用确定性 RAG 上下文增强，解决模型不稳定调用 RAG 工具的问题。
- 使用 trace_id 和结构化日志提升问题排查能力。
- 使用 `/admin/stats` 统计请求量、成功率、工具调用次数和延迟。
- 使用 Docker Compose 实现可复现部署。
- 使用 smoke test 和 eval 回归评测形成上线前验收闭环。
- Docker 版和 systemd 版可以并行运行，降低部署切换风险。

## 18. 后续优化方向

- 增加更多城市攻略知识库。
- 将 Markdown 攻略升级为结构化 JSON / YAML。
- 引入更专业的向量数据库，如 FAISS、Milvus、Qdrant 或 pgvector。
- 增加 Rerank，提高 RAG 检索质量。
- 增加并发压测和长输入边界测试。
- 将 `/admin/stats` 接入 Prometheus / Grafana。
- 增加 HTTPS 和域名。
- 将 Docker 旁路版切换为正式服务。
- 完善 CI/CD 流程，自动执行 smoke test 和 eval。
