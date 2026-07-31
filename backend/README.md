# AI Agent 客服系统 - 后端

基于 LangChain 的智能客服系统后端，使用 FastAPI 构建。

## 功能特性

- 🤖 **智能对话**: 基于 GPT-3.5/GPT-4 的自然语言处理
- 🔧 **多功能工具**: 支持天气查询、计算器、订单查询、知识库搜索等
- 💾 **对话记忆**: 支持多轮对话，自动保存对话历史
- 📡 **流式响应**: 支持 SSE 流式输出，实时展示 AI 回复
- 🌐 **跨域支持**: CORS 配置，支持前端应用访问
- 📚 **RESTful API**: 标准 REST 接口设计

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```
## 也可以用
```bash
python -X utf8 -m pip install -r requirements.txt
```
### 2. 配置环境变量

创建 `.env` 文件：

```env
# OpenAI API 配置
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# 可选配置
TEMPERATURE=0.7
MAX_TOKENS=2000
```

### 3. 启动服务

```bash
# 开发模式（热重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 接口

### 健康检查

```bash
GET /health
```

### 聊天（非流式）

```bash
POST /api/chat
Content-Type: application/json

{
    "message": "我想查询订单状态",
    "conversation_id": "可选的对话ID"
}
```

### 聊天（流式 SSE）

```bash
GET /api/chat/stream?message=你好&conversation_id=可选
```

或使用 POST：

```bash
POST /api/chat/stream
Content-Type: application/json

{
    "message": "你好",
    "conversation_id": "可选的对话ID"
}
```

### 获取对话历史

```bash
GET /api/history/{conversation_id}
```

### 清除对话历史

```bash
DELETE /api/history/{conversation_id}
```

## 项目结构

```
backend/
├── app/
│   ├── __init__.py       # 包初始化
│   ├── main.py           # FastAPI 入口
│   ├── llm.py            # LLM 配置
│   ├── agent.py          # Agent 核心
│   ├── tools.py          # 工具定义
│   ├── memory.py         # 对话记忆
│   └── schemas.py        # 数据模型
├── requirements.txt
└── README.md
```

## 可用工具

| 工具名称 | 功能描述 |
|---------|---------|
| `get_weather` | 查询城市天气信息 |
| `calculate` | 执行数学计算 |
| `get_current_time` | 获取当前时间 |
| `search_knowledge` | 搜索知识库 |
| `get_order_status` | 查询订单状态 |
| `FAQ_answer` | 常见问题解答 |
| `get_user_info` | 获取用户信息 |
| `create_ticket` | 创建客服工单 |

## 技术栈

- **Python**: 3.9+
- **Web 框架**: FastAPI 0.109.0
- **AI 框架**: LangChain 0.1.0
- **LLM**: OpenAI GPT-3.5/GPT-4
- **服务器**: Uvicorn

## 常见问题

### 1. API Key 未设置

确保设置了 `OPENAI_API_KEY` 环境变量。

### 2. CORS 跨域问题

已在 `main.py` 中配置 CORS，支持本地开发。如需调整，请修改 `allow_origins` 列表。

### 3. 对话历史丢失

对话历史默认保存在 `conversation_history/` 目录下的 JSON 文件中。

## License

MIT License

## 安全与运行边界

- 计算器仅支持数字、括号和 `+`、`-`、`*`、`/`；不会执行 Python 代码，并限制表达式长度、节点数和结果范围。
- `CORS_ALLOW_ORIGINS` 是逗号分隔的精确来源白名单。默认仅允许本地开发地址；通配符 `*` 会被忽略，且不应与凭据一起使用。
- 聊天消息最大 4,000 个字符，会话 ID 仅允许字母、数字、`_` 和 `-`，最大 128 个字符。
- 服务进程内最多缓存 100 个 Agent 和 100 个会话记忆，每个会话最多保留 100 条消息。可使用 `.env` 中的缓存变量按部署容量调整。
- API 不向客户端返回异常堆栈或上游服务错误详情；请通过服务端日志排查故障。

## 测试

```bash
cd backend
python -m unittest discover -s tests -v
python -m py_compile app/*.py
```
