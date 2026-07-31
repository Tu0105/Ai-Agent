"""
Agent核心逻辑模块
创建和管理LangChain Agent，处理用户请求
"""

import os
import time
import uuid
from collections import OrderedDict
from threading import RLock
from typing import AsyncGenerator, Optional, Dict, Any, Callable
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import ChatGeneration, GenerationChunk
from langchain_core.messages import AIMessageChunk
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm import get_llm
from .tools import get_all_tools, get_tool_names
from .memory import get_memory_manager, ConversationMemoryManager


# 系统提示词
SYSTEM_PROMPT = """你是一个专业、友好的在线客服助手，隶属于星云科技客户服务中心。

## 你的职责
1. 热情、耐心地解答客户的问题
2. 准确提供产品和服务信息
3. 帮助客户处理订单、售后等问题
4. 引导客户使用自助服务

## 服务原则
- 😊 保持友好、专业的服务态度
- 📝 回答简洁明了，避免过于冗长
- ✅ 主动询问是否需要其他帮助
- 🔧 对于无法处理的问题，引导转人工

## 可用工具
你可以通过以下工具来帮助客户：
- get_weather: 查询天气信息
- calculate: 数学计算
- get_current_time: 获取当前时间
- search_knowledge: 搜索知识库
- get_order_status: 查询订单状态
- FAQ_answer: 常见问题解答
- get_user_info: 查询用户信息
- create_ticket: 创建客服工单

## 回答风格
- 使用友好的称呼，如"亲"、"您好"等
- 适当使用emoji增加亲和力
- 复杂问题分点说明
- 结束时询问是否还有其他问题

## 注意事项
- 不要编造不存在的功能或信息
- 如果不确定，请如实告知客户
- 涉及敏感操作（如退款、修改密码）需验证身份
"""


class CustomerServiceAgent:
    """客服Agent类"""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
        memory_manager: Optional[ConversationMemoryManager] = None
    ):
        """
        初始化客服Agent
        
        Args:
            system_prompt: 自定义系统提示词
            conversation_id: 对话ID
            memory_manager: 记忆管理器实例
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.memory_manager = memory_manager or get_memory_manager()
        
        # 初始化Agent
        self.llm = get_llm()
        self.tools = get_all_tools()
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """
        创建Agent执行器
        
        Returns:
            AgentExecutor: 配置好的Agent执行器
        """
        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建Agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # 创建Agent执行器
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory_manager.get_or_create_memory(self.conversation_id)[1],
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )
        
        return agent_executor
    
    def invoke(self, message: str) -> Dict[str, Any]:
        """
        同步调用Agent
        
        Args:
            message: 用户消息
        
        Returns:
            Dict: 包含响应和元数据的字典
        """
        try:
            # 调用Agent
            response = self.agent_executor.invoke({
                "input": message
            })
            
            # 添加到记忆
            self.memory_manager.add_user_message(self.conversation_id, message)
            self.memory_manager.add_ai_message(self.conversation_id, response["output"])
            
            return {
                "success": True,
                "message": response["output"],
                "conversation_id": self.conversation_id,
                "tool_used": self._get_used_tools(response)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "conversation_id": self.conversation_id
            }
    
    async def ainvoke(self, message: str) -> Dict[str, Any]:
        """
        异步调用Agent
        
        Args:
            message: 用户消息
        
        Returns:
            Dict: 包含响应和元数据的字典
        """
        try:
            response = await self.agent_executor.ainvoke({
                "input": message
            })
            
            # 添加到记忆
            self.memory_manager.add_user_message(self.conversation_id, message)
            self.memory_manager.add_ai_message(self.conversation_id, response["output"])
            
            return {
                "success": True,
                "message": response["output"],
                "conversation_id": self.conversation_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "conversation_id": self.conversation_id
            }
    
    def stream(self, message: str):
        """
        流式调用Agent
        
        Args:
            message: 用户消息
        
        Returns:
            Generator: 生成器，产生流式响应
        """
        # 添加用户消息到记忆
        self.memory_manager.add_user_message(self.conversation_id, message)
        
        # 使用流式响应
        for chunk in self.agent_executor.stream({
            "input": message
        }):
            if "output" in chunk:
                yield chunk["output"]
            elif "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        yield msg.content
        
        # 添加完整响应到记忆（延迟执行，由调用方完成）
    
    def _get_used_tools(self, response: Dict) -> list:
        """
        从响应中提取使用的工具
        
        Args:
            response: Agent响应
        
        Returns:
            list: 使用的工具列表
        """
        tools_used = []
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if len(step) > 0:
                    tool_name = getattr(step[0], "tool", None)
                    if tool_name:
                        tools_used.append(tool_name)
        return tools_used
    
    def get_conversation_id(self) -> str:
        """获取对话ID"""
        return self.conversation_id
    
    def clear_history(self) -> bool:
        """清除对话历史"""
        return self.memory_manager.clear_memory(self.conversation_id)


def _bounded_positive_int(name: str, default: int) -> int:
    """Read a positive cache limit without preventing service startup on bad config."""
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# Agent工厂函数
_AGENT_CACHE_MAX_SIZE = _bounded_positive_int("AGENT_CACHE_MAX_SIZE", 100)
_AGENT_CACHE_TTL_SECONDS = _bounded_positive_int("AGENT_CACHE_TTL_SECONDS", 1800)
_agent_cache: OrderedDict[str, tuple[CustomerServiceAgent, float]] = OrderedDict()
_agent_cache_lock = RLock()


def _evict_expired_agents(now: Optional[float] = None) -> None:
    """Remove expired entries before the cache is read or written."""
    current_time = now if now is not None else time.monotonic()
    expired = [
        conversation_id
        for conversation_id, (_, last_used) in _agent_cache.items()
        if current_time - last_used >= _AGENT_CACHE_TTL_SECONDS
    ]
    for conversation_id in expired:
        _agent_cache.pop(conversation_id, None)


def get_or_create_agent(conversation_id: Optional[str] = None) -> CustomerServiceAgent:
    """
    获取或创建Agent实例
    
    Args:
        conversation_id: 对话ID
    
    Returns:
        CustomerServiceAgent: Agent实例
    """
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    
    with _agent_cache_lock:
        now = time.monotonic()
        _evict_expired_agents(now)
        cached = _agent_cache.get(conversation_id)
        if cached is not None:
            agent, _ = cached
            _agent_cache.move_to_end(conversation_id)
            _agent_cache[conversation_id] = (agent, now)
            return agent

        agent = CustomerServiceAgent(conversation_id=conversation_id)
        _agent_cache[conversation_id] = (agent, now)
        while len(_agent_cache) > _AGENT_CACHE_MAX_SIZE:
            _agent_cache.popitem(last=False)
        return agent


def evict_agent(conversation_id: str) -> None:
    """Remove a deleted conversation from the in-process agent cache."""
    with _agent_cache_lock:
        _agent_cache.pop(conversation_id, None)


def invoke_agent(message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    调用Agent的便捷函数
    
    Args:
        message: 用户消息
        conversation_id: 对话ID
    
    Returns:
        Dict: 响应结果
    """
    agent = get_or_create_agent(conversation_id)
    result = agent.invoke(message)
    
    # 更新conversation_id（如果是新创建的）
    if result.get("conversation_id"):
        return result
    
    return {
        **result,
        "conversation_id": agent.get_conversation_id()
    }


def create_agent() -> CustomerServiceAgent:
    """
    创建新的Agent实例
    
    Returns:
        CustomerServiceAgent: 新的Agent实例
    """
    return CustomerServiceAgent()
