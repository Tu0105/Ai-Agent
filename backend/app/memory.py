"""
对话记忆模块
管理对话历史，支持持久化存储
"""

import os
import json
import uuid
from collections import OrderedDict
from typing import List, Optional, Dict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory


def _bounded_positive_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a configured limit while retaining a safe default on invalid input."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


MAX_CACHED_CONVERSATIONS = _bounded_positive_int("MEMORY_CACHE_MAX_CONVERSATIONS", 100)
MAX_MESSAGES_PER_CONVERSATION = _bounded_positive_int("MEMORY_MAX_MESSAGES", 100, minimum=2)


class ConversationMemoryManager:
    """对话记忆管理器"""
    
    def __init__(self, storage_dir: str = "./conversation_history"):
        """
        初始化记忆管理器
        
        Args:
            storage_dir: 记忆存储目录
        """
        self.storage_dir = storage_dir
        self.memory_store: OrderedDict[str, ConversationBufferMemory] = OrderedDict()
        
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_memory_file_path(self, conversation_id: str) -> str:
        """获取对话记忆文件路径"""
        return os.path.join(self.storage_dir, f"{conversation_id}.json")
    
    def _load_messages_from_file(self, conversation_id: str) -> List[Dict]:
        """从文件加载消息历史"""
        file_path = self._get_memory_file_path(conversation_id)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    return messages[-MAX_MESSAGES_PER_CONVERSATION:] if isinstance(messages, list) else []
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_messages_to_file(self, conversation_id: str, messages: List[Dict]) -> None:
        """保存消息历史到文件"""
        file_path = self._get_memory_file_path(conversation_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "conversation_id": conversation_id,
                    "updated_at": datetime.now().isoformat(),
                    "messages": messages
                }, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"保存对话历史失败: {e}")
    
    def get_or_create_memory(self, conversation_id: Optional[str] = None) -> tuple:
        """
        获取或创建对话记忆
        
        Args:
            conversation_id: 对话ID，如果为None则创建新的
        
        Returns:
            tuple: (conversation_id, memory)
        """
        # 如果没有提供conversation_id，生成一个新的
        if conversation_id is None or conversation_id not in self.memory_store:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            
            # 创建新的记忆
            memory = ConversationBufferMemory(
                return_messages=True,
                output_key="output",
                input_key="input"
            )
            
            # 加载历史消息（如果存在）
            historical_messages = self._load_messages_from_file(conversation_id)
            for msg_data in historical_messages:
                role = msg_data.get("role")
                content = msg_data.get("content")
                
                if role == "user":
                    memory.chat_memory.add_user_message(content)
                elif role == "assistant":
                    memory.chat_memory.add_ai_message(content)
                elif role == "system":
                    memory.chat_memory.add_system_message(content)
            
            self.memory_store[conversation_id] = memory
            while len(self.memory_store) > MAX_CACHED_CONVERSATIONS:
                self.memory_store.popitem(last=False)
        else:
            self.memory_store.move_to_end(conversation_id)
        
        return conversation_id, self.memory_store[conversation_id]
    
    def add_user_message(self, conversation_id: str, message: str) -> None:
        """
        添加用户消息到记忆
        
        Args:
            conversation_id: 对话ID
            message: 用户消息
        """
        _, memory = self.get_or_create_memory(conversation_id)
        memory.chat_memory.add_user_message(message)
        self._persist_memory(conversation_id)
    
    def add_ai_message(self, conversation_id: str, message: str) -> None:
        """
        添加AI消息到记忆
        
        Args:
            conversation_id: 对话ID
            message: AI消息
        """
        _, memory = self.get_or_create_memory(conversation_id)
        memory.chat_memory.add_ai_message(message)
        self._persist_memory(conversation_id)
    
    def get_history(self, conversation_id: str) -> List[Dict]:
        """
        获取对话历史
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            List[Dict]: 消息历史列表
        """
        _, memory = self.get_or_create_memory(conversation_id)
        
        messages = []
        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({
                    "role": "user",
                    "content": msg.content,
                    "timestamp": msg.additional_kwargs.get("timestamp", datetime.now().isoformat())
                })
            elif isinstance(msg, AIMessage):
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "timestamp": msg.additional_kwargs.get("timestamp", datetime.now().isoformat())
                })
            elif isinstance(msg, SystemMessage):
                messages.append({
                    "role": "system",
                    "content": msg.content,
                    "timestamp": msg.additional_kwargs.get("timestamp", datetime.now().isoformat())
                })
        
        return messages
    
    def _persist_memory(self, conversation_id: str) -> None:
        """
        将记忆持久化到文件
        
        Args:
            conversation_id: 对话ID
        """
        if conversation_id in self.memory_store:
            memory = self.memory_store[conversation_id]
            if len(memory.chat_memory.messages) > MAX_MESSAGES_PER_CONVERSATION:
                del memory.chat_memory.messages[:-MAX_MESSAGES_PER_CONVERSATION]
            messages = []
            
            for msg in memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    messages.append({"role": "system", "content": msg.content})
            
            self._save_messages_to_file(conversation_id, messages)
    
    def clear_memory(self, conversation_id: str) -> bool:
        """
        清除对话记忆
        
        Args:
            conversation_id: 对话ID
        
        Returns:
            bool: 是否成功清除
        """
        if conversation_id in self.memory_store:
            # 清除内存中的记忆
            self.memory_store[conversation_id].clear()
            
            # 删除持久化文件
            file_path = self._get_memory_file_path(conversation_id)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 从存储中移除
            del self.memory_store[conversation_id]
            return True
        
        return False
    
    def get_all_conversations(self) -> List[str]:
        """
        获取所有对话ID
        
        Returns:
            List[str]: 对话ID列表
        """
        return list(self.memory_store.keys())


# 全局记忆管理器实例
_memory_manager: Optional[ConversationMemoryManager] = None


def get_memory_manager() -> ConversationMemoryManager:
    """
    获取全局记忆管理器实例
    
    Returns:
        ConversationMemoryManager: 记忆管理器实例
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ConversationMemoryManager()
    return _memory_manager


def load_history(conversation_id: str) -> List[Dict]:
    """
    加载对话历史
    
    Args:
        conversation_id: 对话ID
    
    Returns:
        List[Dict]: 消息历史列表
    """
    return get_memory_manager().get_history(conversation_id)


def save_history(conversation_id: str, messages: List[Dict]) -> None:
    """
    保存对话历史
    
    Args:
        conversation_id: 对话ID
        messages: 消息列表
    """
    manager = get_memory_manager()
    manager.get_or_create_memory(conversation_id)
    
    # 清除现有记忆并重新保存
    manager.clear_memory(conversation_id)
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            manager.add_user_message(conversation_id, content)
        elif role == "assistant":
            manager.add_ai_message(conversation_id, content)
