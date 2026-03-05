"""
LLAISYS Session Manager — 多会话管理 + KV-Cache 快照 + 编辑/重新生成

每个 Session 维护:
  - session_id: 唯一标识
  - history: 对话历史 [{"role": ..., "content": ...}, ...]
  - cache_snapshot: 对应 KV-Cache 快照句柄
  - token_positions: 每条消息对应的 KV-Cache 起止位置
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class MessageRecord:
    """单条消息记录, 附带 KV-Cache 位置信息."""
    role: str
    content: str
    # 该消息对应的 token IDs (encode 后)
    token_ids: List[int] = field(default_factory=list)
    # 该消息 tokens 在 KV-Cache 中的起始位置
    cache_start_pos: int = 0
    # 该消息 tokens 在 KV-Cache 中的结束位置 (不含生成 token)
    cache_end_pos: int = 0


@dataclass
class Session:
    """单个会话."""
    session_id: str
    history: List[MessageRecord] = field(default_factory=list)
    cache_snapshot: Optional[object] = None  # C++ 快照句柄
    # 最后一次完整 prefill 的 token 数 (即 KV-Cache 中已有的 prompt 位置)
    last_prefill_pos: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionManager:
    """管理多个会话, 提供 KV-Cache 感知的会话切换/编辑/重新生成."""

    def __init__(self, model):
        """
        Args:
            model: Qwen2 模型实例 (具备 save_cache/restore_cache/truncate_cache 等方法).
        """
        self.model = model
        self._sessions: Dict[str, Session] = {}
        self._active_session_id: Optional[str] = None

    # ── 会话 CRUD ──

    def create_session(self, session_id: Optional[str] = None) -> Session:
        """创建新会话.
        
        Returns:
            新创建的 Session.
        """
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取指定会话."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """删除会话并释放其 KV-Cache 快照."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        if session.cache_snapshot:
            self.model.destroy_snapshot(session.cache_snapshot)
            session.cache_snapshot = None
        if self._active_session_id == session_id:
            self._active_session_id = None
        return True

    def list_sessions(self) -> List[dict]:
        """列出所有会话的摘要信息."""
        result = []
        for sid, sess in self._sessions.items():
            result.append({
                "session_id": sid,
                "message_count": len(sess.history),
                "created_at": sess.created_at,
                "updated_at": sess.updated_at,
                "is_active": sid == self._active_session_id,
            })
        return result

    @property
    def active_session_id(self) -> Optional[str]:
        return self._active_session_id

    # ── 会话切换 ──

    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话, 保存当前会话的 KV-Cache, 恢复目标会话的 KV-Cache.
        
        Returns:
            切换是否成功.
        """
        target = self._sessions.get(session_id)
        if target is None:
            return False

        # 保存当前活跃会话的 cache
        if self._active_session_id and self._active_session_id != session_id:
            current = self._sessions.get(self._active_session_id)
            if current:
                self._save_session_cache(current)

        # 恢复目标会话的 cache
        if target.cache_snapshot:
            self.model.restore_cache(target.cache_snapshot)
        else:
            self.model.reset_cache()

        self._active_session_id = session_id
        target.updated_at = time.time()
        return True

    def activate_or_create(self, session_id: Optional[str] = None) -> Session:
        """激活现有会话或创建新会话.
        
        如果 session_id 为 None, 创建新会话并激活.
        如果 session_id 指向已有会话, 切换到该会话.
        如果 session_id 不存在, 创建新会话并激活.
        """
        if session_id and session_id in self._sessions:
            self.switch_session(session_id)
            return self._sessions[session_id]

        session = self.create_session(session_id)
        
        # 保存当前活跃会话
        if self._active_session_id:
            current = self._sessions.get(self._active_session_id)
            if current:
                self._save_session_cache(current)

        self.model.reset_cache()
        self._active_session_id = session.session_id
        return session

    # ── 消息管理 ──

    def add_message(self, session_id: str, role: str, content: str,
                    token_ids: Optional[List[int]] = None,
                    cache_start_pos: int = 0, cache_end_pos: int = 0) -> MessageRecord:
        """向会话添加消息记录.
        
        Args:
            session_id: 会话 ID.
            role: 消息角色 ("user" / "assistant" / "system").
            content: 消息内容.
            token_ids: 编码后的 token 列表 (可选).
            cache_start_pos: 该消息在 KV-Cache 中的起始位置.
            cache_end_pos: 该消息在 KV-Cache 中的结束位置.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        msg = MessageRecord(
            role=role,
            content=content,
            token_ids=token_ids or [],
            cache_start_pos=cache_start_pos,
            cache_end_pos=cache_end_pos,
        )
        session.history.append(msg)
        session.updated_at = time.time()
        return msg

    def get_history(self, session_id: str) -> List[dict]:
        """获取对话历史 (用于 API 响应)."""
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return [
            {"role": m.role, "content": m.content}
            for m in session.history
        ]

    # ── 编辑 + 重新生成 ──

    def edit_message(self, session_id: str, message_index: int, new_content: str) -> Tuple[int, List[dict]]:
        """编辑指定位置的消息, 截断后续消息和 KV-Cache.
        
        Args:
            session_id: 会话 ID.
            message_index: 要编辑的消息索引 (0-based).
            new_content: 新的消息内容.
            
        Returns:
            (truncate_pos, removed_messages):
            - truncate_pos: KV-Cache 应截断到的位置.
            - removed_messages: 被删除的消息列表.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if message_index < 0 or message_index >= len(session.history):
            raise IndexError(f"Message index {message_index} out of range")

        target_msg = session.history[message_index]
        truncate_pos = target_msg.cache_start_pos

        # 收集被删除的消息
        removed = [
            {"role": m.role, "content": m.content}
            for m in session.history[message_index:]
        ]

        # 截断历史: 删除目标消息及之后的所有消息
        session.history = session.history[:message_index]

        # 截断 KV-Cache
        self.model.truncate_cache(truncate_pos)

        # 释放旧快照 (已失效)
        if session.cache_snapshot:
            self.model.destroy_snapshot(session.cache_snapshot)
            session.cache_snapshot = None

        session.last_prefill_pos = truncate_pos
        session.updated_at = time.time()

        return truncate_pos, removed

    def prepare_regenerate(self, session_id: str) -> Tuple[int, Optional[dict]]:
        """准备重新生成: 删除最后一条 assistant 消息, 截断 KV-Cache.
        
        Returns:
            (truncate_pos, removed_message):
            - truncate_pos: KV-Cache 截断位置.
            - removed_message: 被删除的 assistant 消息 (如果有).
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if not session.history:
            return 0, None

        # 找到最后一条 assistant 消息
        last = session.history[-1]
        if last.role != "assistant":
            return self.model.get_cache_pos(), None

        truncate_pos = last.cache_start_pos
        removed = {"role": last.role, "content": last.content}

        # 删除该消息
        session.history.pop()

        # 截断 KV-Cache
        self.model.truncate_cache(truncate_pos)

        # 释放旧快照
        if session.cache_snapshot:
            self.model.destroy_snapshot(session.cache_snapshot)
            session.cache_snapshot = None

        session.last_prefill_pos = truncate_pos
        session.updated_at = time.time()

        return truncate_pos, removed

    # ── 内部方法 ──

    def _save_session_cache(self, session: Session):
        """保存会话的 KV-Cache 快照."""
        if session.cache_snapshot:
            self.model.destroy_snapshot(session.cache_snapshot)
        session.cache_snapshot = self.model.save_cache()
        session.last_prefill_pos = self.model.get_cache_pos()
