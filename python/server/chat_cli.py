#!/usr/bin/env python3
"""
LLAISYS CLI Chat — 命令行聊天界面

连接到运行中的 LLAISYS 服务器，支持流式输出、多轮对话、会话管理、编辑和重新生成。

用法:
    python3 -m server.chat_cli [--url http://127.0.0.1:8000]

命令:
    /new            开始新对话 (创建新 session)
    /sessions       查看所有会话列表
    /switch <id>    切换到指定会话
    /history        查看当前对话历史
    /edit <n>       编辑第 n 条消息 (0-based) 并重新生成
    /regen          重新生成最后一条回复
    /delete <id>    删除指定会话
    /quit           退出
"""

import argparse
import json
import sys
import requests


def main():
    parser = argparse.ArgumentParser(description="LLAISYS CLI Chat")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000",
                        help="Server URL")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    api_url = f"{base_url}/v1/chat/completions"
    history: list[dict] = []
    session_id: str | None = None

    print("=" * 60)
    print("  LLAISYS Chat  (输入 /help 查看命令)")
    print("=" * 60)
    print()

    # 启动时自动创建会话
    try:
        resp = requests.post(f"{base_url}/v1/sessions", json={}, timeout=10)
        if resp.ok:
            session_id = resp.json().get("session_id")
            print(f"\033[90m[Session: {session_id}]\033[0m\n")
    except requests.ConnectionError:
        print("\033[93m[提示] 无法连接到服务器, 将在首次发送时重试\033[0m\n")

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── 命令处理 ──
        if user_input == "/quit":
            print("Bye!")
            break

        if user_input == "/help":
            print("""
  /new            开始新对话
  /sessions       查看所有会话
  /switch <id>    切换到指定会话
  /history        查看当前对话历史
  /edit <n>       编辑第 n 条消息并重新生成
  /regen          重新生成最后一条回复
  /delete <id>    删除指定会话
  /quit           退出
""")
            continue

        if user_input == "/new":
            history.clear()
            try:
                resp = requests.post(f"{base_url}/v1/sessions", json={}, timeout=10)
                if resp.ok:
                    session_id = resp.json().get("session_id")
                    print(f"\n--- 新对话已开始 [Session: {session_id}] ---\n")
                else:
                    print(f"\033[91m[错误] 创建会话失败: {resp.text}\033[0m")
            except requests.RequestException as e:
                print(f"\033[91m[错误] {e}\033[0m")
            continue

        if user_input == "/sessions":
            try:
                resp = requests.get(f"{base_url}/v1/sessions", timeout=10)
                if resp.ok:
                    sessions = resp.json().get("sessions", [])
                    if not sessions:
                        print("(无会话)")
                    else:
                        for s in sessions:
                            active = " \033[92m← 当前\033[0m" if s.get("is_active") else ""
                            print(f"  [{s['session_id']}] {s['message_count']} 条消息{active}")
                    print()
            except requests.RequestException as e:
                print(f"\033[91m[错误] {e}\033[0m")
            continue

        if user_input.startswith("/switch "):
            target_id = user_input[8:].strip()
            if not target_id:
                print("用法: /switch <session_id>")
                continue
            try:
                resp = requests.post(f"{base_url}/v1/sessions/{target_id}/switch", timeout=10)
                if resp.ok:
                    session_id = target_id
                    # 获取历史
                    hist_resp = requests.get(f"{base_url}/v1/sessions/{target_id}", timeout=10)
                    if hist_resp.ok:
                        msgs = hist_resp.json().get("messages", [])
                        history = [{"role": m["role"], "content": m["content"]} for m in msgs]
                    print(f"\n--- 已切换到会话 [{session_id}] ({len(history)} 条消息) ---\n")
                else:
                    print(f"\033[91m[错误] 切换失败: {resp.text}\033[0m")
            except requests.RequestException as e:
                print(f"\033[91m[错误] {e}\033[0m")
            continue

        if user_input == "/history":
            if not history:
                print("(空对话)")
            for i, msg in enumerate(history):
                role = msg["role"]
                tag = "\033[92mYou\033[0m" if role == "user" else "\033[96mBot\033[0m"
                print(f"  [{i}] {tag}: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
            print()
            continue

        if user_input.startswith("/edit "):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip().isdigit():
                print("用法: /edit <消息索引>  (输入 /history 查看索引)")
                continue
            msg_idx = int(parts[1].strip())
            if msg_idx < 0 or msg_idx >= len(history):
                print(f"\033[91m[错误] 索引超出范围 (0-{len(history)-1})\033[0m")
                continue
            print(f"  原内容: {history[msg_idx]['content']}")
            try:
                new_content = input("  新内容: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n取消编辑")
                continue
            if not new_content:
                print("取消编辑")
                continue

            try:
                payload = {
                    "session_id": session_id,
                    "message_index": msg_idx,
                    "new_content": new_content,
                    "regenerate": True,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "stream": True,
                }
                resp = requests.post(f"{base_url}/v1/edit", json=payload, stream=True, timeout=600)
                resp.raise_for_status()

                # 截断本地历史
                history = history[:msg_idx]
                history.append({"role": "user", "content": new_content})

                print("\033[96mBot:\033[0m ", end="", flush=True)
                assistant_text = _read_sse_stream(resp)
                print()
                history.append({"role": "assistant", "content": assistant_text})
                print()
            except requests.RequestException as e:
                print(f"\033[91m[错误] {e}\033[0m")
            continue

        if user_input == "/regen":
            if not history or history[-1]["role"] != "assistant":
                print("\033[93m[提示] 没有可重新生成的回复\033[0m")
                continue

            try:
                payload = {
                    "session_id": session_id,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "stream": True,
                }
                resp = requests.post(f"{base_url}/v1/regenerate", json=payload, stream=True, timeout=600)
                resp.raise_for_status()

                history.pop()  # 移除旧回复

                print("\033[96mBot (重新生成):\033[0m ", end="", flush=True)
                assistant_text = _read_sse_stream(resp)
                print()
                history.append({"role": "assistant", "content": assistant_text})
                print()
            except requests.RequestException as e:
                print(f"\033[91m[错误] {e}\033[0m")
            continue

        if user_input.startswith("/delete "):
            target_id = user_input[8:].strip()
            if not target_id:
                print("用法: /delete <session_id>")
                continue
            try:
                resp = requests.delete(f"{base_url}/v1/sessions/{target_id}", timeout=10)
                if resp.ok:
                    print(f"已删除会话 [{target_id}]")
                    if target_id == session_id:
                        session_id = None
                        history.clear()
                else:
                    print(f"\033[91m[错误] {resp.text}\033[0m")
            except requests.RequestException as e:
                print(f"\033[91m[错误] {e}\033[0m")
            continue

        if user_input.startswith("/"):
            print(f"\033[93m[提示] 未知命令: {user_input}. 输入 /help 查看命令列表\033[0m")
            continue

        # ── 发送请求 ──
        history.append({"role": "user", "content": user_input})

        payload = {
            "messages": history,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "stream": True,
            "session_id": session_id,
        }

        try:
            resp = requests.post(api_url, json=payload, stream=True, timeout=600)
            resp.raise_for_status()
        except requests.ConnectionError:
            print("\033[91m[错误] 无法连接到服务器，请确认服务器已启动\033[0m")
            history.pop()
            continue
        except requests.RequestException as e:
            print(f"\033[91m[错误] {e}\033[0m")
            history.pop()
            continue

        # ── 流式读取 ──
        print("\033[96mBot:\033[0m ", end="", flush=True)
        assistant_text = _read_sse_stream(resp)
        print()
        history.append({"role": "assistant", "content": assistant_text})

        # 更新 session_id (如果服务端返回了)
        # session_id 已在请求中传递, 无需从响应中获取
        print()


def _read_sse_stream(resp) -> str:
    """读取 SSE 流式响应, 实时打印, 返回完整文本."""
    assistant_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                print(content, end="", flush=True)
                assistant_text += content
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return assistant_text


if __name__ == "__main__":
    main()
