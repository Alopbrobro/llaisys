#!/usr/bin/env python3
"""
LLAISYS CLI Chat — 命令行聊天界面

连接到运行中的 LLAISYS 服务器，支持流式输出和多轮对话。

用法:
    python3 -m server.chat_cli [--url http://127.0.0.1:8000]

命令:
    /new        开始新对话
    /history    查看对话历史
    /quit       退出
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

    api_url = f"{args.url}/v1/chat/completions"
    history: list[dict] = []

    print("=" * 60)
    print("  LLAISYS Chat  (输入 /new 新对话, /quit 退出)")
    print("=" * 60)
    print()

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
        if user_input == "/new":
            history.clear()
            print("\n--- 新对话已开始 ---\n")
            continue
        if user_input == "/history":
            if not history:
                print("(空对话)")
            for msg in history:
                role = msg["role"]
                tag = "\033[92mYou\033[0m" if role == "user" else "\033[96mBot\033[0m"
                print(f"{tag}: {msg['content']}")
            print()
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

        print()  # 换行
        history.append({"role": "assistant", "content": assistant_text})
        print()


if __name__ == "__main__":
    main()
