"""
DeepSeek API Client - handles communication with DeepSeek API (V4)
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

from openai import OpenAI

from . import config
from .search_engine import search_web, fetch_page_content, format_search_results


# ============================================================
# stdout encoding fix: replace unencodable chars (emoji etc.)
# for Windows terminal (cmd.exe) compatibility
# ============================================================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# ============================================================
# Tool definitions for function calling
# ============================================================
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for latest information. Use this when users ask about real-time info, news, current events, or facts you are unsure about.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, should be clear and specific",
                }
            },
            "required": ["query"],
        },
    },
}

FETCH_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": "Fetch webpage text content for deeper reading. Use when search result snippets are not detailed enough.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Webpage URL",
                }
            },
            "required": ["url"],
        },
    },
}

TOOLS = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]


# ============================================================
# Conversation
# ============================================================
class Conversation:
    """Manages a conversation with context memory (DeepSeek V4)"""

    def __init__(self):
        self.messages = []
        self._init_system()

    def _init_system(self):
        """Initialize conversation with system prompt"""
        system_prompt = config.get_system_prompt().format(
            current_time=self._get_current_time_str()
        )
        self.messages = [{"role": "system", "content": system_prompt}]

    def _get_current_time_str(self) -> str:
        """Get current time string (Asia/Shanghai)"""
        utc_now = datetime.now(timezone.utc)
        shanghai_now = utc_now.astimezone(timezone(timedelta(hours=8)))
        return shanghai_now.strftime("%Y-%m-%d %H:%M:%S (Asia/Shanghai, UTC+8)")

    def _update_system_time(self):
        """Update the current time in the system prompt"""
        now_str = self._get_current_time_str()
        for msg in self.messages:
            if msg["role"] == "system":
                msg["content"] = config.get_system_prompt().format(
                    current_time=now_str
                )
                break

    def clear(self):
        """Start a new conversation"""
        self._init_system()

    def _get_client(self) -> OpenAI:
        """Create DeepSeek API client"""
        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                "DeepSeek API Key not set!\n"
                "Set it via:\n"
                "  1. Environment variable: set DEEPSEEK_API_KEY=your_key\n"
                "  2. In the program: /setkey your_key"
            )
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def _build_api_kwargs(
        self,
        messages: list[dict],
        tools: list = None,
        tool_choice: str = "auto",
        stream: bool = True,
    ) -> dict:
        """
        Build keyword arguments for the DeepSeek API call.

        DeepSeek V4 supports thinking mode on all models via reasoning_effort param.
        When thinking_mode is enabled, reasoning_effort is added to the request.

        NOTE: DeepSeek V4 always returns reasoning_content in stream responses,
        regardless of thinking mode setting. The reasoning_content MUST be
        preserved and passed back on subsequent API calls.
        """
        thinking_mode = config.is_thinking_mode()

        # NO sanitization of reasoning_content needed:
        # DeepSeek V4 ALWAYS returns reasoning_content and expects it back
        # on subsequent API calls, regardless of thinking_mode setting.
        # Stripping it would cause 400 errors on round 2+ (tool calls).
        sanitized = list(messages)

        kwargs = {
            "model": config.get_model(),
            "messages": sanitized,
            "stream": stream,
            "max_tokens": config.get_max_tokens(),
        }

        # Add thinking / reasoning_effort if enabled
        if thinking_mode:
            kwargs["reasoning_effort"] = config.get_reasoning_effort()

        # Add tools if provided (V4 supports function calling even in thinking mode)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        return kwargs

    def _call_deepseek(
        self,
        messages: list[dict],
        tools: list = None,
        tool_choice: str = "auto",
        stream: bool = True,
        silent: bool = False,
        show_reasoning: bool = False,
    ) -> dict:
        """Call DeepSeek API (V4)

        Args:
            messages: conversation messages
            tools: function calling tools
            tool_choice: tool selection strategy
            stream: whether to stream response
            silent: if True, suppress all printing
            show_reasoning: if True, always print reasoning_content even when silent
                           (used for thinking mode during search rounds)
        """
        client = self._get_client()
        kwargs = self._build_api_kwargs(messages, tools, tool_choice, stream)
        response = client.chat.completions.create(**kwargs)

        if stream:
            return self._handle_stream_response(response, silent=silent, show_reasoning=show_reasoning)
        else:
            text = response.choices[0].message.content or ""
            return {"content": text, "tool_calls": []}

    def _handle_stream_response(self, stream, silent=False, show_reasoning=False):
        """
        Handle streaming response from DeepSeek V4.

        In thinking mode:
          - reasoning_content is streamed first (shown in dim/grey style)
          - Then the final content is streamed
        In normal mode:
          - Only content is streamed

        Both modes support tool_calls.

        IMPORTANT: When reasoning_content is present, it MUST be passed back
        to subsequent API calls. See ask() for how this is handled.

        Args:
            stream: API stream response
            silent: if True, suppress content printing
            show_reasoning: if True, print reasoning_content even when silent
                           (so user can see the thinking process during search rounds)

        Returns:
            dict with 'content', 'tool_calls', and 'reasoning_content'
        """
        full_content = ""
        full_reasoning = ""
        tool_calls = []
        reasoning_phase = True  # Are we still in the reasoning phase?

        for chunk in stream:
            choices = chunk.choices
            if not choices:
                continue

            delta = choices[0].delta
            if delta is None:
                continue

            # --- reasoning_content (V4 thinking mode) ---
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                full_reasoning += rc
                # Print reasoning_content UNLESS both silent=True AND show_reasoning=False
                if show_reasoning or not silent:
                    # Print reasoning_content in dim/grey ANSI style
                    print(f"\033[2m{rc}\033[0m", end="", flush=True)

            # --- tool_calls ---
            if delta.tool_calls:
                reasoning_phase = False
                for tc in delta.tool_calls:
                    while len(tool_calls) <= tc.index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.id:
                        tool_calls[tc.index]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls[tc.index]["function"]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

            # --- content ---
            if delta.content:
                if reasoning_phase and full_reasoning and (show_reasoning or not silent):
                    # Transition from reasoning to answer: print a separator
                    print()  # blank line to separate thinking from answer
                reasoning_phase = False
                full_content += delta.content
                if not silent:
                    print(delta.content, end="", flush=True)

        if not silent:
            print()
        elif show_reasoning and full_reasoning:
            # End the reasoning output with a newline even in silent+show_reasoning mode
            print()

        # If we only got reasoning (no final content), use reasoning as answer
        if full_reasoning and not full_content:
            full_content = full_reasoning

        # reasoning_content MUST always be returned for round-trip:
        # DeepSeek V4 ALWAYS returns reasoning_content even in non-thinking mode,
        # and REQUIRES it to be passed back on subsequent API calls.
        # If we discard it, round 2+ (tool calls) will get 400 error.
        rc_to_return = full_reasoning if full_reasoning else None

        return {
            "content": full_content,
            "tool_calls": tool_calls,
            "reasoning_content": rc_to_return,
        }

    def _process_tool_calls(self, tool_calls: list) -> list[dict]:
        """Process tool calls silently, return result messages"""
        tool_results = []

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            tool_call_id = tc["id"]

            if func_name == "web_search":
                query = args.get("query", "")
                results = search_web(query)
                formatted = format_search_results(results)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": formatted,
                })

            elif func_name == "fetch_page":
                url = args.get("url", "")
                content = fetch_page_content(url)
                result_text = content if content else "Failed to fetch page content"
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text,
                })

        return tool_results

    def _build_assistant_message(self, content: str, tool_calls: list, reasoning_content: str = None) -> dict:
        """
        Build an assistant message dict, including reasoning_content if present.

        DeepSeek V4 thinking mode requires reasoning_content to be passed back
        on subsequent API calls.
        """
        msg = {"role": "assistant", "content": content or None}

        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                }
                for tc in tool_calls
            ]

        # reasoning_content MUST be passed back for thinking mode
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        return msg

    def ask(self, user_input: str) -> Optional[str]:
        """
        Ask a question with conversation memory (DeepSeek V4).

        Both deepseek-v4-pro and deepseek-v4-flash support:
          - Normal mode: auto-decides web search via function calling
          - Thinking mode: shows reasoning process + final answer, also supports web search

        IMPORTANT: In thinking mode, the API requires that reasoning_content
        from the assistant is passed back in subsequent requests. This is
        handled by storing it in the conversation messages.

        Args:
            user_input: user's question

        Returns:
            Final response text
        """
        self._update_system_time()
        self.messages.append({"role": "user", "content": user_input})

        max_rounds = 5
        searched = False

        try:
            # Show appropriate status
            if config.is_thinking_mode():
                print("思考模式已开启，正在深度思考中...\n", flush=True)
            else:
                print("正在思考中...", flush=True)

            thinking = config.is_thinking_mode()

            for round_idx in range(max_rounds):
                # API call: in thinking mode, show reasoning even during search rounds
                result = self._call_deepseek(
                    self.messages,
                    tools=TOOLS,
                    silent=True,
                    show_reasoning=thinking,
                )
                content = result.get("content", "")
                tool_calls = result.get("tool_calls", [])
                reasoning_content = result.get("reasoning_content")

                # Build assistant message (includes reasoning_content for thinking mode)
                assistant_msg = self._build_assistant_message(
                    content, tool_calls, reasoning_content
                )
                self.messages.append(assistant_msg)

                if not tool_calls:
                    # Model answered directly without search
                    if content:
                        print(content)
                    return content or "[No response generated]"

                # --- Web search needed: show updated status ---
                if not searched:
                    searched = True
                    print("正在搜索中...", flush=True)

                # Process tool calls
                tool_results = self._process_tool_calls(tool_calls)
                self.messages.extend(tool_results)

            # Exhausted rounds - final call without tools
            result = self._call_deepseek(self.messages, tools=None, silent=False)
            final_content = result.get("content", "")
            if final_content:
                assistant_msg = self._build_assistant_message(
                    final_content, [], result.get("reasoning_content")
                )
                self.messages.append(assistant_msg)
            return final_content or "[No response generated after search]"

        except Exception:
            raise
