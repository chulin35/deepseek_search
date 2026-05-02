#!/usr/bin/env python3
"""
DeepSeek Web Search Assistant - Main Entry Point (V4)

A CLI tool that integrates DeepSeek V4 API with web search capability.
Both deepseek-v4-pro and deepseek-v4-flash support:
  - Normal mode: auto web search via function calling
  - Thinking mode: displays reasoning process + web search support
  - Reasoning effort: high / max
"""

import sys
import signal

from . import config
from .chat_client import Conversation


COMMANDS = {
    "/help": "Show help message",
    "/new": "Start a new conversation (clear context)",
    "/setkey": "Set DeepSeek API Key",
    "/model": "Show or switch model (deepseek-v4-pro / deepseek-v4-flash)",
    "/think": "Toggle thinking mode ON/OFF",
    "/effort": "Set reasoning effort (high / max)",
    "/exit": "Exit program",
}


def print_banner():
    """Print startup banner"""
    model_name = config.get_model()
    thinking = "ON" if config.is_thinking_mode() else "OFF"
    effort = config.get_reasoning_effort()
    banner = f"""
============================================
    DeepSeek V4 Web Search Assistant

    Model : {model_name}
    Think : {thinking}  (effort: {effort})

    /help   - Show help
    /model  - Switch model
    /think  - Toggle thinking mode
    /effort - Set thinking intensity
    /exit   - Exit
============================================
    """
    print(banner)


def print_help():
    """Print help information"""
    print("\n[Help] Available commands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:15s} - {desc}")
    print()
    print("Models (DeepSeek V4):")
    print("  deepseek-v4-pro  - Higher quality (slower, more expensive)")
    print("  deepseek-v4-flash - Faster & cheaper (default)")
    print()
    print("Thinking Mode:")
    print("  /think           - Toggle thinking mode on/off")
    print("  /effort high     - Standard reasoning intensity (default)")
    print("  /effort max      - Maximum reasoning intensity (for complex Agent tasks)")
    print()
    print("Tips:")
    print("  * Just type your question to start")
    print("  * AI will automatically search web when needed")
    print("  * Thinking mode works with web search (both can be used together)")
    print("  * For complex Agent scenarios, enable thinking + effort max")
    print("  * Conversation history is preserved within a session")
    print()


def print_status():
    """Print current configuration status"""
    model = config.get_model()
    thinking = "ON" if config.is_thinking_mode() else "OFF"
    effort = config.get_reasoning_effort()
    print(f"  Model : {model}")
    print(f"  Think : {thinking}")
    if thinking == "ON":
        print(f"  Effort: {effort}")
    print()


def handle_command(cmd: str, conv: Conversation) -> bool:
    """
    Handle internal commands

    Args:
        cmd: the command string
        conv: the current conversation (may be cleared)

    Returns:
        True if should exit
    """
    cmd = cmd.strip().lower()

    if cmd == "/exit":
        print("[Bye] Goodbye!")
        return True

    elif cmd == "/help":
        print_help()

    elif cmd == "/new":
        conv.clear()
        print("[OK] New conversation started. Previous context cleared.")

    elif cmd == "/setkey" or cmd.startswith("/setkey "):
        parts = cmd.split(maxsplit=1)
        if len(parts) == 2:
            config.set_api_key(parts[1])
        else:
            print("Provide API Key: /setkey sk-your-key-here")
        key = config.get_api_key()
        if key:
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else key[:8] + "..."
            print(f"Current API Key: {masked}")

    elif cmd == "/model":
        current = config.get_model()
        print(f"Current model: {current}")
        print(f"Available: {', '.join(config.AVAILABLE_MODELS)}")
        print("Switch: /model deepseek-v4-pro")
        print("   or:  /model deepseek-v4-flash")

    elif cmd.startswith("/model "):
        model_name = cmd.split(maxsplit=1)[1].strip()
        if model_name in config.AVAILABLE_MODELS:
            config.set_model(model_name)
            conv.clear()  # model change requires fresh context
            print(f"[OK] Switched to model: {model_name}")
            print("[OK] Conversation cleared (model changed).")
        else:
            print(f"[Error] Unknown model: {model_name}")
            print(f"  Available: {', '.join(config.AVAILABLE_MODELS)}")

    elif cmd == "/think":
        new_state = not config.is_thinking_mode()
        config.set_thinking_mode(new_state)
        conv.clear()  # thinking mode change may affect conversation behavior
        state_str = "ON" if new_state else "OFF"
        print(f"[OK] Thinking mode turned {state_str}")
        if new_state:
            print(f"    Reasoning effort: {config.get_reasoning_effort()}")
        print("[OK] Conversation cleared (thinking mode changed).")

    elif cmd == "/effort":
        current = config.get_reasoning_effort()
        print(f"Current reasoning effort: {current}")
        print(f"Available: {', '.join(config.AVAILABLE_EFFORTS)}")
        print("Set: /effort max")
        print(" or: /effort high")

    elif cmd.startswith("/effort "):
        effort = cmd.split(maxsplit=1)[1].strip()
        if effort in config.AVAILABLE_EFFORTS:
            config.set_reasoning_effort(effort)
            print(f"[OK] Reasoning effort set to: {effort}")
        else:
            print(f"[Error] Unknown effort: {effort}")
            print(f"  Available: {', '.join(config.AVAILABLE_EFFORTS)}")

    elif cmd == "/status":
        print("\n[Status] Current configuration:")
        print_status()

    else:
        print(f"Unknown command: {cmd}")
        print("Type /help for available commands")

    return False


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\n\n[Bye] Goodbye!")
    sys.exit(0)


def interactive_mode():
    """Interactive chat mode with conversation memory"""
    signal.signal(signal.SIGINT, signal_handler)

    if not config.get_api_key():
        print("[Warning] No API Key detected!")
        key = input("Enter your DeepSeek API Key (or /skip): ").strip()
        if key and key != "/skip":
            config.set_api_key(key)
            print("[OK] API Key saved!\n")
        else:
            print("[Warning] Skipped. Use /setkey later\n")

    conv = Conversation()
    print_banner()

    while True:
        try:
            user_input = input("\n[Input] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_command(user_input, conv):
                break
            continue

        print("\n" + "=" * 50)
        print(f"[Q] {user_input}")
        print("=" * 50)

        try:
            response = conv.ask(user_input)
            if response:
                print("\n" + "=" * 50)
                print("[Done]")
                print("=" * 50)
            else:
                print("[Warning] No response")
        except ValueError as e:
            print(f"\n[Error] {e}")
        except Exception as e:
            print(f"\n[Error] {e}")
            print("   Check network and API Key.")


def quick_mode():
    """Quick query mode: process command line args directly"""
    if len(sys.argv) < 2:
        print("Usage: deepseek-search <your question>")
        print("   or: deepseek-search (interactive mode)")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    if not config.get_api_key():
        print("[Error] No API Key detected!")
        print("Set it via: set DEEPSEEK_API_KEY=your_key")
        sys.exit(1)

    try:
        conv = Conversation()
        response = conv.ask(query)
        if not response:
            print("[Warning] No response")
    except ValueError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "/help":
            print_banner()
            print_help()
        elif arg == "/model":
            current = config.get_model()
            print(f"Current model: {current}")
            print(f"Available: {', '.join(config.AVAILABLE_MODELS)}")
        elif arg == "/think":
            new_state = not config.is_thinking_mode()
            config.set_thinking_mode(new_state)
            state_str = "ON" if new_state else "OFF"
            print(f"Thinking mode turned {state_str}")
        elif arg.startswith("/setkey "):
            parts = arg.split(maxsplit=1)
            if len(parts) == 2:
                config.set_api_key(parts[1])
        elif arg.startswith("-"):
            interactive_mode()
        else:
            quick_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
