"""
Configuration module - manage API keys and settings
"""

import os
import json
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".deepseek_search"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Available models (DeepSeek V4)
AVAILABLE_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
DEFAULT_MODEL = "deepseek-v4-flash"

# Available reasoning effort levels
AVAILABLE_EFFORTS = ["high", "max"]
DEFAULT_EFFORT = "high"


def load_config() -> dict:
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_config(config: dict) -> None:
    """Save configuration to file"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_api_key() -> Optional[str]:
    """Get DeepSeek API Key, priority: env var > config file"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        return api_key
    config = load_config()
    return config.get("api_key")


def set_api_key(api_key: str) -> None:
    """Set DeepSeek API Key and save to config file"""
    config = load_config()
    config["api_key"] = api_key
    save_config(config)
    print("[OK] API Key saved to config file")


def get_search_provider() -> str:
    """Get search engine configuration"""
    config = load_config()
    return config.get("search_provider", "duckduckgo")


def get_max_search_results() -> int:
    """Get max search results count"""
    config = load_config()
    return config.get("max_search_results", 5)


def get_model() -> str:
    """Get model name"""
    cfg = load_config()
    model = cfg.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    return model


def set_model(model_name: str) -> None:
    """Set model name and save to config"""
    cfg = load_config()
    cfg["model"] = model_name
    save_config(cfg)


def is_thinking_mode() -> bool:
    """Check if thinking mode is enabled"""
    cfg = load_config()
    return cfg.get("thinking_mode", False)


def set_thinking_mode(enabled: bool) -> None:
    """Enable or disable thinking mode"""
    cfg = load_config()
    cfg["thinking_mode"] = enabled
    save_config(cfg)


def get_reasoning_effort() -> str:
    """Get reasoning effort level (high / max)"""
    cfg = load_config()
    effort = cfg.get("reasoning_effort", DEFAULT_EFFORT)
    if effort not in AVAILABLE_EFFORTS:
        effort = DEFAULT_EFFORT
    return effort


def set_reasoning_effort(effort: str) -> None:
    """Set reasoning effort level"""
    cfg = load_config()
    cfg["reasoning_effort"] = effort
    save_config(cfg)


def get_max_tokens() -> int:
    """Get max tokens for model response"""
    cfg = load_config()
    return cfg.get("max_tokens", 8192)


def get_system_prompt() -> str:
    """Get system prompt"""
    return """You are a smart AI assistant with internet search capability.
When users ask about real-time information, latest news, current events,
or specific facts you are not sure about, you should use the web_search tool.

Guidelines for using web_search:
1. Must search when questions involve current time, latest data, news events
2. Search for more context when you need deeper understanding of a topic
3. After searching, combine search results with your knowledge for comprehensive answers
4. Cite information sources in your answers
5. CRITICAL: After you have search results, you MUST provide an answer to the user.
   Do NOT keep searching repeatedly. Use what you have and respond.
6. If search results lack the requested information, simply respond with exactly:
   "查询不到相关信息" and do NOT make up information.
7. Keep your answers concise and focused on the question.

Current time: {current_time}
"""
