"""
Search module - web search and content fetching
"""

import re
import requests
from typing import Optional

from . import config


def search_web(query: str, max_results: int = None) -> list[dict]:
    """
    Search the web using multiple engines

    Args:
        query: search keywords
        max_results: max results count, defaults to config
    """
    if max_results is None:
        max_results = config.get_max_search_results()

    # Bing first - best availability in China
    results = _search_bing(query, max_results)
    if results:
        return results

    # Fallback: ddgs
    results = _search_ddgs(query, max_results)
    if results:
        return results

    # Fallback: DuckDuckGo Lite
    results = _search_ddg_lite(query, max_results)
    return results


def _search_bing(query: str, max_results: int) -> list[dict]:
    """Search Bing via HTML parsing"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    try:
        url = "https://www.bing.com/search"
        params = {"q": query, "count": max_results, "mkt": "zh-CN"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()

        html = resp.text
        results = []

        # Strategy 1: <li class="b_algo"> blocks
        blocks = re.findall(
            r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>',
            html,
            re.DOTALL | re.IGNORECASE,
        )

        for block in blocks:
            title_match = re.search(
                r'<h2[^>]*>.*?<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?</h2>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if not title_match:
                continue

            url_result = title_match.group(1)
            title = re.sub(r"<[^>]+>", "", title_match.group(2)).strip()

            snippet = ""
            snippet_match = re.search(
                r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>(.*?)</p>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
            else:
                snippet_match = re.search(
                    r'<p[^>]*>(.*?)</p>', block, re.DOTALL | re.IGNORECASE
                )
                if snippet_match:
                    snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()

            results.append(
                {"title": title or "No title", "url": url_result, "snippet": snippet}
            )
            if len(results) >= max_results:
                break

        if results:
            return results

        # Strategy 2: any <h2><a href="..."> in page
        all_links = re.findall(
            r'<h2[^>]*>.*?<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?</h2>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        for url_result, raw_title in all_links:
            if any(r["url"] == url_result for r in results):
                continue
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            results.append({"title": title or "No title", "url": url_result, "snippet": ""})
            if len(results) >= max_results:
                break

        return results

    except Exception:
        return []


def _search_ddgs(query: str, max_results: int) -> list[dict]:
    """Search via DuckDuckGo SDK (ddgs)"""
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                )
                if len(results) >= max_results:
                    break
        return results
    except Exception:
        return []


def _search_ddg_lite(query: str, max_results: int) -> list[dict]:
    """Search via DuckDuckGo Lite API"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        url = "https://lite.duckduckgo.com/lite/"
        data = {"q": query}
        resp = requests.post(url, data=data, headers=headers, timeout=15)
        resp.raise_for_status()

        from html.parser import HTMLParser

        class LiteParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.in_link = False
                self.current = {}

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result-link" in attrs_dict.get("class", ""):
                    self.in_link = True
                    self.current = {"url": attrs_dict.get("href", ""), "title": ""}
                if tag == "td" and "result-snippet" in attrs_dict.get("class", ""):
                    self.current["snippet"] = ""

            def handle_data(self, data):
                if self.in_link:
                    self.current["title"] += data.strip()

            def handle_endtag(self, tag):
                if tag == "a" and self.in_link and self.current.get("title"):
                    self.results.append(dict(self.current))
                    self.current = {}
                    self.in_link = False

        parser = LiteParser()
        parser.feed(resp.text)
        return parser.results[:max_results]

    except Exception:
        return []


def fetch_page_content(url: str, max_length: int = 3000) -> Optional[str]:
    """
    Fetch and extract text content from a webpage
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        if resp.encoding and resp.encoding.lower() != "utf-8":
            try:
                resp.encoding = resp.apparent_encoding or "utf-8"
            except Exception:
                resp.encoding = "utf-8"

        text = resp.text

        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    except Exception:
        return None


def format_search_results(results: list[dict]) -> str:
    """Format search results as text"""
    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] {r.get('title', 'No title')}")
        formatted.append(f"     URL: {r.get('url', 'No URL')}")
        snippet = r.get("snippet", "")
        if snippet:
            formatted.append(f"     Snippet: {snippet}")
        formatted.append("")

    return "\n".join(formatted)
