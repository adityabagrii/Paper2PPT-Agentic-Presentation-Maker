"""Lightweight web search utility for query grounding."""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current: Dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "class" in attrs and "result__a" in attrs.get("class", ""):
            self._in_title = True
            self._current = {"title": "", "url": attrs.get("href", ""), "snippet": ""}
        if tag == "a" and "class" in attrs and "result__snippet" in attrs.get("class", ""):
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
        if tag == "a" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_title:
            self._current["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


def _normalize_ddg_url(url: str) -> str:
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return url


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query.strip():
        return []

    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=20)
    r.raise_for_status()

    parser = _DDGParser()
    parser.feed(r.text)

    cleaned = []
    seen = set()
    for res in parser.results:
        url = _normalize_ddg_url(res.get("url", "").strip())
        title = html.unescape(res.get("title", "")).strip()
        snippet = html.unescape(res.get("snippet", "")).strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if not url or url in seen:
            continue
        seen.add(url)
        cleaned.append({"title": title, "url": url, "snippet": snippet})
        if len(cleaned) >= max_results:
            break
    return cleaned
