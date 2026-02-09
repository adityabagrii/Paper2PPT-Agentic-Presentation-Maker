"""Lightweight web search utility for query grounding."""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
import random
import time
import json
import os
import logging
from typing import Dict, List, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import requests


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        """Initialize.
        
        Returns:
            None:
        """
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current: Dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        """Function handle starttag.
        
        Args:
            tag (Any):
            attrs (Any):
        
        Returns:
            Any:
        """
        attrs = dict(attrs)
        if tag == "a" and "class" in attrs and "result__a" in attrs.get("class", ""):
            self._in_title = True
            self._current = {"title": "", "url": attrs.get("href", ""), "snippet": ""}
        if tag == "a" and "class" in attrs and "result__snippet" in attrs.get("class", ""):
            self._in_snippet = True

    def handle_endtag(self, tag):
        """Function handle endtag.
        
        Args:
            tag (Any):
        
        Returns:
            Any:
        """
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
        if tag == "a" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        """Function handle data.
        
        Args:
            data (Any):
        
        Returns:
            Any:
        """
        if self._in_title:
            self._current["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


def _normalize_ddg_url(url: str) -> str:
    """Function normalize ddg url.
    
    Args:
        url (str):
    
    Returns:
        str:
    """
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return url


def _parse_ddg_html(text: str) -> List[Dict[str, str]]:
    parser = _DDGParser()
    parser.feed(text)
    return parser.results


def _parse_ddg_lite(text: str) -> List[Dict[str, str]]:
    # Lite results are simple tables with <a href> and nearby snippets.
    results: List[Dict[str, str]] = []
    for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.IGNORECASE | re.DOTALL):
        url = html.unescape(m.group(1)).strip()
        title = re.sub(r"<[^>]+>", "", m.group(2))
        title = html.unescape(title).strip()
        if not url or not title:
            continue
        # DDG lite uses relative /l/?uddg=... links; skip other relative links.
        if url.startswith("/l/"):
            url = "https://duckduckgo.com" + url
        elif url.startswith("/"):
            continue
        results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= 30:
            break
    # Try to attach snippets if present in neighboring <td> blocks.
    if results:
        snippets = re.findall(r"<td[^>]*>([^<].*?)</td>", text, re.IGNORECASE | re.DOTALL)
        if snippets:
            for i, s in enumerate(snippets):
                if i >= len(results):
                    break
                snippet = re.sub(r"<[^>]+>", "", s)
                snippet = html.unescape(re.sub(r"\s+", " ", snippet)).strip()
                if snippet and len(snippet) > 20:
                    results[i]["snippet"] = snippet
    return results


def _search_ddg(query: str, max_results: int) -> List[Dict[str, str]]:
    params = {"q": query}
    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    endpoints = [
        "https://duckduckgo.com/html/",
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    ]
    results: List[Dict[str, str]] = []
    last_err = None
    session = requests.Session()
    for attempt in range(3):
        for url in endpoints:
            try:
                headers = {
                    "User-Agent": random.choice(user_agents),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Referer": "https://duckduckgo.com/",
                    "Upgrade-Insecure-Requests": "1",
                }
                time.sleep(0.6 + random.random() * 0.8)
                r = session.get(url, params=params, headers=headers, timeout=20)
                r.raise_for_status()
                if r.status_code == 202:
                    results = []
                    continue
                text = r.text or ""
                if url.endswith("/html/"):
                    results = _parse_ddg_html(text)
                else:
                    results = _parse_ddg_lite(text)
                if results:
                    break
            except Exception as e:
                last_err = e
                results = []
        if results:
            break
        time.sleep(1.5 + attempt)
    if not results and last_err:
        raise last_err
    # Filter out DDG self-links if they slipped through.
    cleaned = []
    for res in results:
        url = res.get("url", "").strip()
        if "duckduckgo.com" in url:
            continue
        cleaned.append(res)
        if len(cleaned) >= max_results:
            break
    return cleaned


def _cache_dir() -> str:
    base = os.environ.get("RESEARCHOS_CACHE_DIR", "")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".researchos_cache")
    os.makedirs(base, exist_ok=True)
    return base


def _cache_path(key: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\\-]", "_", key)[:200]
    return os.path.join(_cache_dir(), f"{safe}.json")


def _cache_get(key: str, ttl_sec: int = 6 * 3600):
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if (time.time() - obj.get("_cached_at", 0)) > ttl_sec:
            return None
        return obj.get("data")
    except Exception:
        return None


def _cache_set(key: str, data) -> None:
    p = _cache_path(key)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"_cached_at": time.time(), "data": data}, f, ensure_ascii=False)
    except Exception:
        pass


def _request_json_with_backoff(url: str, params: dict | None = None, ttl_sec: int = 6 * 3600) -> dict:
    key = f"{url}::{params}"
    cached = _cache_get(key, ttl_sec=ttl_sec)
    if cached is not None:
        return cached
    last_err = None
    max_sleep = 30.0
    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(min(max_sleep, 5.0 * attempt))
                continue
            r.raise_for_status()
            data = r.json()
            _cache_set(key, data)
            return data
        except Exception as e:
            last_err = e
            time.sleep(min(max_sleep, 4.0 * attempt))
    if last_err:
        raise last_err
    return {}


def _search_openalex(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://api.openalex.org/works"
    params = {"search": query, "per-page": max_results}
    data = _request_json_with_backoff(url, params=params)
    results = []
    for item in data.get("results", [])[:max_results]:
        title = item.get("display_name", "") or ""
        landing = ""
        primary = item.get("primary_location") or {}
        landing = primary.get("landing_page_url") or ""
        oa = item.get("open_access") or {}
        oa_url = oa.get("oa_url") or ""
        pdf_url = ""
        for loc in item.get("locations", []) or []:
            pdf_url = loc.get("pdf_url") or ""
            if pdf_url:
                break
        openalex_id = item.get("id", "")
        url_out = pdf_url or oa_url or landing or openalex_id
        snippet = ""
        venue = (item.get("host_venue") or {}).get("display_name")
        year = item.get("publication_year")
        if venue or year:
            snippet = " ".join([str(x) for x in [venue, year] if x])
        results.append(
            {
                "title": title,
                "url": url_out,
                "snippet": snippet,
                "source": "openalex",
                "year": str(year) if year else "",
                "venue": venue or "",
                "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
                "pdf_url": pdf_url or "",
            }
        )
    return results


def _search_semanticscholar(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,url,abstract,venue,year,openAccessPdf,externalIds",
    }
    data = _request_json_with_backoff(url, params=params)
    results = []
    for item in data.get("data", [])[:max_results]:
        title = item.get("title", "") or ""
        url_out = item.get("url") or ""
        oap = item.get("openAccessPdf") or {}
        if oap.get("url"):
            url_out = oap.get("url")
        snippet = item.get("abstract") or ""
        if not snippet:
            venue = item.get("venue")
            year = item.get("year")
            if venue or year:
                snippet = " ".join([str(x) for x in [venue, year] if x])
        results.append(
            {
                "title": title,
                "url": url_out,
                "snippet": snippet,
                "source": "semantic_scholar",
                "year": str(item.get("year") or ""),
                "venue": item.get("venue") or "",
                "doi": (item.get("externalIds") or {}).get("DOI", ""),
                "arxiv_id": (item.get("externalIds") or {}).get("ArXiv", ""),
                "pdf_url": oap.get("url", "") if isinstance(oap, dict) else "",
            }
        )
    return results


def _search_crossref(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": max_results}
    data = _request_json_with_backoff(url, params=params)
    results = []
    for item in data.get("message", {}).get("items", [])[:max_results]:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        # Prefer direct PDF links if present.
        url_out = ""
        for link in item.get("link", []) or []:
            ct = (link.get("content-type") or "").lower()
            if "pdf" in ct:
                url_out = link.get("URL", "")
                break
        container = (item.get("container-title") or [""])[0]
        year = ""
        for k in ("published-print", "published-online", "created"):
            if k in item and "date-parts" in item[k]:
                year = str(item[k]["date-parts"][0][0])
                break
        snippet = " ".join([str(x) for x in [container, year] if x])
        if url_out:
            results.append(
                {
                    "title": title,
                    "url": url_out,
                    "snippet": snippet,
                    "source": "crossref",
                    "year": year,
                    "venue": container,
                    "doi": item.get("DOI", ""),
                    "pdf_url": url_out,
                }
            )
    return results


def _search_openreview(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://api.openreview.net/notes"
    params = {"search": query, "limit": max_results}
    data = _request_json_with_backoff(url, params=params)
    results = []
    for note in data.get("notes", [])[:max_results]:
        content = note.get("content") or {}
        title = content.get("title") or ""
        forum = note.get("forum") or note.get("id") or ""
        url_out = f"https://openreview.net/forum?id={forum}" if forum else ""
        snippet = content.get("abstract") or ""
        results.append(
            {
                "title": title,
                "url": url_out,
                "snippet": snippet,
                "source": "openreview",
                "pdf_url": f"https://openreview.net/pdf?id={forum}" if forum else "",
            }
        )
    return results


def _search_arxiv(query: str, max_results: int) -> List[Dict[str, str]]:
    import arxiv

    results = []
    search = arxiv.Search(query=query, max_results=max_results)
    for r in search.results():
        results.append(
            {
                "title": r.title,
                "url": r.entry_id,
                "snippet": (r.summary or "")[:400],
                "source": "arxiv",
                "year": str(r.published.year) if getattr(r, "published", None) else "",
                "arxiv_id": r.get_short_id(),
                "pdf_url": r.pdf_url if getattr(r, "pdf_url", None) else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def search_web(query: str, max_results: int = 5, providers: Iterable[str] | None = None) -> List[Dict[str, str]]:
    """Function search web.
    
    Args:
        query (str):
        max_results (int):
    
    Returns:
        List[Dict[str, str]]:
    """
    if not query.strip():
        return []

    if providers is None:
        providers = ["ddg"]
    providers = [p.lower() for p in providers]
    results: List[Dict[str, str]] = []
    seen_urls = set()
    for provider in providers:
        try:
            # if provider == "ddg":
            #     batch = _search_ddg(query, max_results)
            # elif provider == "openalex":
            #     batch = _search_openalex(query, max_results)
            if provider == "semanticscholar":
                batch = _search_semanticscholar(query, max_results)
            elif provider == "crossref":
                batch = _search_crossref(query, max_results)
            elif provider == "openreview":
                batch = _search_openreview(query, max_results)
            elif provider == "arxiv":
                batch = _search_arxiv(query, max_results)
            else:
                batch = []
        except Exception:
            batch = []
        for res in batch:
            url = _normalize_ddg_url(res.get("url", "").strip())
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            res["url"] = url
            results.append(res)
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    cleaned = []
    seen = set()
    for res in results:
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


# Research-focused multi-source search (based on check.ipynb methodology).
ALLOWED_DOMAINS = [
    "arxiv.org",
    "openreview.net",
    "semanticscholar.org",
    "openalex.org",
    "doi.org",
    "crossref.org",
    "openaccess.thecvf.com",
    "cvf.org",
    "cvpr.thecvf.com",
    "iccv.thecvf.com",
    "eccv.ecva.net",
    "proceedings.neurips.cc",
    "neurips.cc",
    "proceedings.mlr.press",
    "icml.cc",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "jmlr.org",
    "github.com",
]


def _host(u: str) -> str:
    try:
        return (urlparse(u).netloc or "").lower()
    except Exception:
        return ""


def _is_allowed(u: str) -> bool:
    h = _host(u)
    if not h:
        return False
    for d in ALLOWED_DOMAINS:
        if h == d or h.endswith("." + d):
            return True
    return False


def _dedupe(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("doi") or "").lower().strip()
        if not key:
            key = (it.get("arxiv_id") or "").lower().strip()
        if not key:
            key = _normalize_ddg_url(it.get("url", "")).lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _score(it: Dict[str, str]) -> float:
    s = 0.0
    src = (it.get("source") or "").lower()
    if src in ("semantic_scholar", "openalex", "arxiv", "crossref", "openreview"):
        s += 60
    if it.get("doi"):
        s += 25
    if it.get("arxiv_id"):
        s += 20
    if it.get("pdf_url"):
        s += 12
    if it.get("venue"):
        v = it.get("venue", "").lower()
        if any(k in v for k in ["cvpr", "iccv", "eccv", "neurips", "icml", "iclr"]):
            s += 20
        else:
            s += 6
    y = it.get("year", "")
    m = re.search(r"(19|20)\\d{2}", y or "")
    if m:
        year = int(m.group(0))
        s += max(0, min(10, (year - 2015) * 0.9))
    return s


def search_research_focused(query: str, max_results: int = 20) -> List[Dict[str, str]]:
    logger = logging.getLogger("researchos")
    items: List[Dict[str, str]] = []
    per_limit = max(1, max_results // 2)
    # Call providers sequentially and log counts.
    providers = [
        ("arxiv", _search_arxiv),
        ("semantic_scholar", _search_semanticscholar),
        ("crossref", _search_crossref),
        ("openreview", _search_openreview),
    ]
    for name, fn in providers:
        try:
            batch = fn(query, per_limit)
        except Exception:
            logger.exception("Provider failed: %s", name)
            batch = []
        items += batch
        logger.info("Provider %s results: %s", name, len(batch))
        if batch:
            logger.info("Provider %s titles:", name)
            for i, it in enumerate(batch, 1):
                logger.info("  %s. %s", i, it.get("title", "").strip())

    # Filter to reputed domains where possible.
    filtered = []
    for it in items:
        url = it.get("url", "")
        if not url:
            continue
        if _is_allowed(url):
            filtered.append(it)
    deduped = _dedupe(filtered)
    for it in deduped:
        it["_score"] = _score(it)
    ranked = sorted(deduped, key=lambda x: x.get("_score", 0), reverse=True)
    return ranked[:max_results]
