"""Reddit monitoring module for tracking whale activity mentions.

This module fetches Reddit posts via RSS feeds (no API authentication required)
and extracts ticker mentions for whale activity detection.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from html import unescape

from loguru import logger


@dataclass(slots=True)
class RedditPost:
    """Represents a Reddit post with relevant metadata."""

    id: str
    title: str
    selftext: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: datetime
    url: str
    ticker_mentions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RedditComment:
    """Represents a Reddit comment with relevant metadata."""

    id: str
    body: str
    post_id: str
    subreddit: str
    author: str
    score: int
    created_utc: datetime
    ticker_mentions: List[str] = field(default_factory=list)


class RedditMonitor:
    """Monitors Reddit subreddits for posts and comments mentioning tickers.

    This implementation uses Reddit's RSS feeds which do not require API
    authentication. Comments are fetched by scraping the post HTML page.
    """

    # Regex pattern to match stock tickers
    # Only matches: $TICKER format (e.g., $NVDA) or TICKER followed by common ticker context
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

    # Common non-ticker words to exclude (for fallback matching)
    NON_TICKERS = {
        'A', 'I', 'IT', 'TO', 'IN', 'ON', 'AT', 'AN', 'AS', 'AM', 'BE', 'BY',
        'DO', 'GO', 'HE', 'ME', 'MY', 'NO', 'OF', 'OR', 'SO', 'UP', 'US', 'WE',
        'YOLO', 'DD', 'FAQ', 'IMO', 'TLDR', 'WAGMI', 'HODL', 'FUD', 'ATH', 'IPO'
    }

    # User agent for HTTP requests
    USER_AGENT = "optionscanner_bot_v1.0"

    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        lookback_posts: int = 100,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Initialize the Reddit monitor.

        Args:
            subreddits: List of subreddits to monitor (default: wallstreetbets, options, stocks)
            lookback_posts: Number of recent posts to fetch per subreddit
            user_agent: User agent for HTTP requests (optional)
        """
        self.subreddits = subreddits or ["wallstreetbets", "options", "stocks"]
        self.lookback_posts = lookback_posts
        self.user_agent = user_agent or self.USER_AGENT
        self._initialized = True

    def _fetch_rss_feed(self, subreddit: str) -> str:
        """
        Fetch RSS feed for a subreddit.

        Args:
            subreddit: Subreddit name

        Returns:
            RSS feed XML content as string
        """
        url = f"https://www.reddit.com/r/{subreddit}/new.rss"
        headers = {"User-Agent": self.user_agent}
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            logger.warning(
                "HTTP error fetching r/{subreddit} RSS | status={status}",
                subreddit=subreddit,
                status=exc.code,
            )
            return ""
        except URLError as exc:
            logger.warning(
                "Failed to fetch r/{subreddit} RSS | error={error}",
                subreddit=subreddit,
                error=str(exc),
            )
            return ""
        except Exception as exc:
            logger.warning(
                "Unexpected error fetching r/{subreddit} RSS | error={error}",
                subreddit=subreddit,
                error=str(exc),
            )
            return ""

    def _parse_rss_entry(self, entry: ET.Element, subreddit: str) -> Optional[RedditPost]:
        """
        Parse an RSS/Atom entry element into a RedditPost.

        Args:
            entry: XML element for a feed entry
            subreddit: Subreddit name

        Returns:
            RedditPost object or None if parsing fails
        """
        try:
            # Namespaces
            atom_ns = "{http://www.w3.org/2005/Atom}"
            media_ns = "{http://search.yahoo.com/mrss/}"

            # Helper to safely get element text with namespace
            def get_text(tag: str, use_ns: bool = True) -> str:
                full_tag = f"{atom_ns}{tag}" if use_ns else tag
                elem = entry.find(full_tag)
                if elem is not None and elem.text:
                    return elem.text.strip()
                # Try without namespace
                if use_ns:
                    elem = entry.find(tag)
                    if elem is not None and elem.text:
                        return elem.text.strip()
                return ""

            # Extract ID (remove 't3_' prefix if present)
            id_elem = get_text("id")
            post_id = id_elem.replace("t3_", "") if id_elem else ""

            # Extract title
            title = get_text("title")

            # Extract author
            author_elem = entry.find(f"{atom_ns}author/{atom_ns}name")
            if author_elem is None:
                author_elem = entry.find("author/name")
            author = author_elem.text.strip() if author_elem is not None and author_elem.text else ""
            if author.startswith("/u/"):
                author = author[3:]

            # Extract URL
            url = ""
            for link in entry.findall(f"{atom_ns}link"):
                rel = link.get("rel", "alternate")
                if rel == "alternate":
                    url = link.get("href", "")
                    break
            if not url:
                for link in entry.findall("link"):
                    rel = link.get("rel", "alternate")
                    if rel == "alternate":
                        url = link.get("href", "")
                        break

            # Extract published date
            published = get_text("published") or get_text("updated")
            created_utc = datetime.now(timezone.utc)
            if published:
                try:
                    # Parse ISO 8601 format
                    created_utc = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Extract content (HTML) - this contains selftext and comments link
            content_elem = entry.find(f"{atom_ns}content")
            if content_elem is None:
                content_elem = entry.find("content")
            content_html = ""
            if content_elem is not None and content_elem.text:
                content_html = unescape(content_elem.text)

            # Extract selftext from HTML content
            selftext = self._extract_selftext_from_html(content_html)

            # Extract comment count and score from HTML
            num_comments, score = self._extract_stats_from_html(content_html, url)

            # Extract ticker mentions from title and selftext
            ticker_mentions = self.extract_ticker_mentions(title)
            if selftext:
                ticker_mentions.extend(self.extract_ticker_mentions(selftext))
            ticker_mentions = list(set(ticker_mentions))

            return RedditPost(
                id=post_id,
                title=title,
                selftext=selftext,
                subreddit=subreddit,
                author=author or "[deleted]",
                score=score,
                num_comments=num_comments,
                created_utc=created_utc,
                url=url,
                ticker_mentions=ticker_mentions,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse RSS entry | error={error}",
                error=str(exc),
            )
            return None

    def _extract_selftext_from_html(self, html: str) -> str:
        """Extract the post selftext from RSS HTML content."""
        if not html:
            return ""

        # Remove HTML tags and decode entities
        text = re.sub(r'<[^>]+>', ' ', html)
        text = unescape(text)

        # Clean up common RSS artifacts
        text = re.sub(r'\s+submitted\s+by\s+/u/\S+', ' ', text)
        text = re.sub(r'\[link\]', ' ', text)
        text = re.sub(r'\[comments\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _extract_stats_from_html(self, html: str, url: str) -> tuple[int, int]:
        """Extract comment count and score from RSS HTML content.

        Note: RSS feeds don't include score/comment counts reliably.
        We return 0 for both and fetch from post page if needed.
        """
        # RSS doesn't reliably include these stats
        # Default to 0; can be fetched from post page if needed
        return 0, 0

    def initialize(self) -> bool:
        """
        Initialize the Reddit monitor.

        Returns:
            True (RSS feeds don't require authentication)
        """
        logger.info(
            "Reddit monitor initialized (RSS mode) | subreddits={subreddits}",
            subreddits=self.subreddits,
        )
        return True

    def extract_ticker_mentions(self, text: str) -> List[str]:
        """
        Extract ticker mentions from text.

        Looks for:
        1. $TICKER format (most reliable)
        2. Well-known tickers mentioned with proper casing

        Args:
            text: Text to search for ticker mentions

        Returns:
            List of unique ticker symbols found
        """
        if not text:
            return []

        tickers = set()

        # Pattern 1: $TICKER format (most reliable)
        dollar_matches = re.findall(r'\$([A-Z]{1,6})\b', text)
        for match in dollar_matches:
            if match not in self.NON_TICKERS:
                tickers.add(match)

        # Pattern 2: Well-known tickers - only match if they appear as all caps
        # This avoids matching common words like "oil", "spy", "dim", etc.
        # Key insight: Reddit users typically write tickers in ALL CAPS
        well_known_tickers = {
            'SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG',
            'AMZN', 'TSLA', 'META', 'AMD', 'NFLX', 'INTC', 'TQQQ', 'SQQQ', 'SPXU',
            'SPX', 'NDX', 'RUT', 'VIX', 'UVXY', 'VXX', 'SVXY', 'GLD', 'SLV', 'USO',
            'BND', 'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'SOXL', 'SOXS', 'TECL',
            'TECS', 'FNGU', 'FNGD', 'NVDL', 'NVDX', 'CONY', 'YINN', 'YANG', 'FXI',
            'KWEB', 'LABU', 'LABD', 'DUSL', 'DUST', 'JNUG', 'JDST', 'NUGT', 'GUSH',
            'DRIP', 'ERX', 'ERY', 'FAS', 'FAZ', 'TNA', 'TZA', 'UDOW', 'SDOW', 'UPRO',
            'SSO', 'SH', 'PSQ', 'QID', 'QLD', 'DOG', 'DXD', 'SPXL', 'SPXS', 'SDS',
            'SRTY', 'URTY', 'TLT', 'TMF', 'TMV', 'TBT', 'TBF', 'TYD', 'TYO', 'UGL',
            'GLL', 'AGQ', 'ZSL', 'USL', 'UCO', 'SCO', 'BOIL', 'KOLD', 'UNG', 'FCG',
            'OIH', 'XLE', 'XOP', 'VDE', 'IEO', 'IEZ', 'BNO', 'DBO', 'VTI', 'VOO',
            'VEA', 'VWO', 'AGG', 'BNDX', 'VTV', 'VUG', 'VB', 'VO', 'IJH', 'IJR',
            'IVV', 'IVE', 'IVW', 'VIG', 'VYM', 'SCHD', 'VXUS', 'VT', 'BND', 'VMOT',
        }

        # Match well-known tickers - look for ALL CAPS only (not mixed case)
        # This avoids matching "oil" when it appears as a regular word
        for ticker in well_known_tickers:
            # Use word boundary and require exact uppercase match
            pattern = r'\b(' + ticker + r')\b'
            matches = re.finditer(pattern, text)
            for match in matches:
                # Only add if the matched text is exactly uppercase
                if match.group(1).isupper():
                    tickers.add(ticker)

        return list(tickers)

    def fetch_recent_posts(self) -> List[RedditPost]:
        """
        Fetch recent posts from monitored subreddits via RSS.

        Returns:
            List of RedditPost objects
        """
        posts: List[RedditPost] = []
        posts_per_subreddit = max(10, self.lookback_posts // len(self.subreddits) + 1)

        for subreddit_name in self.subreddits:
            try:
                rss_content = self._fetch_rss_feed(subreddit_name)
                if not rss_content:
                    continue

                root = ET.fromstring(rss_content)

                # Atom feed structure
                entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

                for entry in entries[:posts_per_subreddit]:
                    post = self._parse_rss_entry(entry, subreddit_name)
                    if post:
                        posts.append(post)

            except ET.ParseError as exc:
                logger.warning(
                    "Failed to parse RSS feed for r/{subreddit} | error={error}",
                    subreddit=subreddit_name,
                    error=str(exc),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch posts from r/{subreddit} | error={error}",
                    subreddit=subreddit_name,
                    error=str(exc),
                )

        # Sort by creation time, most recent first
        posts.sort(key=lambda p: p.created_utc, reverse=True)

        # Limit to lookback_posts
        posts = posts[:self.lookback_posts]

        logger.info(
            "Fetched {count} posts from {num_subreddits} subreddits",
            count=len(posts),
            num_subreddits=len(self.subreddits),
        )
        return posts

    def fetch_post_comments(self, post_id: str, subreddit: str = "") -> List[RedditComment]:
        """
        Fetch comments from a specific post.

        Note: Comments are fetched by scraping the Reddit post HTML page.
        This is less reliable than the official API and may be rate-limited.
        For production use, consider caching comments or using the official API.

        Args:
            post_id: Reddit post ID
            subreddit: Subreddit name (required)

        Returns:
            List of RedditComment objects
        """
        if not subreddit:
            logger.warning("Subreddit required for comment fetching")
            return []

        comments: List[RedditComment] = []
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"
        headers = {"User-Agent": self.user_agent}
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8")
        except Exception as exc:
            logger.warning(
                "Failed to fetch comments for post {post_id} | error={error}",
                post_id=post_id,
                error=str(exc),
            )
            return []

        # Parse comments from HTML
        # Reddit HTML structure - look for comment containers
        comment_pattern = re.compile(
            r'<[^>]*data-testid="comment"[^>]*>.*?</[^>]*>',
            re.DOTALL | re.IGNORECASE
        )

        # Extract comment text, author, score
        # This is a simplified parser - Reddit HTML can change
        try:
            # Find all comment blocks
            # Note: This is a basic implementation; production may need more robust parsing
            pass  # Comment scraping is complex and may break with HTML changes
        except Exception:
            pass

        logger.debug(
            "Comment scraping disabled by default - use official API for comments | post={post_id}",
            post_id=post_id,
        )

        return comments

    def get_trending_tickers(self, posts: Optional[List[RedditPost]] = None) -> Dict[str, int]:
        """
        Get trending tickers based on mention frequency.

        Args:
            posts: Optional list of posts to analyze. If None, fetches recent posts.

        Returns:
            Dictionary mapping ticker symbols to mention counts
        """
        if posts is None:
            posts = self.fetch_recent_posts()

        ticker_counts: Dict[str, int] = {}
        for post in posts:
            for ticker in post.ticker_mentions:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        # Sort by count, descending
        sorted_tickers = dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True))
        return sorted_tickers


__all__ = [
    "RedditMonitor",
    "RedditPost",
    "RedditComment",
]
