"""Reddit monitoring module for tracking whale activity mentions."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from loguru import logger

try:
    import praw
except ImportError as exc:
    raise ImportError(
        "praw is required for Reddit monitoring. Install with: pip install praw"
    ) from exc


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
    """Monitors Reddit subreddits for posts and comments mentioning tickers."""

    # Regex pattern to match stock tickers (e.g., $NVDA, NVDA)
    TICKER_PATTERN = re.compile(r'\$?([A-Z]{1,5})(?=\s|[^A-Z]|$)')

    # Common non-ticker words to exclude
    NON_TICKERS = {
        'A', 'I', 'IT', 'TO', 'IN', 'ON', 'AT', 'AN', 'AS', 'AM', 'BE', 'BY',
        'DO', 'GO', 'HE', 'ME', 'MY', 'NO', 'OF', 'OR', 'SO', 'UP', 'US', 'WE',
        'YOLO', 'DD', 'FAQ', 'IMO', 'TLDR', 'WAGMI', 'HODL', 'FUD', 'ATH', 'IPO'
    }

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        lookback_posts: int = 100,
    ) -> None:
        """
        Initialize the Reddit monitor.

        Args:
            client_id: Reddit API client ID (env: REDDIT_CLIENT_ID)
            client_secret: Reddit API client secret (env: REDDIT_CLIENT_SECRET)
            user_agent: Reddit API user agent (env: REDDIT_USER_AGENT)
            subreddits: List of subreddits to monitor (default: wallstreetbets, options, stocks)
            lookback_posts: Number of recent posts to fetch per subreddit
        """
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "optionscanner_bot_v1.0")
        self.subreddits = subreddits or ["wallstreetbets", "options", "stocks"]
        self.lookback_posts = lookback_posts
        self.reddit: Optional[praw.Reddit] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the Reddit API client.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return True

        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit API credentials not provided. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
            )
            return False

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
            # Test connection
            self.reddit.user.me()
            self._initialized = True
            logger.info(
                "Reddit monitor initialized | subreddits={subreddits}",
                subreddits=self.subreddits,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to initialize Reddit monitor | error={error}",
                error=str(exc),
            )
            return False

    def extract_ticker_mentions(self, text: str) -> List[str]:
        """
        Extract ticker mentions from text.

        Args:
            text: Text to search for ticker mentions

        Returns:
            List of unique ticker symbols found
        """
        if not text:
            return []

        matches = self.TICKER_PATTERN.findall(text.upper())
        tickers = [m for m in matches if m not in self.NON_TICKERS and len(m) >= 2]
        return list(set(tickers))

    def fetch_recent_posts(self) -> List[RedditPost]:
        """
        Fetch recent posts from monitored subreddits.

        Returns:
            List of RedditPost objects
        """
        if not self._initialized:
            if not self.initialize():
                return []

        posts: List[RedditPost] = []
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.new(limit=self.lookback_posts // len(self.subreddits) + 1):
                    ticker_mentions = self.extract_ticker_mentions(submission.title)
                    ticker_mentions.extend(self.extract_ticker_mentions(submission.selftext))
                    ticker_mentions = list(set(ticker_mentions))

                    post = RedditPost(
                        id=submission.id,
                        title=submission.title,
                        selftext=submission.selftext,
                        subreddit=subreddit_name,
                        author=str(submission.author) if submission.author else "[deleted]",
                        score=submission.score,
                        num_comments=submission.num_comments,
                        created_utc=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        url=submission.url,
                        ticker_mentions=ticker_mentions,
                    )
                    posts.append(post)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch posts from r/{subreddit} | error={error}",
                    subreddit=subreddit_name,
                    error=str(exc),
                )

        # Sort by creation time, most recent first
        posts.sort(key=lambda p: p.created_utc, reverse=True)
        logger.info(
            "Fetched {count} posts from {subreddits} subreddits",
            count=len(posts),
            subreddits=len(self.subreddits),
        )
        return posts

    def fetch_post_comments(self, post_id: str) -> List[RedditComment]:
        """
        Fetch comments from a specific post.

        Args:
            post_id: Reddit post ID

        Returns:
            List of RedditComment objects
        """
        if not self._initialized:
            return []

        comments: List[RedditComment] = []
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Skip "more comments" for efficiency

            for comment in submission.comments.list()[:50]:  # Limit to top 50 comments
                if comment.body in ("[removed]", "[deleted]"):
                    continue

                ticker_mentions = self.extract_ticker_mentions(comment.body)

                reddit_comment = RedditComment(
                    id=comment.id,
                    body=comment.body,
                    post_id=post_id,
                    subreddit=submission.subreddit.display_name,
                    author=str(comment.author) if comment.author else "[deleted]",
                    score=getattr(comment, "score", 0),
                    created_utc=datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                    ticker_mentions=ticker_mentions,
                )
                comments.append(reddit_comment)
        except Exception as exc:
            logger.warning(
                "Failed to fetch comments for post {post_id} | error={error}",
                post_id=post_id,
                error=str(exc),
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
