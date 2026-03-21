"""Whale activity detection from Reddit posts and comments."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from reddit_monitor import RedditPost, RedditComment


class WhaleDirection(str, Enum):
    """Direction of whale activity."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class WhaleActivity:
    """Represents detected whale activity for a ticker."""

    symbol: str
    direction: WhaleDirection
    confidence: float  # 0.0 to 1.0
    source_posts: List[str] = field(default_factory=list)  # Post IDs
    source_comments: List[str] = field(default_factory=list)  # Comment IDs
    mention_count: int = 0
    total_score: int = 0  # Sum of upvotes
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


@dataclass(slots=True)
class WhaleSignal:
    """A trading signal based on whale activity."""

    symbol: str
    direction: str  # CALL or PUT
    confidence: float
    rationale: str
    whale_activity: WhaleActivity


class WhaleDetector:
    """Detects whale activity mentions in Reddit posts and comments."""

    # Keywords indicating large/whale trades
    WHALE_KEYWORDS = {
        # Whale-related terms
        "whale": 1.0,
        "whales": 1.0,
        "big money": 0.9,
        "smart money": 0.8,
        "institutional": 0.7,
        "institutions": 0.7,
        # Size indicators
        "large order": 0.9,
        "large orders": 0.9,
        "huge order": 0.95,
        "huge orders": 0.95,
        "massive order": 0.95,
        "massive orders": 0.95,
        "unusual order": 0.85,
        "unusual orders": 0.85,
        "unusual activity": 0.8,
        "unusual volume": 0.85,
        "sweeps": 0.9,
        "sweep": 0.9,
        # Money amounts
        "$1m": 0.9,
        "$1mm": 0.9,
        "$1 million": 0.9,
        "$2m": 0.95,
        "$5m": 1.0,
        "$10m": 1.0,
        "$100m": 1.0,
        "million dollar": 0.85,
        "millions": 0.8,
        # Position descriptors
        "heavy buying": 0.8,
        "heavy selling": 0.8,
        "aggressive buying": 0.85,
        "aggressive selling": 0.85,
        "accumulating": 0.7,
        "dumping": 0.75,
    }

    # Bullish indicators
    BULLISH_KEYWORDS = {
        "bullish": 0.8,
        "bull": 0.7,
        "calls": 0.7,
        "call": 0.65,
        "long": 0.6,
        "buy": 0.5,
        "moon": 0.7,
        "rocket": 0.7,
        "squeeze": 0.8,
        "breakout": 0.75,
        "upside": 0.7,
        "rally": 0.75,
        "surge": 0.8,
        "jump": 0.6,
        "gap up": 0.8,
    }

    # Bearish indicators
    BEARISH_KEYWORDS = {
        "bearish": 0.8,
        "bear": 0.7,
        "puts": 0.7,
        "put": 0.65,
        "short": 0.6,
        "sell": 0.5,
        "crash": 0.8,
        "dump": 0.75,
        "downside": 0.7,
        "drop": 0.65,
        "fall": 0.6,
        "collapse": 0.85,
        "plunge": 0.8,
        "gap down": 0.8,
    }

    # Confidence thresholds
    MIN_CONFIDENCE = 0.3
    HIGH_CONFIDENCE = 0.7

    def __init__(
        self,
        min_confidence: float = 0.3,
        min_mention_count: int = 2,
        min_total_score: int = 50,
    ) -> None:
        """
        Initialize the whale detector.

        Args:
            min_confidence: Minimum confidence score to report whale activity
            min_mention_count: Minimum number of mentions required
            min_total_score: Minimum total upvote score required
        """
        self.min_confidence = min_confidence
        self.min_mention_count = min_mention_count
        self.min_total_score = min_total_score
        self._whale_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficient matching."""
        patterns = {}
        # Whale keywords pattern
        whale_words = sorted(self.WHALE_KEYWORDS.keys(), key=len, reverse=True)
        patterns["whale"] = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in whale_words) + r")\b",
            re.IGNORECASE
        )
        # Bullish keywords pattern
        bullish_words = sorted(self.BULLISH_KEYWORDS.keys(), key=len, reverse=True)
        patterns["bullish"] = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in bullish_words) + r")\b",
            re.IGNORECASE
        )
        # Bearish keywords pattern
        bearish_words = sorted(self.BEARISH_KEYWORDS.keys(), key=len, reverse=True)
        patterns["bearish"] = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in bearish_words) + r")\b",
            re.IGNORECASE
        )
        return patterns

    def score_whale_keywords(self, text: str) -> Tuple[float, List[str]]:
        """
        Score text for whale-related keywords.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (max_score, matched_keywords)
        """
        if not text:
            return 0.0, []

        matches = self._whale_patterns["whale"].finditer(text)
        max_score = 0.0
        matched = []

        for match in matches:
            keyword = match.group(1).lower()
            score = self.WHALE_KEYWORDS.get(keyword, 0.0)
            if score > max_score:
                max_score = score
            if score >= 0.5:  # Only track significant matches
                matched.append(keyword)

        return max_score, list(set(matched))

    def detect_direction(self, text: str) -> Tuple[WhaleDirection, float]:
        """
        Detect the directional bias of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (direction, confidence)
        """
        if not text:
            return WhaleDirection.NEUTRAL, 0.0

        text_lower = text.lower()

        # Score bullish indicators
        bullish_matches = self._whale_patterns["bullish"].finditer(text_lower)
        bullish_score = sum(
            self.BULLISH_KEYWORDS.get(m.group(1).lower(), 0.0)
            for m in bullish_matches
        )

        # Score bearish indicators
        bearish_matches = self._whale_patterns["bearish"].finditer(text_lower)
        bearish_score = sum(
            self.BEARISH_KEYWORDS.get(m.group(1).lower(), 0.0)
            for m in bearish_matches
        )

        if bullish_score > bearish_score * 1.2:  # 20% threshold for confidence
            confidence = min(bullish_score / 2.0, 1.0)
            return WhaleDirection.BULLISH, confidence
        elif bearish_score > bullish_score * 1.2:
            confidence = min(bearish_score / 2.0, 1.0)
            return WhaleDirection.BEARISH, confidence
        else:
            return WhaleDirection.NEUTRAL, 0.0

    def analyze_post(self, post: RedditPost) -> Optional[WhaleActivity]:
        """
        Analyze a Reddit post for whale activity.

        Args:
            post: RedditPost to analyze

        Returns:
            WhaleActivity if detected, None otherwise
        """
        if not post.ticker_mentions:
            return None

        # Combine title and selftext for analysis
        full_text = f"{post.title} {post.selftext}"

        # Score for whale keywords
        whale_score, matched_keywords = self.score_whale_keywords(full_text)
        if whale_score < 0.5:  # Not significant whale activity
            return None

        # Detect direction
        direction, direction_confidence = self.detect_direction(full_text)
        if direction == WhaleDirection.NEUTRAL:
            return None

        # Calculate overall confidence
        # Factors: whale score, direction confidence, post engagement
        # Note: RSS feeds don't provide score/num_comments reliably, so we use
        # a simplified engagement factor based on content length as a proxy
        engagement_factor = min(1.0, len(full_text) / 500)  # Longer posts often more detailed
        confidence = (whale_score * 0.5 + direction_confidence * 0.35 + engagement_factor * 0.15)

        if confidence < self.min_confidence:
            return None

        details = f"Detected {direction.value} whale activity: {', '.join(matched_keywords)}"

        return WhaleActivity(
            symbol=post.ticker_mentions[0],  # Primary ticker
            direction=direction,
            confidence=confidence,
            source_posts=[post.id],
            source_comments=[],
            mention_count=1,
            total_score=post.score,  # May be 0 from RSS
            details=details,
        )

    def analyze_comment(self, comment: RedditComment) -> Optional[WhaleActivity]:
        """
        Analyze a Reddit comment for whale activity.

        Args:
            comment: RedditComment to analyze

        Returns:
            WhaleActivity if detected, None otherwise
        """
        if not comment.ticker_mentions:
            return None

        whale_score, matched_keywords = self.score_whale_keywords(comment.body)
        if whale_score < 0.5:
            return None

        direction, direction_confidence = self.detect_direction(comment.body)
        if direction == WhaleDirection.NEUTRAL:
            return None

        engagement_factor = min(1.0, comment.score / 50)
        confidence = (whale_score * 0.4 + direction_confidence * 0.4 + engagement_factor * 0.2)

        if confidence < self.min_confidence:
            return None

        details = f"Detected {direction.value} whale activity in comment: {', '.join(matched_keywords)}"

        return WhaleActivity(
            symbol=comment.ticker_mentions[0],
            direction=direction,
            confidence=confidence,
            source_comments=[comment.id],
            mention_count=1,
            total_score=comment.score,
            details=details,
        )

    def aggregate_activities(
        self,
        activities: List[WhaleActivity],
    ) -> Dict[str, WhaleActivity]:
        """
        Aggregate whale activities by symbol.

        Args:
            activities: List of WhaleActivity objects

        Returns:
            Dictionary mapping symbols to aggregated WhaleActivity
        """
        aggregated: Dict[str, List[WhaleActivity]] = {}
        for activity in activities:
            if activity.symbol not in aggregated:
                aggregated[activity.symbol] = []
            aggregated[activity.symbol].append(activity)

        result: Dict[str, WhaleActivity] = {}
        for symbol, acts in aggregated.items():
            # Combine activities for same symbol
            total_mentions = sum(a.mention_count for a in acts)
            total_score = sum(a.total_score for a in acts)

            # Weighted average confidence
            avg_confidence = sum(a.confidence * a.mention_count for a in acts) / max(total_mentions, 1)

            # Determine dominant direction
            bullish_count = sum(1 for a in acts if a.direction == WhaleDirection.BULLISH)
            bearish_count = sum(1 for a in acts if a.direction == WhaleDirection.BEARISH)

            if bullish_count > bearish_count:
                dominant_direction = WhaleDirection.BULLISH
            elif bearish_count > bullish_count:
                dominant_direction = WhaleDirection.BEARISH
            else:
                dominant_direction = WhaleDirection.NEUTRAL

            # Boost confidence if multiple sources agree
            if len(acts) >= 2 and dominant_direction != WhaleDirection.NEUTRAL:
                avg_confidence = min(avg_confidence * 1.2, 1.0)

            all_post_ids = []
            all_comment_ids = []
            for a in acts:
                all_post_ids.extend(a.source_posts)
                all_comment_ids.extend(a.source_comments)

            details = f"Aggregated from {len(acts)} sources | Posts: {len(all_post_ids)}, Comments: {len(all_comment_ids)}"

            result[symbol] = WhaleActivity(
                symbol=symbol,
                direction=dominant_direction,
                confidence=avg_confidence,
                source_posts=list(set(all_post_ids)),
                source_comments=list(set(all_comment_ids)),
                mention_count=total_mentions,
                total_score=total_score,
                details=details,
            )

        return result

    def detect_from_posts(
        self,
        posts: List[RedditPost],
    ) -> Dict[str, WhaleActivity]:
        """
        Detect whale activity from a list of Reddit posts.

        Args:
            posts: List of RedditPost objects

        Returns:
            Dictionary mapping symbols to WhaleActivity
        """
        activities: List[WhaleActivity] = []

        # Analyze posts
        for post in posts:
            activity = self.analyze_post(post)
            if activity:
                activities.append(activity)
                logger.debug(
                    "Whale activity detected in post | symbol={symbol} direction={direction} confidence={confidence:.2f}",
                    symbol=activity.symbol,
                    direction=activity.direction.value,
                    confidence=activity.confidence,
                )

        # Aggregate by symbol
        aggregated = self.aggregate_activities(activities)

        # Filter by confidence and mention count
        filtered = {
            symbol: activity
            for symbol, activity in aggregated.items()
            if activity.confidence >= self.min_confidence
            and activity.mention_count >= self.min_mention_count
        }

        logger.info(
            "Whale detection complete | symbols={count} total_posts={posts}",
            count=len(filtered),
            posts=len(posts),
        )

        return filtered


__all__ = [
    "WhaleDetector",
    "WhaleActivity",
    "WhaleDirection",
    "WhaleSignal",
]
