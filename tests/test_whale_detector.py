"""Unit tests for whale detection functionality."""
import unittest
from datetime import datetime, timezone

from whale_detector import WhaleActivity, WhaleDetector, WhaleDirection
from reddit_monitor import RedditPost, RedditComment


class WhaleDetectorTests(unittest.TestCase):
    """Tests for the WhaleDetector class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.detector = WhaleDetector(min_confidence=0.3)

    def test_whale_keyword_detection(self) -> None:
        """Test detection of whale-related keywords."""
        text = "Huge whale order detected! Someone is buying $1M in calls!"
        score, matched = self.detector.score_whale_keywords(text)

        self.assertGreater(score, 0.5)
        self.assertIn("whale", matched)

    def test_no_whale_keyword(self) -> None:
        """Test text without whale keywords."""
        text = "Just bought some calls, YOLO!"
        score, matched = self.detector.score_whale_keywords(text)

        self.assertEqual(score, 0.0)
        self.assertEqual(len(matched), 0)

    def test_bullish_direction_detection(self) -> None:
        """Test detection of bullish sentiment."""
        text = "Massive whale buying calls! This is going to the moon! Bullish!"
        direction, confidence = self.detector.detect_direction(text)

        self.assertEqual(direction, WhaleDirection.BULLISH)
        self.assertGreater(confidence, 0.5)

    def test_bearish_direction_detection(self) -> None:
        """Test detection of bearish sentiment."""
        text = "Huge whale dumping puts! Crash incoming! Bearish!"
        direction, confidence = self.detector.detect_direction(text)

        self.assertEqual(direction, WhaleDirection.BEARISH)
        self.assertGreater(confidence, 0.5)

    def test_neutral_direction(self) -> None:
        """Test neutral sentiment detection."""
        text = "Market is quiet today, no big moves."
        direction, confidence = self.detector.detect_direction(text)

        self.assertEqual(direction, WhaleDirection.NEUTRAL)

    def test_analyze_post_with_whale_activity(self) -> None:
        """Test analyzing a post with whale activity."""
        post = RedditPost(
            id="test123",
            title="Huge whale buying NVDA calls! $10M order!",
            selftext="Someone just dropped serious money on NVDA 150c calls expiring next month.",
            subreddit="wallstreetbets",
            author="diamond_hands_42",
            score=150,
            num_comments=45,
            created_utc=datetime.now(timezone.utc),
            url="https://reddit.com/r/wallstreetbets/test123",
            ticker_mentions=["NVDA"],
        )

        activity = self.detector.analyze_post(post)

        self.assertIsNotNone(activity)
        self.assertEqual(activity.symbol, "NVDA")
        self.assertEqual(activity.direction, WhaleDirection.BULLISH)
        self.assertGreater(activity.confidence, self.detector.min_confidence)

    def test_analyze_post_without_whale_activity(self) -> None:
        """Test analyzing a post without whale activity."""
        post = RedditPost(
            id="test456",
            title="Just bought some NVDA calls",
            selftext="Feeling bullish on NVDA",
            subreddit="wallstreetbets",
            author="regular_trader",
            score=5,
            num_comments=2,
            created_utc=datetime.now(timezone.utc),
            url="https://reddit.com/r/wallstreetbets/test456",
            ticker_mentions=["NVDA"],
        )

        activity = self.detector.analyze_post(post)

        # Should be None because no whale keywords
        self.assertIsNone(activity)

    def test_analyze_comment_with_whale_activity(self) -> None:
        """Test analyzing a comment with whale activity."""
        comment = RedditComment(
            id="comment789",
            body="Saw a huge whale sweep on SPY puts! Someone is betting on a crash!",
            post_id="post123",
            subreddit="options",
            author="options_pro",
            score=75,
            created_utc=datetime.now(timezone.utc),
            ticker_mentions=["SPY"],
        )

        activity = self.detector.analyze_comment(comment)

        self.assertIsNotNone(activity)
        self.assertEqual(activity.symbol, "SPY")
        self.assertEqual(activity.direction, WhaleDirection.BEARISH)

    def test_aggregate_activities_same_symbol(self) -> None:
        """Test aggregation of multiple activities for the same symbol."""
        activities = [
            WhaleActivity(
                symbol="AAPL",
                direction=WhaleDirection.BULLISH,
                confidence=0.7,
                source_posts=["post1"],
                mention_count=1,
                total_score=100,
            ),
            WhaleActivity(
                symbol="AAPL",
                direction=WhaleDirection.BULLISH,
                confidence=0.8,
                source_posts=["post2"],
                mention_count=1,
                total_score=150,
            ),
            WhaleActivity(
                symbol="AAPL",
                direction=WhaleDirection.BEARISH,
                confidence=0.5,
                source_comments=["comment1"],
                mention_count=1,
                total_score=30,
            ),
        ]

        aggregated = self.detector.aggregate_activities(activities)

        self.assertIn("AAPL", aggregated)
        result = aggregated["AAPL"]
        self.assertEqual(result.mention_count, 3)
        self.assertEqual(result.total_score, 280)
        # Bullish should win (2 vs 1)
        self.assertEqual(result.direction, WhaleDirection.BULLISH)

    def test_aggregate_activities_multiple_symbols(self) -> None:
        """Test aggregation with multiple symbols."""
        activities = [
            WhaleActivity(
                symbol="NVDA",
                direction=WhaleDirection.BULLISH,
                confidence=0.7,
                source_posts=["post1"],
                mention_count=1,
                total_score=100,
            ),
            WhaleActivity(
                symbol="TSLA",
                direction=WhaleDirection.BEARISH,
                confidence=0.6,
                source_posts=["post2"],
                mention_count=1,
                total_score=80,
            ),
        ]

        aggregated = self.detector.aggregate_activities(activities)

        self.assertEqual(len(aggregated), 2)
        self.assertIn("NVDA", aggregated)
        self.assertIn("TSLA", aggregated)
        self.assertEqual(aggregated["NVDA"].direction, WhaleDirection.BULLISH)
        self.assertEqual(aggregated["TSLA"].direction, WhaleDirection.BEARISH)

    def test_detect_from_posts_filters_low_confidence(self) -> None:
        """Test that low confidence activities are filtered."""
        posts = [
            RedditPost(
                id="post1",
                title="Small whale buying TSLA calls",
                selftext="",
                subreddit="wallstreetbets",
                author="user1",
                score=10,  # Low engagement
                num_comments=2,
                created_utc=datetime.now(timezone.utc),
                url="https://reddit.com/r/wallstreetbets/post1",
                ticker_mentions=["TSLA"],
            ),
        ]

        # Detector with higher thresholds
        detector = WhaleDetector(min_confidence=0.5, min_mention_count=2)
        result = detector.detect_from_posts(posts, include_comments=False)

        # Should be filtered out due to low confidence and mention count
        self.assertEqual(len(result), 0)


class RedditMonitorTests(unittest.TestCase):
    """Tests for the RedditMonitor class."""

    def test_ticker_extraction(self) -> None:
        """Test extraction of ticker symbols from text."""
        from reddit_monitor import RedditMonitor

        monitor = RedditMonitor()

        # Test basic ticker extraction
        text = "Buying $NVDA and AAPL calls today!"
        tickers = monitor.extract_ticker_mentions(text)

        self.assertIn("NVDA", tickers)
        self.assertIn("AAPL", tickers)

    def test_ticker_extraction_excludes_common_words(self) -> None:
        """Test that common words are excluded from ticker extraction."""
        from reddit_monitor import RedditMonitor

        monitor = RedditMonitor()

        text = "I think IT is going up and TO the moon!"
        tickers = monitor.extract_ticker_mentions(text)

        # Common words should be excluded
        self.assertNotIn("I", tickers)
        self.assertNotIn("IT", tickers)
        self.assertNotIn("TO", tickers)

    def test_ticker_extraction_single_letter(self) -> None:
        """Test that single letters are excluded."""
        from reddit_monitor import RedditMonitor

        monitor = RedditMonitor()

        text = "A is for apple, buy $AAPL"
        tickers = monitor.extract_ticker_mentions(text)

        self.assertNotIn("A", tickers)
        self.assertIn("AAPL", tickers)


if __name__ == "__main__":
    unittest.main()
