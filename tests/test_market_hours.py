"""Tests for market hours detection and automatic market data type selection."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from optionscanner.market_hours import MarketHoursChecker


class MarketHoursCheckerTest(unittest.TestCase):
    """Test market hours detection logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.checker = MarketHoursChecker()
        self.pt = ZoneInfo("America/Los_Angeles")

    def test_is_market_hours_during_trading(self) -> None:
        """Test that market hours are detected correctly during trading hours."""
        # Test a time during market hours: 10:00 AM PT on a weekday
        test_time = datetime(2026, 4, 6, 10, 0, 0, tzinfo=self.pt)  # Monday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertTrue(self.checker.is_market_hours())

    def test_is_market_hours_before_market_opens(self) -> None:
        """Test that before market hours returns False."""
        # Test 5:00 AM PT (before 6:30 AM open)
        test_time = datetime(2026, 4, 6, 5, 0, 0, tzinfo=self.pt)  # Monday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertFalse(self.checker.is_market_hours())

    def test_is_market_hours_exactly_at_open(self) -> None:
        """Test that exactly at 6:30 AM returns True."""
        test_time = datetime(2026, 4, 6, 6, 30, 0, tzinfo=self.pt)  # Monday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertTrue(self.checker.is_market_hours())

    def test_is_market_hours_exactly_at_close(self) -> None:
        """Test that exactly at 1:00 PM returns True (market closes at this time)."""
        test_time = datetime(2026, 4, 6, 13, 0, 0, tzinfo=self.pt)  # Monday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            # Market is still open at exactly 1:00 PM (closing time)
            self.assertTrue(self.checker.is_market_hours())

    def test_is_market_hours_after_market_closes(self) -> None:
        """Test that after market hours returns False."""
        # Test 2:00 PM PT (after 1:00 PM close)
        test_time = datetime(2026, 4, 6, 14, 0, 0, tzinfo=self.pt)  # Monday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertFalse(self.checker.is_market_hours())

    def test_is_market_hours_weekend_saturday(self) -> None:
        """Test that weekends return False (Saturday)."""
        # Saturday, April 4, 2026
        test_time = datetime(2026, 4, 4, 10, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertFalse(self.checker.is_market_hours())

    def test_is_market_hours_weekend_sunday(self) -> None:
        """Test that weekends return False (Sunday)."""
        # Sunday, April 5, 2026
        test_time = datetime(2026, 4, 5, 10, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertFalse(self.checker.is_market_hours())

    def test_is_market_hours_weekday_friday(self) -> None:
        """Test that Friday market hours are detected correctly."""
        # Friday, April 3, 2026 at 10:00 AM
        test_time = datetime(2026, 4, 3, 10, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertTrue(self.checker.is_market_hours())

    def test_get_market_data_type_during_market_hours(self) -> None:
        """Test that LIVE is returned during market hours."""
        test_time = datetime(2026, 4, 6, 10, 0, 0, tzinfo=self.pt)  # Monday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertEqual("LIVE", self.checker.get_market_data_type())

    def test_get_market_data_type_outside_market_hours(self) -> None:
        """Test that FROZEN is returned outside market hours."""
        test_time = datetime(2026, 4, 6, 15, 0, 0, tzinfo=self.pt)  # Monday after hours
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertEqual("FROZEN", self.checker.get_market_data_type())

    def test_get_market_data_type_weekend(self) -> None:
        """Test that FROZEN is returned on weekends."""
        test_time = datetime(2026, 4, 4, 10, 0, 0, tzinfo=self.pt)  # Saturday
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            self.assertEqual("FROZEN", self.checker.get_market_data_type())

    def test_get_next_market_open_before_market_hours(self) -> None:
        """Test next market open calculation before market hours."""
        # Monday 5:00 AM - market opens at 6:30 AM same day
        test_time = datetime(2026, 4, 6, 5, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_open = self.checker.get_next_market_open()
            expected = datetime(2026, 4, 6, 6, 30, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_open)

    def test_get_next_market_open_during_market_hours(self) -> None:
        """Test next market open calculation during market hours."""
        # Monday 10:00 AM - next market open is Tuesday 6:30 AM
        test_time = datetime(2026, 4, 6, 10, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_open = self.checker.get_next_market_open()
            expected = datetime(2026, 4, 7, 6, 30, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_open)

    def test_get_next_market_open_after_market_hours(self) -> None:
        """Test next market open calculation after market hours."""
        # Monday 3:00 PM - next market open is Tuesday 6:30 AM
        test_time = datetime(2026, 4, 6, 15, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_open = self.checker.get_next_market_open()
            expected = datetime(2026, 4, 7, 6, 30, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_open)

    def test_get_next_market_open_friday_after_hours(self) -> None:
        """Test next market open on Friday after hours (should be Monday)."""
        # Friday 3:00 PM - next market open is Monday 6:30 AM
        test_time = datetime(2026, 4, 3, 15, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_open = self.checker.get_next_market_open()
            expected = datetime(2026, 4, 6, 6, 30, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_open)

    def test_get_next_market_close_before_market_hours(self) -> None:
        """Test next market close calculation before market hours."""
        # Monday 5:00 AM - market closes at 1:00 PM same day
        test_time = datetime(2026, 4, 6, 5, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_close = self.checker.get_next_market_close()
            expected = datetime(2026, 4, 6, 13, 0, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_close)

    def test_get_next_market_close_during_market_hours(self) -> None:
        """Test next market close calculation during market hours."""
        # Monday 10:00 AM - market closes today at 1:00 PM
        test_time = datetime(2026, 4, 6, 10, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_close = self.checker.get_next_market_close()
            expected = datetime(2026, 4, 6, 13, 0, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_close)

    def test_get_next_market_close_after_market_hours(self) -> None:
        """Test next market close calculation after market hours."""
        # Monday 3:00 PM - next market close is Tuesday 1:00 PM
        test_time = datetime(2026, 4, 6, 15, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_close = self.checker.get_next_market_close()
            expected = datetime(2026, 4, 7, 13, 0, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_close)

    def test_get_next_market_close_friday_after_hours(self) -> None:
        """Test next market close on Friday after hours (should be Monday)."""
        # Friday 3:00 PM - next market close is Monday 1:00 PM
        test_time = datetime(2026, 4, 3, 15, 0, 0, tzinfo=self.pt)
        with patch.object(self.checker, '_get_current_time', return_value=test_time):
            next_close = self.checker.get_next_market_close()
            expected = datetime(2026, 4, 6, 13, 0, 0, tzinfo=self.pt)
            self.assertEqual(expected, next_close)

    def test_get_current_time_uses_pt_timezone(self) -> None:
        """Test that _get_current_time returns time in PT timezone."""
        current_time = self.checker._get_current_time()
        self.assertEqual(current_time.tzinfo, self.pt)


if __name__ == "__main__":
    unittest.main()
