"""Market hours detection for automatic market data type selection.

This module provides utilities to determine whether the market is currently open
and automatically select the appropriate market data type (LIVE vs FROZEN) based
on market hours.

Market Hours:
    - Regular Trading Hours: 6:30 AM - 1:00 PM Pacific Time
    - Days: Weekdays only (Monday-Friday)
    - Timezone: America/Los_Angeles
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class MarketHoursChecker:
    """Determines if the market is currently open and provides related utilities."""

    # Market hours in Pacific Time
    MARKET_OPEN_HOUR = 6
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 13
    MARKET_CLOSE_MINUTE = 0

    def __init__(self, timezone: str = "America/Los_Angeles") -> None:
        """Initialize the market hours checker.

        Args:
            timezone: The timezone to use for market hours calculations.
                     Defaults to America/Los_Angeles (Pacific Time).
        """
        self.timezone = ZoneInfo(timezone)

    def _get_current_time(self) -> datetime:
        """Get the current time in the configured timezone.

        Returns:
            Current datetime in the configured timezone.
        """
        return datetime.now(self.timezone)

    def is_market_hours(self, check_time: datetime | None = None) -> bool:
        """Check if the current time is within market hours.

        Market hours are 6:30 AM to 1:00 PM Pacific Time, weekdays only.

        Args:
            check_time: Optional datetime to check. If None, uses current time.

        Returns:
            True if currently within market hours, False otherwise.
        """
        if check_time is None:
            check_time = self._get_current_time()
        else:
            # Convert to our timezone if not already
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=self.timezone)
            else:
                check_time = check_time.astimezone(self.timezone)

        # Check if it's a weekend (Saturday=5, Sunday=6)
        if check_time.weekday() >= 5:
            return False

        # Create datetime objects for market open and close on the same day
        market_open = check_time.replace(
            hour=self.MARKET_OPEN_HOUR,
            minute=self.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        market_close = check_time.replace(
            hour=self.MARKET_CLOSE_HOUR,
            minute=self.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )

        # Check if current time is within market hours
        return market_open <= check_time <= market_close

    def get_market_data_type(self, check_time: datetime | None = None) -> str:
        """Get the appropriate market data type based on current time.

        Returns "LIVE" during market hours and "FROZEN" outside market hours.

        Args:
            check_time: Optional datetime to check. If None, uses current time.

        Returns:
            "LIVE" if within market hours, "FROZEN" otherwise.
        """
        if self.is_market_hours(check_time):
            return "LIVE"
        return "FROZEN"

    def get_next_market_open(self, check_time: datetime | None = None) -> datetime:
        """Get the datetime of the next market open.

        Args:
            check_time: Optional datetime to check. If None, uses current time.

        Returns:
            Datetime of the next market open in the configured timezone.
        """
        if check_time is None:
            check_time = self._get_current_time()
        else:
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=self.timezone)
            else:
                check_time = check_time.astimezone(self.timezone)

        # Create market open time for the current day
        market_open = check_time.replace(
            hour=self.MARKET_OPEN_HOUR,
            minute=self.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )

        # If we're already past market open today, start checking from tomorrow
        if check_time >= market_open:
            check_time = check_time + timedelta(days=1)
            market_open = check_time.replace(
                hour=self.MARKET_OPEN_HOUR,
                minute=self.MARKET_OPEN_MINUTE,
                second=0,
                microsecond=0,
            )

        # Skip weekends (Saturday=5, Sunday=6)
        while market_open.weekday() >= 5:
            market_open = market_open + timedelta(days=1)
            market_open = market_open.replace(
                hour=self.MARKET_OPEN_HOUR,
                minute=self.MARKET_OPEN_MINUTE,
                second=0,
                microsecond=0,
            )

        return market_open

    def get_next_market_close(self, check_time: datetime | None = None) -> datetime:
        """Get the datetime of the next market close.

        Args:
            check_time: Optional datetime to check. If None, uses current time.

        Returns:
            Datetime of the next market close in the configured timezone.
        """
        if check_time is None:
            check_time = self._get_current_time()
        else:
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=self.timezone)
            else:
                check_time = check_time.astimezone(self.timezone)

        # Create market close time for the current day
        market_close = check_time.replace(
            hour=self.MARKET_CLOSE_HOUR,
            minute=self.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )

        # If we're already past market close today, start checking from tomorrow
        if check_time >= market_close:
            check_time = check_time + timedelta(days=1)
            market_close = check_time.replace(
                hour=self.MARKET_CLOSE_HOUR,
                minute=self.MARKET_CLOSE_MINUTE,
                second=0,
                microsecond=0,
            )

        # Skip weekends (Saturday=5, Sunday=6)
        while market_close.weekday() >= 5:
            market_close = market_close + timedelta(days=1)
            market_close = market_close.replace(
                hour=self.MARKET_CLOSE_HOUR,
                minute=self.MARKET_CLOSE_MINUTE,
                second=0,
                microsecond=0,
            )

        return market_close


__all__ = ["MarketHoursChecker"]
