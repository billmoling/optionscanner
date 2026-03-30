"""AI-powered economic calendar fetcher using Google Gemini and web search."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

from optionscanner.economic_calendar import EconomicEvent, EconomicEventTracker


class EconomicCalendarAIFetcher:
    """Fetches and parses economic calendar data using Google Gemini AI.

    Uses Google's AI to:
    1. Search for upcoming economic events from web sources
    2. Parse and structure event data into JSON format
    3. Update the economic_events.json file

    This avoids manual updates and API subscriptions for economic data.
    """

    # High-impact event types to prioritize
    HIGH_IMPACT_TYPES = {"FOMC", "CPI", "NFP", "PCE", "GDP", "Jackson Hole", "Fed Chair Speech"}

    # JSON format prompt for Gemini
    EVENT_FORMAT_PROMPT = """
Return the economic calendar data as a JSON array with this exact structure:

[
  {
    "event_id": "unique_id_like_cpi_2026_apr",
    "date": "YYYY-MM-DD",
    "event_type": "CPI|FOMC|NFP|PCE|GDP|PPI|Retail Sales|Initial Jobless Claims",
    "impact": "HIGH|MEDIUM|LOW",
    "description": "Brief description like 'CPI YoY' or 'FOMC Rate Decision'",
    "country": "US",
    "forecast": "Consensus forecast if available (e.g., '2.8%', '185K', '4.50%')",
    "previous": "Previous period value if available"
  }
]

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no explanations.
Include all high-impact US economic events for the specified time period.
Event types to include:
- FOMC: Federal Reserve rate decisions and meetings
- CPI: Consumer Price Index (inflation)
- NFP: Non-Farm Payrolls (employment, usually first Friday of month)
- PCE: Personal Consumption Expenditures (Fed's preferred inflation gauge)
- GDP: Gross Domestic Product
- PPI: Producer Price Index
- Retail Sales
- Initial Jobless Claims (weekly, every Thursday)
"""

    def __init__(
        self,
        events_file: Optional[Path] = None,
        lookback_days: int = 7,
        lookahead_days: int = 90,
    ) -> None:
        """Initialize the AI fetcher.

        Args:
            events_file: Path to economic_events.json (default: data/economic_events.json)
            lookback_days: Include events from past week (for results)
            lookahead_days: How far ahead to fetch events (default: 90 days)
        """
        self._events_file = events_file or Path("data") / "economic_events.json"
        self._lookback_days = lookback_days
        self._lookahead_days = lookahead_days
        self._tracker = EconomicEventTracker(events_file=self._events_file)

        # Ensure directory exists
        self._events_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> Any:
        """Get Google Gemini client."""
        if genai is None:
            raise ImportError(
                "google-genai package not installed. Install with: pip install google-genai"
            )

        # Try to get API key from environment
        import os
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        return genai.Client(api_key=api_key)

    def fetch_calendar(self) -> List[EconomicEvent]:
        """Fetch economic calendar data using AI.

        Returns:
            List of EconomicEvent objects
        """
        if genai is None:
            logger.warning("google-genai not installed, cannot fetch economic calendar")
            return []

        client = self._get_client()

        # Build the prompt with date context
        today = date.today()
        end_date = date.fromordinal(today.toordinal() + self._lookahead_days)
        start_date = date.fromordinal(today.toordinal() - self._lookback_days)

        prompt = f"""
Today is {today.isoformat()}.

Fetch upcoming US economic events from {start_date.isoformat()} to {end_date.isoformat()}.

{self.EVENT_FORMAT_PROMPT}

Focus on high-impact events that move markets:
1. FOMC rate decisions and Fed Chair Powell speeches
2. CPI/PCE inflation data
3. NFP employment reports
4. GDP releases
5. Other significant economic data

Search for the most recent consensus forecasts where available.
"""

        try:
            # Use Gemini with Google Search grounding
            logger.info("Fetching economic calendar via AI...")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for structured data
                    max_output_tokens=8192,
                ),
            )

            # Parse JSON from response
            events = self._parse_events(response.text)
            logger.info("Fetched {count} economic events", count=len(events))
            return events

        except Exception as exc:
            logger.exception("Failed to fetch economic calendar | error={error}", error=exc)
            return []

    def _parse_events(self, response_text: str) -> List[EconomicEvent]:
        """Parse JSON response into EconomicEvent objects."""
        events = []

        # Clean up response - remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON response | error={error}", error=exc)
            # Try to extract JSON from text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not extract JSON from response")
                    return events
            else:
                return events

        if not isinstance(data, list):
            logger.warning("Expected JSON array, got {type}", type=type(data).__name__)
            return events

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                event = self._parse_event_item(item)
                if event:
                    events.append(event)
            except Exception as exc:
                logger.debug("Failed to parse event item | item={item} error={error}", item=item, error=exc)

        # Sort by date
        events.sort(key=lambda e: e.date)
        return events

    def _parse_event_item(self, item: Dict[str, Any]) -> Optional[EconomicEvent]:
        """Parse a single event item into EconomicEvent."""
        # Required fields
        event_id = str(item.get("event_id", ""))
        date_str = str(item.get("date", ""))
        event_type = str(item.get("event_type", ""))
        description = str(item.get("description", ""))

        if not event_id or not date_str or not event_type:
            return None

        # Parse date
        try:
            event_date = date.fromisoformat(date_str)
        except ValueError:
            logger.debug("Invalid date format | date={date}", date=date_str)
            return None

        # Impact level
        impact = str(item.get("impact", "MEDIUM")).upper()
        if impact not in ("LOW", "MEDIUM", "HIGH"):
            impact = "MEDIUM"

        # Country
        country = str(item.get("country", "US"))

        # Optional fields
        forecast = item.get("forecast")
        if forecast is not None:
            forecast = str(forecast)
        previous = item.get("previous")
        if previous is not None:
            previous = str(previous)

        return EconomicEvent(
            event_id=event_id,
            date=event_date,
            event_type=event_type.upper(),
            impact=impact,
            description=description,
            country=country,
            forecast=forecast,
            previous=previous,
        )

    def refresh_calendar(self) -> int:
        """Refresh the economic calendar file with new data.

        Returns:
            Number of events added/updated
        """
        logger.info("Refreshing economic calendar...")

        # Fetch new events
        new_events = self.fetch_calendar()
        if not new_events:
            logger.warning("No events fetched, keeping existing calendar")
            return 0

        # Load existing events
        existing_events = self._tracker.get_upcoming_events(window_days=365)
        existing_by_id = {e.event_id: e for e in existing_events}

        # Merge events (new events take precedence)
        merged_by_id = {e.event_id: e for e in existing_events}
        new_count = 0
        updated_count = 0

        for new_event in new_events:
            if new_event.event_id not in merged_by_id:
                merged_by_id[new_event.event_id] = new_event
                new_count += 1
                logger.info("Added event | type={type} date={date}", type=new_event.event_type, date=new_event.date)
            else:
                merged_by_id[new_event.event_id] = new_event
                updated_count += 1
                logger.debug("Updated event | type={type} date={date}", type=new_event.event_type, date=new_event.date)

        # Filter to only future events plus recent past (for results)
        today = date.today()
        cutoff = date.fromordinal(today.toordinal() - self._lookback_days)
        future_events = [e for e in merged_by_id.values() if e.date >= cutoff]

        # Save to file
        cache_data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "events": [
                {
                    "event_id": e.event_id,
                    "date": e.date.isoformat(),
                    "event_type": e.event_type,
                    "impact": e.impact,
                    "description": e.description,
                    "country": e.country,
                    "forecast": e.forecast,
                    "previous": e.previous,
                }
                for e in future_events
            ],
        }

        self._events_file.parent.mkdir(parents=True, exist_ok=True)
        self._events_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")

        logger.info(
            "Economic calendar refreshed | new={new} updated={updated} total={total}",
            new=new_count,
            updated=updated_count,
            total=len(future_events),
        )

        return len(future_events)

    def should_refresh(self) -> bool:
        """Check if calendar should be refreshed (not updated in 7+ days)."""
        try:
            if not self._events_file.exists():
                return True

            data = json.loads(self._events_file.read_text(encoding="utf-8"))
            last_updated_str = data.get("last_updated")
            if not last_updated_str:
                return True

            last_updated = datetime.fromisoformat(last_updated_str)
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)

            days_since = (datetime.now(timezone.utc) - last_updated).days
            return days_since >= 7

        except Exception:
            return True


__all__ = ["EconomicCalendarAIFetcher"]
