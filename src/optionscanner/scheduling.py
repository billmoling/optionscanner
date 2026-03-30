"""Utility helpers for computing scheduler run times."""

from __future__ import annotations

from datetime import datetime, timedelta, time as dt_time
from typing import Any, Dict, List

from loguru import logger

DEFAULT_SCHEDULE_TIME = "07:00"


def parse_schedule_times(schedule_config: Dict[str, Any]) -> List[dt_time]:
    """Parse schedule configuration into a sorted list of ``datetime.time`` entries."""

    raw_times = schedule_config.get("times")
    time_strings: List[str]

    if isinstance(raw_times, (list, tuple)):
        time_strings = [str(value) for value in raw_times if value is not None]
    elif raw_times is not None:
        time_strings = [str(raw_times)]
    else:
        time_strings = [str(schedule_config.get("time", DEFAULT_SCHEDULE_TIME))]

    parsed_times: List[dt_time] = []
    for time_str in time_strings:
        try:
            parsed_times.append(datetime.strptime(time_str, "%H:%M").time())
        except ValueError:
            logger.warning("Invalid schedule time '{time}', skipping", time=time_str)

    if not parsed_times:
        logger.warning(
            "No valid schedule times configured; defaulting to {default}",
            default=DEFAULT_SCHEDULE_TIME,
        )
        parsed_times = [datetime.strptime(DEFAULT_SCHEDULE_TIME, "%H:%M").time()]

    parsed_times.sort()
    return parsed_times


def compute_next_run(now: datetime, scheduled_times: List[dt_time]) -> datetime:
    """Compute the next scheduled run time after *now* for the configured times."""

    candidates: List[datetime] = []
    for scheduled_time in scheduled_times:
        candidate = now.replace(
            hour=scheduled_time.hour,
            minute=scheduled_time.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        candidates.append(candidate)

    return min(candidates)


__all__ = ["compute_next_run", "parse_schedule_times", "DEFAULT_SCHEDULE_TIME"]
