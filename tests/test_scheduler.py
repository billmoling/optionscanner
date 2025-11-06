import unittest
from datetime import datetime

from zoneinfo import ZoneInfo

from scheduling import compute_next_run, parse_schedule_times


class SchedulerUtilsTests(unittest.TestCase):
    def test_parse_schedule_times_handles_multiple_entries(self):
        config = {"times": ["10:00", "06:30", "12:15"]}

        parsed = parse_schedule_times(config)

        formatted = [time.strftime("%H:%M") for time in parsed]
        self.assertEqual(formatted, ["06:30", "10:00", "12:15"])

    def test_parse_schedule_times_falls_back_to_default(self):
        config = {"times": ["invalid"]}

        parsed = parse_schedule_times(config)

        self.assertEqual([time.strftime("%H:%M") for time in parsed], ["07:00"])

    def test_compute_next_run_selects_same_day_future_time(self):
        tz = ZoneInfo("America/Los_Angeles")
        now = datetime(2024, 5, 1, 7, 0, tzinfo=tz)
        schedule = parse_schedule_times({"times": ["06:30", "10:00", "12:15"]})

        next_run = compute_next_run(now, schedule)

        self.assertEqual(next_run, datetime(2024, 5, 1, 10, 0, tzinfo=tz))

    def test_compute_next_run_rolls_to_next_day_when_needed(self):
        tz = ZoneInfo("America/Los_Angeles")
        now = datetime(2024, 5, 1, 13, 0, tzinfo=tz)
        schedule = parse_schedule_times({"times": ["06:30", "10:00", "12:15"]})

        next_run = compute_next_run(now, schedule)

        self.assertEqual(next_run, datetime(2024, 5, 2, 6, 30, tzinfo=tz))


if __name__ == "__main__":
    unittest.main()
