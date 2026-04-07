# Auto Market Data Type Selection Plan

**Date:** 2026-04-06
**Author:** Claude Code
**Status:** Approved for implementation

## Overview

Implement automatic market data type selection that switches between LIVE and FROZEN based on market hours. This eliminates the need for manual configuration changes between trading sessions.

## Market Hours

- **Regular Trading Hours:** 6:30 AM - 1:00 PM Pacific Time
- **Days:** Weekdays only (Monday-Friday)
- **Timezone:** `America/Los_Angeles`

## Behavior

| Condition | Market Data Type |
|-----------|------------------|
| Weekday 6:30 AM - 1:00 PM PT | LIVE |
| Outside market hours | FROZEN |
| Weekends | FROZEN |

## Tasks

### Task 1: Create `market_hours.py` Module

**File:** `src/optionscanner/market_hours.py`

**Components:**
- `MarketHoursChecker` class with:
  - `is_market_hours()` method - returns True if currently within market hours
  - `get_market_data_type()` method - returns "LIVE" or "FROZEN" based on current time
  - `get_next_market_open()` method - returns datetime of next market open
  - `get_next_market_close()` method - returns datetime of next market close
- Use `zoneinfo.ZoneInfo("America/Los_Angeles")` for timezone handling

**Tests:** `tests/test_market_hours.py`
- Test `is_market_hours()` during/after hours
- Test `get_market_data_type()` returns correct value
- Test weekend handling
- Test edge cases (exactly at 6:30 AM, exactly at 1:00 PM)

### Task 2: Update `main.py` to Support AUTO Mode

**Changes:**
- Add "AUTO" to the `--market-data` argument choices
- When AUTO is selected, use `MarketHoursChecker.get_market_data_type()` to determine the actual type
- Log the resolved market data type at startup

### Task 3: Update `test_ibkr_integration.py` to Support AUTO Mode

**Changes:**
- Update the test to handle AUTO mode by resolving to actual type before use
- Import and use `MarketHoursChecker`

### Task 4: Update Documentation

**Files:**
- `.env.example` - Add comment explaining AUTO mode
- `README.md` - Document the new AUTO option

## Acceptance Criteria

1. [ ] `MarketHoursChecker` class exists with all required methods
2. [ ] All tests pass for market_hours module
3. [ ] `--market-data AUTO` works in main.py
4. [ ] Integration test supports AUTO mode
5. [ ] Documentation updated
6. [ ] `.env` file updated to use `IBKR_MARKET_DATA_TYPE=AUTO`

## Implementation Order

1. Create `market_hours.py` with tests (TDD)
2. Update `main.py`
3. Update integration tests
4. Update documentation
5. Update `.env` file
