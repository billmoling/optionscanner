import sys
from pathlib import Path
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if "nautilus_trader" not in sys.modules:
    nautilus_trader = types.ModuleType("nautilus_trader")
    trading = types.ModuleType("nautilus_trader.trading")
    strategy = types.ModuleType("nautilus_trader.trading.strategy")

    class DummyStrategy:  # pragma: no cover - simple stub for tests
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
            """Accept arbitrary arguments."""

    strategy.Strategy = DummyStrategy
    trading.strategy = strategy
    nautilus_trader.trading = trading
    sys.modules["nautilus_trader"] = nautilus_trader
    sys.modules["nautilus_trader.trading"] = trading
    sys.modules["nautilus_trader.trading.strategy"] = strategy
