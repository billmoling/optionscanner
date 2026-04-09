"""Unit tests for Black-Scholes option pricing model."""
from __future__ import annotations

import pytest

from optionscanner.analytics.bs_model import BSModel


class TestBSModelCalculations:
    """Test Black-Scholes calculation accuracy."""

    def test_d1_d2_calculation(self) -> None:
        """Verify d1 and d2 calculation against known values."""
        model = BSModel(risk_free_rate=0.05)

        # Test case: S=100, K=100, T=1.0, sigma=0.2
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
        d1, d2 = model.calculate_d1_d2(S, K, T, sigma)

        # Expected values (from standard BS calculators)
        assert d1 == pytest.approx(0.35, abs=0.01), f"d1={d1}, expected ~0.35"
        assert d2 == pytest.approx(0.15, abs=0.01), f"d2={d2}, expected ~0.15"

    def test_put_option_price(self) -> None:
        """Verify put option price calculation."""
        model = BSModel(risk_free_rate=0.05)

        # ATM put: S=100, K=100, T=1.0, sigma=0.2
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
        price = model.calculate_option_price(S, K, T, sigma, "PUT")

        # Expected: ~5.57 (standard BS calculator with r=0.05)
        assert price == pytest.approx(5.57, abs=0.1), f"Put price={price}, expected ~5.57"

    def test_call_option_price(self) -> None:
        """Verify call option price calculation."""
        model = BSModel(risk_free_rate=0.05)

        # ATM call: S=100, K=100, T=1.0, sigma=0.2
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
        price = model.calculate_option_price(S, K, T, sigma, "CALL")

        # Expected: ~10.4 (standard BS calculator reference)
        assert price == pytest.approx(10.4, abs=0.2), f"Call price={price}, expected ~10.4"

    def test_put_otm_probability(self) -> None:
        """Verify put OTM probability calculation."""
        model = BSModel(risk_free_rate=0.05)

        # OTM put: S=100, K=95 (below current price), T=0.1, sigma=0.3
        S, K, T, sigma = 100.0, 95.0, 0.1, 0.3
        prob = model.calculate_otm_probability(S, K, T, sigma, "PUT")

        # Put expires OTM if S > K, should be high probability
        assert prob > 0.7, f"Put OTM prob={prob}, expected >0.7"

    def test_call_otm_probability(self) -> None:
        """Verify call OTM probability calculation."""
        model = BSModel(risk_free_rate=0.05)

        # OTM call: S=100, K=105 (above current price), T=0.1, sigma=0.3
        S, K, T, sigma = 100.0, 105.0, 0.1, 0.3
        prob = model.calculate_otm_probability(S, K, T, sigma, "CALL")

        # Call expires OTM if S < K, should be moderate probability
        assert prob > 0.65, f"Call OTM prob={prob}, expected >0.65"

    def test_edge_case_zero_time(self) -> None:
        """Verify handling of zero time to expiration."""
        model = BSModel()

        S, K, sigma = 100.0, 100.0, 0.2
        price = model.calculate_option_price(S, K, 0.0, sigma, "PUT")
        prob = model.calculate_otm_probability(S, K, 0.0, sigma, "PUT")

        assert price == 0.0
        assert prob == 0.5

    def test_edge_case_zero_volatility(self) -> None:
        """Verify handling of zero volatility."""
        model = BSModel()

        S, K, T = 100.0, 100.0, 1.0
        price = model.calculate_option_price(S, K, T, 0.0, "PUT")
        prob = model.calculate_otm_probability(S, K, T, 0.0, "PUT")

        assert price == 0.0
        assert prob == 0.5
