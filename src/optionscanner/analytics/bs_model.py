"""Black-Scholes option pricing model for probability calculations."""
from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm


class BSModel:
    """Black-Scholes option pricing model.

    Provides methods for calculating option prices and probabilities
    using the Black-Scholes formula.
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        """Initialize the Black-Scholes model.

        Args:
            risk_free_rate: Annual risk-free interest rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_d1_d2(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
    ) -> tuple[float, float]:
        """Calculate d1 and d2 from the Black-Scholes formula.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (annualized)

        Returns:
            Tuple of (d1, d2) values
        """
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        d1 = (math.log(S / K) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def calculate_option_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: Literal["CALL", "PUT"],
    ) -> float:
        """Calculate theoretical option price using Black-Scholes formula.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (annualized)
            option_type: "CALL" or "PUT"

        Returns:
            Theoretical option price
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)

        if option_type == "CALL":
            price = S * norm.cdf(d1) - K * math.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:  # PUT
            price = K * math.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.0)

    def calculate_otm_probability(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: Literal["CALL", "PUT"],
    ) -> float:
        """Calculate probability of option expiring OTM (out of the money).

        For a put option, this is the probability that S > K at expiration.
        For a call option, this is the probability that S < K at expiration.

        Uses risk-neutral probability based on d2.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (annualized)
            option_type: "CALL" or "PUT"

        Returns:
            Probability of expiring OTM (0.0 to 1.0)
        """
        if T <= 0 or sigma <= 0:
            return 0.5

        _, d2 = self.calculate_d1_d2(S, K, T, sigma)

        if option_type == "PUT":
            # Put expires OTM if S > K, probability is N(d2)
            return norm.cdf(d2)
        else:  # CALL
            # Call expires OTM if S < K, probability is N(-d2)
            return norm.cdf(-d2)
