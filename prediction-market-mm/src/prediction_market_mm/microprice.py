"""Volume-weighted micro-price estimator."""

from __future__ import annotations

from prediction_market_mm.orderbook import PriceLevel


def micro_price(best_bid: PriceLevel, best_ask: PriceLevel) -> float:
    """Size-weighted micro-price in the same integer-tick units as the inputs."""
    bid_size, ask_size = best_bid.size, best_ask.size
    # Negative sizes are rejected because they could make the denominator zero
    # (e.g. 5 and -5) or push the result outside [bid, ask].
    if bid_size < 0 or ask_size < 0:
        raise ValueError(f"sizes must be non-negative, got bid={bid_size}, ask={ask_size}")
    total = bid_size + ask_size
    if total == 0:
        raise ValueError("micro-price is undefined when both sizes are 0")
    # Crossed weighting: a larger bid queue means buying pressure, so the estimate
    # leans toward the ask (and vice versa).
    return (bid_size * best_ask.price + ask_size * best_bid.price) / total
