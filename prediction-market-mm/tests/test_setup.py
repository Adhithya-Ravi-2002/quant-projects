"""Sanity check that the test suite runs and the package imports correctly."""

import prediction_market_mm


def test_package_imports():
    assert prediction_market_mm is not None
