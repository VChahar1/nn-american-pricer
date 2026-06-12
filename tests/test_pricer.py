"""Tests for the BS02 pricer and the NN workflow."""
from __future__ import annotations

import numpy as np
import pytest

from src.bjerksund_stensland import american_put_bs02, bs_european_put


def test_american_put_exceeds_european():
    am = american_put_bs02(100, 100, 1.0, 0.05, 0.2)
    eu = bs_european_put(100, 100, 1.0, 0.05, 0.2)
    assert am >= eu - 0.01


def test_deep_itm_put_near_intrinsic():
    am = american_put_bs02(50, 100, 1.0, 0.05, 0.2)
    intrinsic = 100 - 50
    assert am >= intrinsic - 0.5


def test_deep_otm_put_near_zero():
    am = american_put_bs02(200, 100, 1.0, 0.05, 0.2)
    assert am < 0.5


def test_price_increases_with_sigma():
    p1 = american_put_bs02(100, 100, 1.0, 0.05, 0.15)
    p2 = american_put_bs02(100, 100, 1.0, 0.05, 0.30)
    assert p2 > p1


def test_price_increases_with_T():
    p1 = american_put_bs02(100, 100, 0.5, 0.05, 0.2)
    p2 = american_put_bs02(100, 100, 2.0, 0.05, 0.2)
    assert p2 > p1


def test_expired_option_equals_intrinsic():
    am = american_put_bs02(90, 100, 0.0, 0.05, 0.2)
    assert abs(am - 10.0) < 0.01


def test_zero_vol_equals_discounted_intrinsic():
    am = american_put_bs02(90, 100, 1.0, 0.05, 0.0)
    expected = max(100 * np.exp(-0.05) - 90, 0.0)
    assert abs(am - expected) < 0.5