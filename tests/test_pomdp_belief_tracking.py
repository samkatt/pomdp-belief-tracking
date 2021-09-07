#!/usr/bin/env python
"""Tests for :mod:`pomdp_belief_tracking`"""


import pytest  # type: ignore


@pytest.mark.parametrize("truth", [(True, True)])
def test_this_is_working(truth):
    """If this fails I can not help you"""
    assert truth


if __name__ == "__main__":
    pytest.main([__file__])
