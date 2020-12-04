#!/usr/bin/env python
"""Tests for `pomdp_belief_tracking` package."""


import pytest  # type: ignore


@pytest.mark.parametrize("truth", [(True, True)])
def test_this_is_working(truth):
    """If this fails I can not help you"""
    assert truth
