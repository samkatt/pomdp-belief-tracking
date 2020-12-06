#!/usr/bin/env python
"""Tests for :py:class:`pomdp_belief_tracking.pf` module."""


import random

import pytest  # type: ignore

from pomdp_belief_tracking.pf import Particle, ParticleFilter


def test_pf_data_model():
    """Tests :py:class:`~pomdp_belief_tracking.pf.ParticleFilter` container functions"""
    pf = ParticleFilter([0, 0, 0])

    assert len(pf) == 3
    assert 0 in pf
    assert True not in pf
    assert (
        repr(pf) == "ParticleFilter([Particle(state=0, weight=0.3333333333333333), "
        "Particle(state=0, weight=0.3333333333333333), "
        "Particle(state=0, weight=0.3333333333333333)])"
    )

    for state, weight in pf:
        assert state == 0
        assert weight == 1 / 3

    pf = ParticleFilter([True, 0])

    assert len(pf) == 2
    assert 0 in pf
    assert True in pf
    assert 3 not in pf
    assert (
        repr(pf)
        == "ParticleFilter([Particle(state=True, weight=0.5), Particle(state=0, weight=0.5)])"
    )

    for state, weight in pf:
        assert state or state == 0
        assert weight == 0.5

    pf = ParticleFilter([])

    assert len(pf) == 0
    assert 0 not in pf
    assert 3 not in pf
    assert repr(pf) == "ParticleFilter([])"

    for _ in pf:
        assert False


def test_pf_from_distribution():
    """Tests :py:meth:`~pomdp_belief_tracking.pf.from_distribution`"""

    pf = ParticleFilter.from_distribution(lambda: 100, 5)

    assert pf.particles == [
        Particle(100, 0.2),
        Particle(100, 0.2),
        Particle(100, 0.2),
        Particle(100, 0.2),
        Particle(100, 0.2),
    ]

    pf = ParticleFilter.from_distribution(lambda: random.choice([3, -5]), 20)

    assert 3 in pf
    assert -5 in pf
    assert 2 not in pf


def test_pf_from_particles():
    """Tests :py:meth:`~pomdp_belief_tracking.pf.from_particles`"""
    particles = [Particle(4, 3.0), Particle(2, 9.0), Particle(4, 3.0)]

    pf = ParticleFilter.from_particles(particles)

    assert pf.particles == [Particle(4, 0.2), Particle(2, 0.6), Particle(4, 0.2)]


@pytest.mark.parametrize(
    "particles,total_weight",
    [
        ([Particle(2, 10.0)], 10.0),
        ([Particle(2, 3.2)], 3.2),
        ([Particle(2, 10.0), Particle(False, 30.0)], 40.0),
        ([Particle(2, 10.0), Particle(2, 30.0)], 40.0),
    ],
)
def test_pf_total_weight(particles, total_weight):
    """Tests :py:func:`~pomdp_belief_tracking.pf.ParticleFilter.total_weight`"""
    assert ParticleFilter.total_weight(particles) == total_weight


def test_pf_call():
    """Tests :py:meth:`~pomdp_belief_tracking.pf.ParticleFilter.__call__` to sample"""

    pf = ParticleFilter([0, 0, 0])
    assert pf() == 0

    pf = ParticleFilter.from_particles([Particle(10, 100000.0), Particle(-1, 1.0)])
    assert pf() == 10

    pf = ParticleFilter([True, 0])
    samples = [pf() for _ in range(100)]
    assert True in samples
    assert 0 in samples

    pf = ParticleFilter.from_particles(
        [Particle(10, 1), Particle(-1, 1), Particle(10, 1)]
    )
    samples = [pf() for _ in range(100)]
    assert 10 in samples
    assert -1 in samples
    assert samples.count(10) > samples.count(-1)
