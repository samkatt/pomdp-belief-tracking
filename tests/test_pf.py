#!/usr/bin/env python
"""Tests for :py:class:`pomdp_belief_tracking.pf` module."""


import random
from operator import eq

import pytest  # type: ignore

from pomdp_belief_tracking.pf import Particle, ParticleFilter, general_rejection_sample


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


def test_general_rejection_sample():
    """Tests :py:funct:`~pomdp_belief_tracking.pf.general_rejection_sample`"""

    def distr():
        return random.choice([[10], [3]])

    def proposal(x):
        x[0] += 2
        return (x, random.choice([True, False]))

    def accept_function(_, ctx):
        return ctx

    def process_accepted(x, _):
        x[0] -= 1
        return x

    with pytest.raises(AssertionError):
        general_rejection_sample(proposal, accept_function, distr, 0, process_accepted)

    samples = general_rejection_sample(
        proposal, accept_function, distr, 4, process_accepted
    )

    assert len(samples) == 4, f"Expecting requested samples, not {len(samples)}"
    assert all(list(x[0] in [11, 4] for x in samples)), "samples should be incremented"

    start_samples = [[10], [3]]

    def distr_no_copy():
        return random.choice(start_samples)

    def process_rejected_reset(x, _):
        x[0] -= 2
        return x

    samples = general_rejection_sample(
        proposal,
        accept_function,
        distr_no_copy,
        20,
        process_accepted,
        process_rejected_reset,
    )

    assert len(samples) == 20, f"Expecting requested samples, not {len(samples)}"
    assert not all(list(x[0] in [11, 4] for x in samples)), "samples are modified"

    def process_accepted_copy(x, _):
        copy = [x[0]]
        # reset original
        x[0] -= 2
        return copy

    start_samples = [[10], [3]]

    samples = general_rejection_sample(
        proposal,
        accept_function,
        distr_no_copy,
        20,
        process_accepted_copy,
        process_rejected_reset,
    )

    assert len(samples) == 20
    assert all(list(x[0] in [12, 5] for x in samples)), ""


@pytest.mark.parametrize(
    "particles,equality,particle,prob",
    [
        ([Particle(10, 1.0)], eq, 10, 1.0),
        ([Particle(10, 1.0), Particle(10, 1.0)], eq, 10, 1.0),
        ([Particle(10, 1.0), Particle(10, 1.0)], eq, 5, .0),
        ([Particle(10, 1.0), Particle(5, 1.0)], eq, 10, 0.5),
        ([Particle(10, 1.0), Particle(5, 1.0)], lambda o1, o2: True, 5, 1.0),
        ([Particle(10, 1.0), Particle(5, 1.0)], lambda o1, o2: False, 5, 0.0),
    ],
)
def test_pf_probability_of(particles, equality, particle, prob):
    """Tests :py:meth:`~pomdp_belief_tracking.pf.ParticleFilter.probability_of`"""
    assert (
        ParticleFilter.from_particles(particles).probability_of(particle, equality)
        == prob
    )
