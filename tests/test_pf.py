#!/usr/bin/env python
"""Tests for :class:`pomdp_belief_tracking.pf` module."""

import random
from functools import partial
from operator import eq

import pytest  # type: ignore

from pomdp_belief_tracking.pf import particle_filter, rejection_sampling


def test_pf_data_model():
    """Tests :class:`~pomdp_belief_tracking.pf.ParticleFilter` container functions"""
    pf = particle_filter.ParticleFilter([0, 0, 0])

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

    pf = particle_filter.ParticleFilter([True, 0])

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

    pf = particle_filter.ParticleFilter([])

    assert len(pf) == 0
    assert 0 not in pf
    assert 3 not in pf
    assert repr(pf) == "ParticleFilter([])"

    for _ in pf:
        assert False


def test_pf_from_distribution():
    """Tests :meth:`~pomdp_belief_tracking.pf.from_distribution`"""

    pf = particle_filter.ParticleFilter.from_distribution(lambda: 100, 5)

    assert pf.particles == [
        particle_filter.Particle(100, 0.2),
        particle_filter.Particle(100, 0.2),
        particle_filter.Particle(100, 0.2),
        particle_filter.Particle(100, 0.2),
        particle_filter.Particle(100, 0.2),
    ]

    pf = particle_filter.ParticleFilter.from_distribution(
        lambda: random.choice([3, -5]), 20
    )

    assert 3 in pf
    assert -5 in pf
    assert 2 not in pf


def test_pf_from_particles():
    """Tests :meth:`~pomdp_belief_tracking.pf.from_particles`"""
    particles = [
        particle_filter.Particle(4, 3.0),
        particle_filter.Particle(2, 9.0),
        particle_filter.Particle(4, 3.0),
    ]

    pf = particle_filter.ParticleFilter.from_particles(particles)

    assert pf.particles == [
        particle_filter.Particle(4, 0.2),
        particle_filter.Particle(2, 0.6),
        particle_filter.Particle(4, 0.2),
    ]


@pytest.mark.parametrize(
    "particles,total_weight",
    [
        ([particle_filter.Particle(2, 10.0)], 10.0),
        ([particle_filter.Particle(2, 3.2)], 3.2),
        (
            [particle_filter.Particle(2, 10.0), particle_filter.Particle(False, 30.0)],
            40.0,
        ),
        ([particle_filter.Particle(2, 10.0), particle_filter.Particle(2, 30.0)], 40.0),
    ],
)
def test_pf_total_weight(particles, total_weight):
    """Tests :func:`~pomdp_belief_tracking.pf.ParticleFilter.total_weight`"""
    assert particle_filter.ParticleFilter.total_weight(particles) == total_weight


def test_pf_call():
    """Tests :meth:`~pomdp_belief_tracking.pf.ParticleFilter.__call__` to sample"""

    pf = particle_filter.ParticleFilter([0, 0, 0])
    assert pf() == 0

    pf = particle_filter.ParticleFilter.from_particles(
        [particle_filter.Particle(10, 100000.0), particle_filter.Particle(-1, 1.0)]
    )
    assert pf() == 10

    pf = particle_filter.ParticleFilter([True, 0])
    samples = [pf() for _ in range(100)]
    assert True in samples
    assert 0 in samples

    pf = particle_filter.ParticleFilter.from_particles(
        [
            particle_filter.Particle(10, 1),
            particle_filter.Particle(-1, 1),
            particle_filter.Particle(10, 1),
        ]
    )
    samples = [pf() for _ in range(100)]
    assert 10 in samples
    assert -1 in samples
    assert samples.count(10) > samples.count(-1)


@pytest.mark.parametrize(
    "num_desired,num_accepted,expected",
    [(5, 4, False), (5, 5, True), (10, 1, False), (10, 100, True)],
)
def test_have_sampled_enough(num_desired, num_accepted, expected):
    """Tests :func:`~pomdp_belief_tracking.pf.have_sampled_enough`"""
    assert (
        rejection_sampling.have_sampled_enough(
            num_desired, {"num_accepted": num_accepted}
        )
        == expected
    )


def test_have_sampled_edge_cases():
    """Tests :func:`~pomdp_belief_tracking.pf.have_sampled_enough` edge cases"""

    with pytest.raises(AssertionError):
        rejection_sampling.have_sampled_enough(-1, {"num_accepted": 10})

    with pytest.raises(AssertionError):
        rejection_sampling.have_sampled_enough(0, {"num_accepted": 10})

    with pytest.raises(AssertionError):
        rejection_sampling.have_sampled_enough(10, {"num_accepted": -1})


def test_general_rejection_sample():
    """Tests :func:`~pomdp_belief_tracking.pf.general_rejection_sample`"""

    def distr():
        return random.choice([[10], [3]])

    def proposal(x, _):
        x[0] += 2
        return (x, random.choice([True, False]))

    def accept_function(_, ctx, __):
        return ctx

    def process_accepted(x, _, __):
        x[0] -= 1
        return x

    with pytest.raises(AssertionError):
        rejection_sampling.general_rejection_sample(
            partial(rejection_sampling.have_sampled_enough, 0),
            proposal,
            accept_function,
            distr,
            process_accepted,
        )

    desired_samples = 8
    samples, info = rejection_sampling.general_rejection_sample(
        partial(rejection_sampling.have_sampled_enough, desired_samples),
        proposal,
        accept_function,
        distr,
        process_accepted,
    )

    assert (
        len(samples) == desired_samples
    ), f"Expecting requested samples, not {len(samples)}"
    assert all(list(x[0] in [11, 4] for x in samples)), "samples should be incremented"
    assert (
        info["num_accepted"] == desired_samples
    ), f"Expecting to accurately report number of accepts, not {info['num_accepted']}"
    assert (
        info["iteration"] > desired_samples
    ), f"Expecting some particles to be rejected, not {info['iteration']}"

    start_samples = [[10], [3]]

    def distr_no_copy():
        return random.choice(start_samples)

    def process_rejected_reset(x, _, __):
        x[0] -= 2
        return x

    samples, _ = rejection_sampling.general_rejection_sample(
        partial(rejection_sampling.have_sampled_enough, 20),
        proposal,
        accept_function,
        distr_no_copy,
        process_accepted,
        process_rejected_reset,
    )

    assert len(samples) == 20, f"Expecting requested samples, not {len(samples)}"
    assert not all(list(x[0] in [11, 4] for x in samples)), "samples are modified"

    def process_accepted_copy(x, _, __):
        copy = [x[0]]
        # reset original
        x[0] -= 2
        return copy

    start_samples = [[10], [3]]

    samples, _ = rejection_sampling.general_rejection_sample(
        partial(rejection_sampling.have_sampled_enough, 20),
        proposal,
        accept_function,
        distr_no_copy,
        process_accepted_copy,
        process_rejected_reset,
    )

    assert len(samples) == 20
    assert all(list(x[0] in [12, 5] for x in samples)), ""

    def accept_all_function(_, __, ___):
        return True

    samples, info = rejection_sampling.general_rejection_sample(
        partial(rejection_sampling.have_sampled_enough, 20),
        proposal,
        accept_all_function,
        distr_no_copy,
        process_accepted_copy,
        process_rejected_reset,
    )

    assert len(samples) == 20
    assert info["num_accepted"] == 20
    assert info["iteration"] == 20


@pytest.mark.parametrize(
    "particles,equality,particle,prob",
    [
        ([particle_filter.Particle(10, 1.0)], eq, 10, 1.0),
        (
            [particle_filter.Particle(10, 1.0), particle_filter.Particle(10, 1.0)],
            eq,
            10,
            1.0,
        ),
        (
            [particle_filter.Particle(10, 1.0), particle_filter.Particle(10, 1.0)],
            eq,
            5,
            0.0,
        ),
        (
            [particle_filter.Particle(10, 1.0), particle_filter.Particle(5, 1.0)],
            eq,
            10,
            0.5,
        ),
        (
            [particle_filter.Particle(10, 1.0), particle_filter.Particle(5, 1.0)],
            lambda o1, o2: True,
            5,
            1.0,
        ),
        (
            [particle_filter.Particle(10, 1.0), particle_filter.Particle(5, 1.0)],
            lambda o1, o2: False,
            5,
            0.0,
        ),
    ],
)
def test_pf_probability_of(particles, equality, particle, prob):
    """Tests :meth:`~pomdp_belief_tracking.pf.ParticleFilter.probability_of`"""
    assert (
        particle_filter.ParticleFilter.from_particles(particles).probability_of(
            particle, equality
        )
        == prob
    )
