#!/usr/bin/env python
"""Tests for :class:`pomdp_belief_tracking.pf` module."""

import random
from functools import partial
from operator import eq

import pytest  # type: ignore

from pomdp_belief_tracking.pf.importance_sampling import (
    general_importance_sample,
    resample,
)
from pomdp_belief_tracking.pf.particle_filter import (
    Particle,
    ParticleFilter,
    apply,
    effective_sample_size,
)
from pomdp_belief_tracking.pf.rejection_sampling import (
    general_rejection_sample,
    have_sampled_enough,
)


def test_pf_data_model():
    """Tests :class:`~pomdp_belief_tracking.pf.ParticleFilter` container functions"""
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
    """Tests :meth:`~pomdp_belief_tracking.pf.from_distribution`"""

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
    """Tests :meth:`~pomdp_belief_tracking.pf.from_particles`"""
    particles = [
        Particle(4, 3.0),
        Particle(2, 9.0),
        Particle(4, 3.0),
    ]

    pf = ParticleFilter.from_particles(particles)

    assert pf.particles == [
        Particle(4, 0.2),
        Particle(2, 0.6),
        Particle(4, 0.2),
    ]


@pytest.mark.parametrize(
    "particles,total_weight",
    [
        ([Particle(2, 10.0)], 10.0),
        ([Particle(2, 3.2)], 3.2),
        (
            [Particle(2, 10.0), Particle(False, 30.0)],
            40.0,
        ),
        ([Particle(2, 10.0), Particle(2, 30.0)], 40.0),
    ],
)
def test_pf_total_weight(particles, total_weight):
    """Tests :func:`~pomdp_belief_tracking.pf.ParticleFilter.total_weight`"""
    assert ParticleFilter.total_weight(particles) == total_weight


def test_pf_call():
    """Tests :meth:`~pomdp_belief_tracking.pf.ParticleFilter.__call__` to sample"""

    pf = ParticleFilter([0, 0, 0])
    assert pf() == 0

    pf = ParticleFilter.from_particles([Particle(10, 100000.0), Particle(-1, 1.0)])
    assert pf() == 10

    pf = ParticleFilter([True, 0])
    samples = [pf() for _ in range(100)]
    assert True in samples
    assert 0 in samples

    pf = ParticleFilter.from_particles(
        [
            Particle(10, 1),
            Particle(-1, 1),
            Particle(10, 1),
        ]
    )
    samples = [pf() for _ in range(100)]
    assert 10 in samples
    assert -1 in samples
    assert samples.count(10) > samples.count(-1)


@pytest.mark.parametrize(
    "weights,neff",
    [
        ([3.4567], 1.0),
        ([0.5, 0.5], 2.0),
        ([10, 10], 2.0),
        ([3.5, 3.5, 35], 1.411764),
    ],
)
def test_effective_sample_size(weights, neff):
    """Tests :func:`~pomdp_belief_tracking.pf.importance_sampling.resample`"""
    assert effective_sample_size(weights) == pytest.approx(neff)


@pytest.mark.parametrize(
    "num_desired,num_accepted,expected",
    [(5, 4, False), (5, 5, True), (10, 1, False), (10, 100, True)],
)
def test_have_sampled_enough(num_desired, num_accepted, expected):
    """Tests :func:`~pomdp_belief_tracking.pf.have_sampled_enough`"""
    assert have_sampled_enough(num_desired, {"num_accepted": num_accepted}) == expected


def test_have_sampled_edge_cases():
    """Tests :func:`~pomdp_belief_tracking.pf.have_sampled_enough` edge cases"""

    with pytest.raises(AssertionError):
        have_sampled_enough(-1, {"num_accepted": 10})

    with pytest.raises(AssertionError):
        have_sampled_enough(0, {"num_accepted": 10})

    with pytest.raises(AssertionError):
        have_sampled_enough(10, {"num_accepted": -1})


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
        general_rejection_sample(
            partial(have_sampled_enough, 0),
            proposal,
            accept_function,
            distr,
            process_accepted,
        )

    desired_samples = 8
    samples, info = general_rejection_sample(
        partial(have_sampled_enough, desired_samples),
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

    samples, _ = general_rejection_sample(
        partial(have_sampled_enough, 20),
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

    samples, _ = general_rejection_sample(
        partial(have_sampled_enough, 20),
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

    samples, info = general_rejection_sample(
        partial(have_sampled_enough, 20),
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
        ([Particle(10, 1.0)], eq, 10, 1.0),
        (
            [Particle(10, 1.0), Particle(10, 1.0)],
            eq,
            10,
            1.0,
        ),
        (
            [Particle(10, 1.0), Particle(10, 1.0)],
            eq,
            5,
            0.0,
        ),
        (
            [Particle(10, 1.0), Particle(5, 1.0)],
            eq,
            10,
            0.5,
        ),
        (
            [Particle(10, 1.0), Particle(5, 1.0)],
            lambda o1, o2: True,
            5,
            1.0,
        ),
        (
            [Particle(10, 1.0), Particle(5, 1.0)],
            lambda o1, o2: False,
            5,
            0.0,
        ),
    ],
)
def test_pf_probability_of(particles, equality, particle, prob):
    """Tests :meth:`~pomdp_belief_tracking.pf.ParticleFilter.probability_of`"""
    assert (
        ParticleFilter.from_particles(particles).probability_of(particle, equality)
        == prob
    )


def test_resample():
    """Tests :func:`~pomdp_belief_tracking.pf.importance_sampling.resample`"""

    def sample_zero():
        return 0

    particles = resample(sample_zero, 5)

    assert len(particles) == 5
    assert random.choice(particles).state == 0

    def sample_bool():
        return random.choice([False, True])

    particles = resample(sample_bool, 100)

    assert len(particles) == 100

    pf = ParticleFilter.from_particles(particles)
    assert len(particles) == 100
    assert 0.4 < pf.probability_of(False) < 0.6
    assert 0.4 < pf.probability_of(True) < 0.6


def test_general_importance_sampling():
    """Tests :func:`~pomdp_belief_tracking.pf.rejection_sampling.general_rejection_sample`"""

    def prop_plus2(s, _):
        return s + 2, {"weight_should_be": 1 / s}

    def weight_1(_, ctx, __):
        return ctx["weight_should_be"]

    particles = ParticleFilter([10, 20, 10])

    pf, _ = general_importance_sample(prop_plus2, weight_1, particles)
    assert len(set([12, 22]) - set(p.state for p in pf)) == 0
    assert len(pf) == 3
    assert all(p.weight == pytest.approx(2 / 5) for p in pf if p.state == 12), pf
    assert all(p.weight == pytest.approx(1 / 5) for p in pf if p.state == 22), pf


@pytest.mark.parametrize(
    "particles,f,expected_new_particles",
    [
        ([0], lambda _: 1, [1]),
        ([0, 1, 2, 3], lambda _: 1, [1, 1, 1, 1]),
        ([0, 1, 2, 3], lambda x: -x, [0, -1, -2, -3]),
    ],
)
def test_pf_apply(particles, f, expected_new_particles):
    pf = ParticleFilter(particles)
    new_pf = apply(f, pf)
    assert [p.state for p in new_pf] == expected_new_particles
    assert [p.state for p in pf] == particles
