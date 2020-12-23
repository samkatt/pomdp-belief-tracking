#!/usr/bin/env python
"""Tests full belief updates or other larger components"""
import random
from typing import Tuple

from pomdp_belief_tracking.pf import particle_filter, rejection_sampling
from pomdp_belief_tracking.types import Action, Observation, State


class Tiger:
    """The Tiger POMDP environment"""

    L = 0
    R = 1
    H = 2

    H_REWARD = -1
    OPEN_CORRECT_REWARD = 10
    OPEN_INCORRECT_REWARD = -100

    @staticmethod
    def sample_observation(s: State) -> Observation:
        """85% hear tiger correctly"""
        if random.uniform(0, 1) < 0.85:
            return s
        return int(not s)

    @staticmethod
    def sim(s: State, a: Action) -> Tuple[State, Observation]:
        """Simulates the tiger dynamics"""

        if a == Tiger.H:
            o = Tiger.sample_observation(s)
            return s, o

        assert a in [Tiger.L, Tiger.R]

        o = random.choice([Tiger.L, Tiger.R])
        s = random.choice([Tiger.L, Tiger.R])

        return s, o


def uniform_tiger_belief():
    """Sampling returns 'left' and 'right' state equally"""
    return random.choice([Tiger.L, Tiger.R])


def tiger_left_belief():
    """Sampling returns 'left' state"""
    return Tiger.L


def tiger_right_belief():
    """Sampling returns 'right' state"""
    return Tiger.R


def test_rejection_sampling():
    """tests :func:`~online_pomdp_planning.mcts.create_POUCT` on Tiger"""

    belief_update = rejection_sampling.create_rejection_sampling(
        Tiger.sim,
        100,
        process_acpt=rejection_sampling.accept_noop,
        process_rej=rejection_sampling.reject_noop,
    )

    b, info = belief_update(uniform_tiger_belief, Tiger.H, Tiger.L)

    assert isinstance(b, particle_filter.ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) > b.probability_of(Tiger.R)
    assert info["num_accepted"] == 100
    assert 50 < info["iteration"] - info["num_accepted"] < 150

    b, info = belief_update(uniform_tiger_belief, Tiger.H, Tiger.R)

    assert isinstance(b, particle_filter.ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) < b.probability_of(Tiger.R)
    assert info["num_accepted"] == 100
    assert 50 < info["iteration"] - info["num_accepted"] < 150

    b, info = belief_update(tiger_right_belief, Tiger.H, Tiger.L)
    assert Tiger.L not in b and Tiger.R in b
    assert info["num_accepted"] == 100

    b, info = belief_update(tiger_right_belief, Tiger.L, Tiger.L)
    assert Tiger.L in b and Tiger.R in b
    assert info["num_accepted"] == 100
