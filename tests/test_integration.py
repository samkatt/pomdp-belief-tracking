#!/usr/bin/env python
"""Tests full belief updates or other larger components"""
import random
from typing import Tuple

from pomdp_belief_tracking.pf import CountAcceptedSamples, ParticleFilter, create_rejection_sampling
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
    """tests :py:func:`~online_pomdp_planning.mcts.create_POUCT` on Tiger"""

    count_accepts = CountAcceptedSamples()
    count_rejects = CountAcceptedSamples()
    belief_update = create_rejection_sampling(
        Tiger.sim, 100, process_acpt=count_accepts, process_rej=count_rejects
    )

    b = belief_update(uniform_tiger_belief, Tiger.H, Tiger.L)

    assert isinstance(b, ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) > b.probability_of(Tiger.R)
    assert count_accepts.count == 100
    assert 50 < count_rejects.count < 150

    b = belief_update(uniform_tiger_belief, Tiger.H, Tiger.R)

    assert isinstance(b, ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) < b.probability_of(Tiger.R)
    assert 100 < count_rejects.count < 300
    assert count_accepts.count == 200
