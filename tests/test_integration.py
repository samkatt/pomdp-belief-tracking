#!/usr/bin/env python
"""Tests full belief updates or other larger components"""
import random
from typing import List, Tuple

from pomdp_belief_tracking.pf.importance_sampling import create_importance_sampling
from pomdp_belief_tracking.pf.rejection_sampling import (
    ParticleFilter,
    accept_noop,
    create_rejection_sampling,
    reject_noop,
)
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

    @staticmethod
    def observation_model(a: Action, next_s: State) -> List[float]:
        """Returns the observation probabilities a, next_s' pair

        :param next_s: next state
        :type next_s: State
        :param a: taken action
        :type a: Action
        :return: [prob hearing left, prob hearing right]
        :rtype: List[float]
        """
        if a != Tiger.H:
            return [0.5, 0.5]

        if next_s == Tiger.L:
            return [0.85, 0.15]

        assert next_s == Tiger.R

        return [0.15, 0.85]


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
    """tests :func:`~pomdp_belief_tracking.pf.rejection_sampling.rejection_sample` on Tiger"""

    belief_update = create_rejection_sampling(
        Tiger.sim,
        100,
        process_acpt=accept_noop,
        process_rej=reject_noop,
    )

    b, info = belief_update(uniform_tiger_belief, Tiger.H, Tiger.L)

    assert isinstance(b, ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) > b.probability_of(Tiger.R)
    assert info["num_accepted"] == 100
    assert 50 < info["iteration"] - info["num_accepted"] < 150

    b, info = belief_update(uniform_tiger_belief, Tiger.H, Tiger.R)

    assert isinstance(b, ParticleFilter)
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


def test_importance_sampling():
    """tests :func:`~pomdp_belief_tracking.pf.importance_sampling.importance_sample` on Tiger"""

    n = 100

    def trans_func(s, a):
        return Tiger.sim(s, a)[0]

    def obs_func(s, a, ss, o):
        return Tiger.observation_model(a, ss)[o]

    belief_update = create_importance_sampling(trans_func, obs_func, n)

    b, _ = belief_update(uniform_tiger_belief, Tiger.H, Tiger.L)

    assert isinstance(b, ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) > b.probability_of(Tiger.R)
    assert len(b) == n

    b, _ = belief_update(uniform_tiger_belief, Tiger.H, Tiger.R)

    assert isinstance(b, ParticleFilter)
    assert Tiger.L in b and Tiger.R in b
    assert b.probability_of(Tiger.L) < b.probability_of(Tiger.R)
    assert len(b) == n

    b, _ = belief_update(tiger_right_belief, Tiger.H, Tiger.L)
    assert Tiger.L not in b and Tiger.R in b
    assert len(b) == n

    b, _ = belief_update(tiger_right_belief, Tiger.L, Tiger.L)
    assert Tiger.L in b and Tiger.R in b
    assert len(b) == n

    b, _ = belief_update(b, Tiger.H, Tiger.L)
    next_b, _ = belief_update(b, Tiger.H, Tiger.L)

    assert 0.5 < b.probability_of(Tiger.L) < next_b.probability_of(Tiger.L)
    assert 0.5 > b.probability_of(Tiger.R) > next_b.probability_of(Tiger.R)
    assert len(next_b) == n
