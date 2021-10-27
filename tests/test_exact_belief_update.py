"""Tests for the exact belief_over_two_states tracking module :mod:`pomdp_belief_tracking.exact_belief_update`"""

from collections import Counter
from functools import partial
from typing import Any, Dict

import pytest

from pomdp_belief_tracking.exact_belief_update import (
    Belief,
    StateSpace,
    exact_belief_update,
    sample_belief,
)


def test_exact_belief_sampling():
    """Tests :func:`sample_belief`"""
    belief = [("s1", 0.2), ("s2", 0.4), (True, 0.05), (100, 0.35)]

    tot_samples = 100000
    sample_counts = Counter([sample_belief(belief) for _ in range(tot_samples)])

    assert sample_counts["s1"] / tot_samples == pytest.approx(0.2, abs=0.01)
    assert sample_counts["s2"] / tot_samples == pytest.approx(0.4, abs=0.01)
    assert sample_counts[True] / tot_samples == pytest.approx(0.05, abs=0.01)
    assert sample_counts[100] / tot_samples == pytest.approx(0.35, abs=0.01)


def toggle_transition_model(s: bool, a: bool, ss: bool, toggle_prob: float) -> float:
    """'Toggle' transition model

    The true dynamics keep `s` unchanged if `not a`, else it toggles the state
    with probability `toggle_prob`.
    """
    if not a:
        return float(s == ss)

    return toggle_prob if (s != ss) else 1 - toggle_prob


def observation_model(ss: bool, o: bool, prob: float) -> float:
    """Observation model that returns `prob` if `ss == 0`"""
    return prob if ss == o else 1 - prob


def dynamics_model(
    state: bool,
    action: bool,
    next_state: bool,
    o: bool,
    observation_prob: float,
    toggle_prob: float,
) -> float:
    """Combines :func:`toggle_transition_model` with :func:`observation_model` as dynamics"""
    return toggle_transition_model(
        state, action, next_state, toggle_prob
    ) * observation_model(next_state, o, observation_prob)


def belief_over_two_states(s: bool, prob_s_false: float) -> float:
    """A weird representation of a belief_over_two_states

    Assuming a state space of `[True, False]`, we compute the probability of
    `s` given the probability that `s` is `True`.

    This seems a little weird, but it can be used to create a belief_over_two_states::

        partial(belief_over_two_states, prob_s_true=.5)  # creates uniform belief_over_two_states
    """
    return prob_s_false if not s else 1 - prob_s_false


def create_state_probabilities(states: StateSpace, belief_f) -> Belief:
    """Helper function for casting beliefs and spaces into the ([state, prob]) format"""
    return [(s, belief_f(s)) for s in states]


def test_fully_observable_deterministic_transition_belief_update():
    """Tests exact belief_over_two_states tracking given a deterministic fully observable dynamics"""
    info: Dict[str, Any] = {}
    s1 = False
    s2 = True
    simple_space = [s1, s2]

    o1 = s1
    o2 = s2

    stay_action = False
    toggle_action = True

    fully_observable_dynamics = partial(
        dynamics_model, observation_prob=1.0, toggle_prob=1.0
    )

    # When starting in belief_over_two_states p(s1) = 1., and _not toggling_, then
    # the next belief_over_two_states should also assign full probability to s1
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=1.0)
        ),
        stay_action,
        o1,
        fully_observable_dynamics,
        simple_space,
        info,
    )
    assert next_belief[0] == (s1, 1.0)
    assert next_belief[1] == (s2, 0.0)

    # When starting in belief_over_two_states p(s1) = 0.4, and _not toggling_ but observing `o1`, then
    # the next belief_over_two_states should assign full probability to `s1`
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=0.4)
        ),
        stay_action,
        o1,
        fully_observable_dynamics,
        simple_space,
        info,
    )
    assert next_belief[0] == (s1, 1.0)
    assert next_belief[1] == (s2, 0.0)

    # When starting in belief_over_two_states p(s1) = 0.4, and _not toggling_ but observing `o2`, then
    # the next belief_over_two_states should assign full probability to `s2`
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=0.4)
        ),
        stay_action,
        o2,
        fully_observable_dynamics,
        simple_space,
        info,
    )
    assert next_belief[0] == (s1, 0.0)
    assert next_belief[1] == (s2, 1.0)

    # When starting in belief_over_two_states p(s1) = 0.4, and _toggling_ but observing `o2`, then
    # the next belief_over_two_states should assign full probability to `s2`
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=0.4)
        ),
        toggle_action,
        o2,
        fully_observable_dynamics,
        simple_space,
        info,
    )
    assert next_belief[0] == (s1, 0.0)
    assert next_belief[1] == (s2, 1.0)
    assert "belief_update_runtime" in info


def test_complex_exact_belief_update():
    """Tests exact belief_over_two_states tracking given a complex dynamics model"""
    info: Dict[str, Any] = {}
    s1 = False
    s2 = True
    simple_space = [s1, s2]

    o1 = s1
    o2 = s2

    stay_action = False
    toggle_action = True

    toggle_prob = 0.75
    observation_prob = 0.9

    partial_observable_dynamics = partial(
        dynamics_model, observation_prob=observation_prob, toggle_prob=toggle_prob
    )

    # When starting in belief_over_two_states p(s1) = 0.3, and _not toggling_, then
    # the next belief_over_two_states should also assign full probability to s1
    p_s1 = 1.0
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=p_s1)
        ),
        stay_action,
        o1,
        partial_observable_dynamics,
        simple_space,
        info,
    )
    assert next_belief[0] == (s1, 1.0)
    assert next_belief[1] == (s2, 0.0)

    # When starting in belief_over_two_states p(s1) = 0.4, and _not toggling_ but observing `o1`, then
    # the next belief_over_two_states should be some more complex thing
    p_s1 = 0.4
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=p_s1)
        ),
        stay_action,
        o1,
        partial_observable_dynamics,
        simple_space,
        info,
    )
    unorm_s1_prob = p_s1 * observation_prob
    unorm_s2_prob = (1 - p_s1) * (1 - observation_prob)
    norm = unorm_s1_prob + unorm_s2_prob
    assert next_belief[0] == (s1, pytest.approx(unorm_s1_prob / norm))
    assert next_belief[1] == (s2, pytest.approx(unorm_s2_prob / norm))

    # When starting in belief_over_two_states p(s1) = 0.4, and _not toggling_ but observing `o2`, then
    # the next belief_over_two_states should be some complex thing
    p_s1 = 0.4
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=0.4)
        ),
        stay_action,
        o2,
        partial_observable_dynamics,
        simple_space,
        info,
    )
    unorm_s1_prob = p_s1 * (1 - observation_prob)
    unorm_s2_prob = (1 - p_s1) * observation_prob
    norm = unorm_s1_prob + unorm_s2_prob
    assert next_belief[0] == (s1, pytest.approx(unorm_s1_prob / norm))
    assert next_belief[1] == (s2, pytest.approx(unorm_s2_prob / norm))

    # When starting in belief_over_two_states p(s1) = 0.4, and _toggling_ but observing `o2`, then
    # the next belief_over_two_states should be some complex thing
    p_s1 = 0.4
    next_belief = exact_belief_update(
        create_state_probabilities(
            simple_space, partial(belief_over_two_states, prob_s_false=0.4)
        ),
        toggle_action,
        o2,
        partial_observable_dynamics,
        simple_space,
        info,
    )
    unorm_s1_prob = (
        p_s1 * (1 - observation_prob) * (1 - toggle_prob)
        + (1 - p_s1) * (1 - observation_prob) * toggle_prob
    )
    unorm_s2_prob = p_s1 * observation_prob * toggle_prob + (
        1 - p_s1
    ) * observation_prob * (1 - toggle_prob)
    norm = unorm_s1_prob + unorm_s2_prob
    assert next_belief[0] == (s1, pytest.approx(unorm_s1_prob / norm))
    assert next_belief[1] == (s2, pytest.approx(unorm_s2_prob / norm))


if __name__ == "__main__":
    pytest.main([__file__])
