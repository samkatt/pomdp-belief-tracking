"""Exact computations of the belief update.

Will exactly compute the new belief (posterior) given an old belief (prior)
and a new action-observation pair. Requires iterating twice over the state
space, and thus has complexity O(S*S).

Concretely, this module provides a simple belief representation --- a list of
state-probability tuples --- and a way of updating and sampling from it.

    - :class:`Belief`
    - :func:`exact_belief_update`
    - :func:`sample_belief`

"""

import random
from timeit import default_timer as timer
from typing import Iterable, List, Tuple

from typing_extensions import Protocol

from pomdp_belief_tracking.types import Action, Info, Observation, State

StateSpace = Iterable[State]
"""The (finite) space of possible states (iterable)"""

Belief = List[Tuple[State, float]]
"""The least assuming representation of a belief, state-probability tuples"""


def sample_belief(belief: Belief) -> State:
    """Samples a :class:`State` from `belief`"""
    u = random.uniform(0, 1.0)

    acc = 0.0
    for s, p in belief:
        acc += p
        if u <= acc:
            return s

    raise ValueError("Input belief total probability is less than 1: " + str(acc))


class DynamicsModel(Protocol):
    """The most general interface of a dynamics models, returns the probability of a (s,a,ss,o) transition

    .. automethod __call__
    """

    def __call__(
        self, state: State, action: Action, next_state: State, obs: Observation
    ) -> float:
        """The signature of the expected callable

        Provides the probability of a state, action, observation, and next state

        :param state: the current state from which an `action` is taken
        :param action: the action that is taken from `state`
        :param next_state: the state that is transitioned into from `state` given `action`
        :param obs: the observation that is generated when transitioning from `state` to `next_state` given `action`
        """
        ...


def exact_belief_update(
    belief: Belief,
    action: Action,
    obs: Observation,
    dynamics_model: DynamicsModel,
    space: StateSpace,
    info: Info,
) -> List[Tuple[State, float]]:
    """The most general exact belief update possible

    Simply iterates over all states twice and computes `belief(s) *
    dynamics_model(s,a,ss,o)`. It then normalizes the resulting belief.

    Has the fewest possible assumptions I can think of, only that there is
    'some' function that can give us the probability `p(s',o|s,a)`. Similarly,
    since we know _nothing_ about :class:`State`, including whether it can be
    used as an index, our return is super dumb: a tuple of state and
    probability.

    Populates `info` with "belief_update_runtime" (float)

    :param belief: is the current belief (just a list of states and their probabilities)
    :param action: is the taken action
    :param obs: the perceived observation
    :param dynamics_model: the probability of dynamics p(s',o|s,a)
    :param space: the space of possible states
    :param info: used as information store
    """

    t = timer()

    unnormalized_belief = [
        (ss, sum(s_prob * dynamics_model(s, action, ss, obs) for s, s_prob in belief))
        for ss in space
    ]

    normalization_constant = sum(w for _, w in unnormalized_belief)
    belief = [(ss, w / normalization_constant) for ss, w in unnormalized_belief]

    info["belief_update_runtime"] = timer() - t

    return belief
