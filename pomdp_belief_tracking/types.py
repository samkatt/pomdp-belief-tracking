"""Defines some types for ease of reading"""

from typing import Protocol, Tuple


class Action(Protocol):
    """The abstract type representing actions"""


class Observation(Protocol):
    """The abstract type representing observations"""


class State(Protocol):
    """The abstract type representing states"""


class Simulator(Protocol):
    """The abstract type representing simulators

    We expect simulators to map a state and action into a next state and
    observation.

    .. automethod:: __call__
    """

    def __call__(self, s: State, a: Action) -> Tuple[State, Observation]:
        """Simulate a transition

        :param s: the current state
        :type s: State
        :param a: the executed action
        :type a: Action
        :return: next state and observation
        :rtype: Tuple[State, Observation]
        """


class StateDistribution(Protocol):
    """The abstract type representing state distributions

    We expect to be able to sample states

    .. automethod:: __call__
    """

    def __call__(self) -> State:
        """Required implementation of distribution: the ability to sample states

        :return: state sampled according to distribution
        :rtype: State
        """


class BeliefUpdate(Protocol):
    """The signature for belief updates"""

    def __call__(
        self, p: StateDistribution, a: Action, o: Observation
    ) -> StateDistribution:
        """Updates the distribution `p` given an action and observation

        :param p: current distribution
        :type p: StateDistribution
        :param a: taken action
        :type a: Action
        :param o: perceived observation
        :type o: Observation
        :return: next distribution
        :rtype: StateDistribution
        """


class Belief:
    """A belief is the combination of a update function and current distribution"""

    def __init__(
        self, initial_distribution: StateDistribution, update_function: BeliefUpdate
    ):
        self.distribution = initial_distribution
        self.update_function = update_function

    def update(self, a: Action, o: Observation) -> None:
        """Updates (in place) the state distribution given an action and observation

        :param a: the executed action
        :type a: Action
        :param o: the perceived observation
        :type o: Observation
        :return: Side effect: updates in place
        :rtype: None
        """
        self.distribution = self.update_function(self.distribution, a, o)

    def sample(self) -> State:
        """Samples from its distribution

        :return: state sampled according to distribution
        :rtype: State
        """
        return self.distribution()
