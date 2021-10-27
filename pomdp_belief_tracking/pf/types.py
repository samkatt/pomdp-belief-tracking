"""Some additional (shared) particle filtering types"""
from __future__ import annotations

from typing import Any, Tuple

from typing_extensions import Protocol

from pomdp_belief_tracking.types import Action, Info, Observation, State


class TransitionFunction(Protocol):
    """The signature for transition functions"""

    def __call__(self, s: State, a: Action) -> State:
        """A transition function is a (stochastic) mapping from s x a -> s

        :param s: state at time step t
        :param a: action at time step t
        :return: state at time step t + 1
        """


class Simulator(Protocol):
    """The abstract type representing simulators

    We expect simulators to map a state and action into a next state and
    observation.

    .. automethod:: __call__
    """

    def __call__(self, s: State, a: Action) -> Tuple[State, Observation]:
        """Simulate a transition

        :param s: the current state
        :param a: the executed action
        :return: next state and observation
        """
        raise NotImplementedError()


class GenerativeStateDistribution(Protocol):
    """The abstract type representing (generative) state distributions

    We expect to be able to sample states

    .. automethod:: __call__
    """

    def __call__(self) -> State:
        """Required implementation of distribution: the ability to sample states

        :return: state sampled according to distribution
        """


class BeliefUpdate(Protocol):
    """The signature for belief updates"""

    def __call__(
        self, p: GenerativeStateDistribution, a: Action, o: Observation
    ) -> Tuple[GenerativeStateDistribution, Info]:
        """Updates the distribution ``p`` given an action and observation

        :param p: current distribution
        :param a: taken action
        :param o: perceived observation
        :return: next distribution
        """
        raise NotImplementedError()


class ProposalDistribution(Protocol):
    """The signature for the proposal distribution for sampling"""

    def __call__(self, s: State, info: Info) -> Tuple[State, Any]:
        """Proposes an updated sample from some initial ``s``

        :param s: a sample at t
        :param info: run time information
        :return: an updated sample at t+1 and any additional context
        """
        raise NotImplementedError()
