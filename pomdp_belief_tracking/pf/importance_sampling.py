"""

.. autofunction:: general_importance_sample
   :noindex:

Our belief update version of importance sampling calls this function with the
appropriate parameters:

.. autofunction:: importance_sample
   :noindex:

Which can best be created through our construction function (with sane
defaults, otherwise you can also apply partial):

.. autofunction:: create_importance_sampling
   :noindex:

.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
  Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
  signal processing magazine, 20(5), 19-38.

"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple

from typing_extensions import Protocol

from pomdp_belief_tracking.pf.particle_filter import Particle, ParticleFilter
from pomdp_belief_tracking.pf.types import ProposalDistribution
from pomdp_belief_tracking.types import (
    Action,
    BeliefUpdate,
    Info,
    Observation,
    State,
    StateDistribution,
    TransitionFunction,
)


class WeightFunction(Protocol):
    """Signature of a weighting function in :func:`general_importance_sample`

    .. automethod:: __call__

    """

    def __call__(self, proposal: State, sample_ctx: Any, info: Info) -> float:
        """Weights a ``state`` -> ``proposal`` transition under ``sample_ctx``

        :param proposal: proposed (updated) sample
        :type proposal: State
        :param sample_ctx: context around proposal
        :type sample_ctx: Any
        :param info: global information stored during importance sampling
        :type info: Info
        :return: a 0 <= weight <= 1
        :rtype: float
        """


def general_importance_sample(
    proposal_distr: ProposalDistribution,
    weight_func: WeightFunction,
    particles: Iterable[Particle],
) -> Tuple[ParticleFilter, Info]:
    """The particle filter implementation of IS

    The underlying algorithm for importance sampling uses a
    :class:`~pomdp_belief_tracking.pf.types.ProposalDistribution` and weighting
    distribution to update a particle filter::

        for weight, sample in particles:
            sample ~ proposal_distr(sample)
            weight <- weight * weight_func(sample)

    :param proposal_distr: function to propose sample updates
    :type proposal_distr: ProposalDistribution
    :param weight_func: function that weights propsals
    :type weight_func: WeightFunction
    :param particles: the starting set of particles
    :type particles: Iterable[Particle]
    :return: a new particle set
    :rtype: Tuple[ParticleFilter, Info]
    """

    info: Info = {}

    new_particles: List[Particle] = []

    for state, weight in particles:
        next_state, ctx = proposal_distr(state, info)
        weight = weight * weight_func(next_state, ctx, info)

        new_particles.append(Particle(next_state, weight))

    return ParticleFilter.from_particles(new_particles), info


def resample(distr: StateDistribution, n: int) -> List[Particle]:
    """Samples ``n`` particles from ``distr``

    :param distr: distribution to resample
    :type distr: StateDistribution
    :param n: number of desired samples in returned PF
    :type n: int
    :return: particles resulted from resampling from ``distr``
    :rtype: List[Particle]
    """
    assert n > 0

    w = 1.0 / n
    return list(Particle(distr(), w) for _ in range(n))


def importance_sample(
    transition_func: TransitionFunction,
    observation_model: Callable[[State, Action, State, Observation], float],
    n: Optional[int],  # pylint: disable=unsubscriptable-object
    initial_state_distribution: StateDistribution,
    a: Action,
    o: Observation,
) -> Tuple[ParticleFilter, Info]:
    """Applies :func:`general_importance_sample` on POMDPs

    Here the ``transition_func`` is used to propose next states, which are
    weighted according to the ``weight_func`` given ``o``. If
    ``initial_state_distribution`` is _not_ a particle filter (with a given
    size), then we sample ``n`` particles with weight 1 to start IS. Otherwise
    we use the particles in the PF.

    ``n`` is necessary when ``initial_state_distribution`` is not a
    :class:`~pomdp_belief_tracking.pf.particle_filter.ParticleFilter`.
    Otherwise ignored.

    :param transition_func: the proposal function
    :type transition_func: TransitionFunction
    :param observation_model: the model to weight the probability of generating observation ``o``
    :type observation_model: Callable[[State, Action, State, Observation], float]
    :param n: num samples, optional
    :type n: Optional[int]
    :param initial_state_distribution: the starting distribution
    :type initial_state_distribution: StateDistribution
    :param a: taken action
    :type a: Action
    :param o: taken observation
    :type o: Observation
    :return: updated belief
    :rtype: Tuple[ParticleFilter, Info]
    """

    # create particles to give to importance sampling
    if not isinstance(initial_state_distribution, ParticleFilter):
        assert n and n > 0
        initial_state_distribution = ParticleFilter.from_particles(
            resample(initial_state_distribution, n)
        )
    particles = iter(initial_state_distribution.particles)

    def prop(s: State, info: Info) -> Tuple[State, Any]:
        """turns the transition function into a proposal function"""
        ss = transition_func(s, a)
        return ss, {"action": a, "state": s, "observation": o}

    def weighting(proposal: State, sample_ctx: Any, info: Info) -> float:
        """weights the proposal according to the transition predicting observation"""
        s, a, o = sample_ctx["state"], sample_ctx["action"], sample_ctx["observation"]
        return observation_model(s, a, proposal, o)

    return general_importance_sample(prop, weighting, particles)


def create_importance_sampling(
    transition_func: TransitionFunction,
    observation_model: Callable[[State, Action, State, Observation], float],
    n: Optional[int],  # pylint: disable=unsubscriptable-object
) -> BeliefUpdate:
    """Partial function that returns a regular IS belief update

    A simple wrapper around
    :func:`~pomdp_belief_tracking.pf.importance_sampling.importance_sample`

    Here the ``transition_func`` is used to propose next states, which are
    weighted according to the ``weight_func``. If the belief update is _not_ a
    particle filter (with a given size), then we sample ``n`` particles with
    weight 1 to start IS. Otherwise we use the particles in the PF.

    ``n`` is necessary when ``initial_state_distribution`` is not a
    :class:`~pomdp_belief_tracking.pf.particle_filter.ParticleFilter`.
    Otherwise ignored.

    :param transition_func: how to update states
    :type transition_func: TransitionFunction
    :param observation_model: how to weight transitions
    :type observation_model: Callable[[State, Action, State, Observation], float]
    :param n: num samples, optional
    :type n: Optional[int]
    :return: func:`importance_sample` as :class:`pomdp_belief_tracking.types.BeliefUpdate`
    :rtype: BeliefUpdate
    """

    return partial(importance_sample, transition_func, observation_model, n)
