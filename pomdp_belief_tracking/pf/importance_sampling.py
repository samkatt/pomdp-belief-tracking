"""Provides a general implementation of importance sampling:

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

Sequential importance sampling, the application of the belief update over
multiple time steps, often involes :func:`resample` to avoid particle
degeneration. When to resample is not straightforward; we provide a general
condition protocol

.. autoclass:: ResampleCondition
   :noindex:

Lastly, we provide a factory function that combines importance sampling with
resampling

.. autofunction:: create_sequential_importance_sampling
   :noindex:

.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
  Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
  signal processing magazine, 20(5), 19-38.

"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from timeit import default_timer as timer
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
        :param sample_ctx: context around proposal
        :param info: global information stored during importance sampling
        :return: a 0 <= weight <= 1
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

    Returns how long the update took in ``info`` with key
    "belief_update_runtime"

    :param proposal_distr: function to propose sample updates
    :param weight_func: function that weights propsals
    :param particles: the starting set of particles
    :return: a new particle set
    """

    info: Info = {}

    new_particles: List[Particle] = []

    t = timer()

    for state, weight in particles:
        next_state, ctx = proposal_distr(state, info)
        weight = weight * weight_func(next_state, ctx, info)

        new_particles.append(Particle(next_state, weight))

    info["belief_update_runtime"] = timer() - t

    return ParticleFilter.from_particles(new_particles), info


def resample(pf: ParticleFilter, n: int) -> ParticleFilter:
    """Samples ``n`` particles from ``distr``

    .. todo:

        This implementation is squared in the number of particles - because
        sampling from particle filter is linear in number of particles -. Good
        excuse for optimization, maybe allow for sampling multiple at a time)

    :param pf: incoming particle filter
    :param n: number of desired samples in returned PF
    :return: the resulting particle filter of resampling ``pf``
    """
    assert n > 0
    return ParticleFilter(list(deepcopy(pf()) for _ in range(n)))


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
    :param observation_model: the model to weight the probability of generating observation ``o``
    :param n: num samples, optional
    :param initial_state_distribution: the starting distribution
    :param a: taken action
    :param o: taken observation
    :return: updated belief
    """

    # create particles to give to importance sampling
    if not isinstance(initial_state_distribution, ParticleFilter):
        assert n and n > 0
        initial_state_distribution = ParticleFilter.from_distribution(
            initial_state_distribution, n
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
    :param observation_model: how to weight transitions
    :param n: num samples, optional
    :return: func:`importance_sample` as :class:`pomdp_belief_tracking.types.BeliefUpdate`
    """

    return partial(importance_sample, transition_func, observation_model, n)


class ResampleCondition(Protocol):
    """The signature of a resample condition

    .. automethod:: __call__

    Provided implementations:

    .. autosummary::
       :nosignatures:

       ineffective_sample_size
    """

    def __call__(self, pf: ParticleFilter) -> bool:
        """Inspects ``pf`` and decides whether it is time to re-sample


        :param pf: the particle filter to potentially resample
        :return: ``True`` if ``pf`` should be resampled
        """


def ineffective_sample_size(minimal_size: float, pf: ParticleFilter):
    """Returns whether the sample size of ``pf`` is lower than ``minimal_size``

    When given ``minimal_size`` this implements :class:`ResampleCondition`
    protocol. Asserts that ``minimal_size`` > 0

    Calls
    :func:`~pomdp_belief_tracking.pf.particle_filter.effective_sample_size`
    under the hood

    :param minimal_size: the required sample size for this to return False (> 0)
    :param pf: the particle filter to test the sample size of
    :returns: True if ``minimal_size`` > sample size of ``pf``

    """
    assert minimal_size > 0, f"effective sample size ({minimal_size}) must be positive"
    return minimal_size > pf.effective_sample_size()


def create_sequential_importance_sampling(
    resample_condition: ResampleCondition,
    transition_func: TransitionFunction,
    observation_model: Callable[[State, Action, State, Observation], float],
    n: Optional[int] = None,  # pylint: disable=unsubscriptable-object
) -> BeliefUpdate:
    """Main entry point of this module to create importance sampling update

    A simple wrapper combining :func:`resample` (if ``resample_condition`` is
    met) with :func:`importance_sample` (created by calling
    :func:`create_importance_sampling`

    ``n`` is necessary when ``initial_state_distribution`` is not a
    :class:`~pomdp_belief_tracking.pf.particle_filter.ParticleFilter`.
    Otherwise ignored.

    :param resample_condition: when to resample (called before IS)
    :param transition_func: the transition function to propose particles
    :param observation_model: the function to weight the new particles
    :param n: number of desired particles
    """

    IS = create_importance_sampling(transition_func, observation_model, n)

    def belief_update(
        p: StateDistribution, a: Action, o: Observation
    ) -> Tuple[StateDistribution, Info]:
        """belief_update.

        :param p:
        :param a:
        :param o:
        """

        resampled = False
        if isinstance(p, ParticleFilter) and resample_condition(p):
            p = resample(p, len(p))
            resampled = True

        belief, info = IS(p, a, o)
        info["importance_sampling_resampled"] = resampled

        return belief, info

    return belief_update
