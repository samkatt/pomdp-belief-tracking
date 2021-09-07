""" Implementation of particle filters (PF [particle-filtering]_)

PF approximate distributions with **weighted particles**. A
:class:`~pomdp_belief_tracking.pf.particle_filter.Particle` is a
:class:`~pomdp_belief_tracking.types.State`, and the weight corresponds to its
relative probability.

.. autoclass:: pomdp_belief_tracking.pf.particle_filter.Particle
    :members:
    :noindex:

The particle filter is a
:class:`~pomdp_belief_tracking.types.StateDistribution`, but provides the
additional API. Mainly some convenient constructors, and the ability to use
them as containers.

.. autoclass:: pomdp_belief_tracking.pf.particle_filter.ParticleFilter
    :members: probability_of, from_distribution, from_particles, effective_sample_size
    :special-members: __call__, __len__, __contains__, __iter__
    :noindex:

.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
   Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
   signal processing magazine, 20(5), 19-38.

"""
from __future__ import annotations

from math import isclose
from operator import eq
from random import uniform
from typing import Callable, Iterable, NamedTuple, Optional, Sequence

from pomdp_belief_tracking.types import State, StateDistribution


class Particle(NamedTuple):
    """A (weighted) particle contains a state and weight"""

    state: State
    """The 'particle' or value"""
    weight: float
    """The (relative) probability of the particle"""


class ParticleFilter(StateDistribution):
    """A distribution from weighted particles"""

    def __init__(self, states: Sequence[State]):
        """Creates a particle filter from provided states

        Provides uniform weights to ``states``

        :param states: the particles
        """
        super().__init__()

        if len(states) > 0:
            weight = 1 / len(states)
            self.particles = [Particle(s, weight) for s in states]
        else:
            self.particles = []

    def __call__(self) -> State:
        """Implements :class:`StateDistribution` protocol: sample states

        Simply samples a state according to the distribution specified by the
        particles.

        Returns the **actual state**, so beware of changing it, as it will
        affect the distribution.

        :return: a sample state
        """
        u = uniform(0, 1)

        acc = 0.0
        for s, w in self.particles:
            acc += w
            if acc > u:
                return s

        assert False, f"Somehow our total weight did not add up to 1, but to {acc}"

        # for those that disable assertion, let us hope this is not a bug in
        # our code but one of rounding off errors, and thus return the last
        # element
        return self.particles[-1].state

    def __len__(self):
        """Returns the number of particles

        Allows for `len(particle_filter)` syntax
        """
        return len(self.particles)

    def __repr__(self):
        return f"ParticleFilter({repr(self.particles)})"

    def __iter__(self):
        """Returns iterator over the particles

        Allows for `for state, weight in particle_filter:`
        """
        return iter(self.particles)

    def __contains__(self, item):
        """Checks whether ``item`` is a state in our particles

        Allows for `s in particle_filter` syntax

        :param item: the item to check for
        """
        return any(item == particle.state for particle in self.particles)

    @staticmethod
    def total_weight(particles: Iterable[Particle]):
        """Computes the total weight in ``particles``

        :param particles: particles to compute the total weight of
        """
        return sum(p.weight for p in particles)

    def probability_of(
        self, s: State, equality_function: Callable[[State, State], bool] = eq
    ) -> float:
        """Returns the probability of ``s`` with equality ``equality_function``

        :param s: the state of which we will compute the probability
        :param equality_function: returns whether two states are the same, defaults to eq
        :return: a number 0 ... 1
        """
        return ParticleFilter.total_weight(
            p for p in self if equality_function(p.state, s)
        )

    def effective_sample_size(self) -> float:
        """Returns the "effective sample size" of the particle filter

        Calls :func:`effective_sample_size`

        :return: effective sample size of `self`
        """
        # we _know_ (ensure) that the total weight is always 1.
        return effective_sample_size((p.weight for p in self.particles), 1.0)

    @staticmethod
    def from_distribution(distr: StateDistribution, n: int) -> ParticleFilter:
        """Constructs a particle filter of ``n`` particles from ``distr``

        Basically samples ``n`` particles.

        NOTE::

            Does not _copy_ samples drawn from ``distr``. Make sure `distr()`
            copies the particles if necessary

        :param distr: the distribution to approximate
        :param n: the number of particles to approximate with
        :return: a particle filter approximating ``distr`` with ``n`` particles
        """
        return ParticleFilter([distr() for _ in range(n)])

    @staticmethod
    def _from_normalized_particles(particles: Iterable[Particle]) -> ParticleFilter:
        """A private function, creates a particle filter from provided particles

        Assumes (but checks) whether the total weight of ``particles`` is approximately 1.0

        :param particles: the particles to create a filter from
        :return: a particle filter that contains ``particles``
        """
        assert isclose(
            ParticleFilter.total_weight(particles), 1.0
        ), f"total weight is {ParticleFilter.total_weight(particles)}"

        pf = ParticleFilter([])
        pf.particles = list(particles)

        return pf

    @staticmethod
    def from_particles(particles: Iterable[Particle]) -> ParticleFilter:
        """Creates a particle filter from ``particles``

        Normalizes the weights in ``particles``

        Note:
            Does not copy any of the particles, and particles may change in the
            future through belief updates


        :param particles: the particles to create the filter from
        :return: a particle filter from given ``particles``
        """
        particles = list(particles)
        total_weight = ParticleFilter.total_weight(particles)

        return ParticleFilter._from_normalized_particles(
            list(Particle(p.state, p.weight / total_weight) for p in particles)
        )


def effective_sample_size(
    weights: Iterable[float],
    total_weight: Optional[float] = None,
) -> float:
    """Computes the "effective sample size" of the given weights

    This value represents how "healthy" the underlying samples are. The lower
    this value, the fewer "real samples" are represented. As in, the closer
    this comes to zero, the more degenerated the samples are.

    See `https://en.wikipedia.org/wiki/Effective_sample_size`

    :param weights: the weights of the samples
    :param total_weight: total weight of all samples, requires extra computation if not given
    :return: the effective sample size of ``weights``
    """
    if total_weight is None:
        total_weight = sum(weights)

    assert total_weight and total_weight > 0  # for mypy

    return pow(total_weight, 2) / sum(pow(w, 2) for w in weights)


def apply(f: Callable[[State], State], pf: ParticleFilter) -> ParticleFilter:
    """Returns a new :class:`ParticleFilter` with ``f`` applied on states in ``pf``

    Note: assumes ``f`` does _not_ affect the input :class:`State`, otherwise
    the input ``pf`` will be affected to

    :param f: the function to apply to particles
    :param pf: input starting particle filter / belief
    :return: the result of applying ``f`` onto ``pf``
    """
    return ParticleFilter.from_particles(Particle(f(s), w) for s, w in pf)
