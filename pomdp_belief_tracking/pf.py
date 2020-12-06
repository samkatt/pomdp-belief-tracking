"""Particle Filters and their update functions

Particle filters are approximations of distributions. They represent them
through :py:class:`Particle`, a :py:class:`~pomdp_belief_tracking.types.State`
with a relative weight.

In this module we implement such a :py:class:`ParticleFilter` and some
functions to update them:

    - :py:func:`rejection_sample`
"""

from __future__ import annotations

from math import isclose
from random import uniform
from typing import NamedTuple, Sequence

from pomdp_belief_tracking.types import State, StateDistribution


class Particle(NamedTuple):  # pylint: disable=inherit-non-class
    """A (weighted) particle contains a state and weight"""

    state: State
    """The 'particle' or value"""
    weight: float
    """The (relative) probability of the particle"""


class ParticleFilter:
    """A distribution from weighted particles"""

    def __init__(self, states: Sequence[State]):
        """Creates a particle filter from provided states

        Provides uniform weights to ``states``

        :param states: the particles
        :type states: Sequence[State]
        """

        if len(states) > 0:
            weight = 1 / len(states)

        self.particles = [Particle(s, weight) for s in states]

    def __call__(self) -> State:
        """Implements `StateDistribution` protocol: sample states

        Simply samples a state according to the distribution specified by the
        particles.

        Returns the **actual state**, so beware of changing it, as it will
        affect the distribution.

        :return: a sample state
        :rtype: State
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
        return self.particles[-1]

    def __len__(self):
        """Returns the number of particles

        Allows for ``len(particle_filter)`` syntax
        """
        return len(self.particles)

    def __repr__(self):
        return f"ParticleFilter({repr(self.particles)})"

    def __iter__(self):
        """Returns iterator over the particles

        Allows for ``for state, weight in particle_filter:``
        """
        return iter(self.particles)

    def __contains__(self, item):
        """Checks whether `item` is a state in our particles

        Allows for ``s in particle_filter`` syntax

        :param item: the item to check for
        :type item: hopefully :py:class:`~pomdp_belief_tracking.types.State`
        """
        return any(item == particle.state for particle in self.particles)

    @staticmethod
    def total_weight(particles: Sequence[Particle]):
        """Computes the total weight in ``particles``

        :param particles: particles to compute the total weight of
        :type particles: Sequence[Particle]
        """
        return sum(p.weight for p in particles)

    @staticmethod
    def from_distribution(distr: StateDistribution, n: int) -> ParticleFilter:
        """Constructs a particle filter of ``n`` particles from ``distr``

        Basically samples ``n`` particles.

        NOTE:
            Does not _copy_ samples drawn from ``distr``. Make sure ``distr()``
            copies the particles if necessary

        :param distr: the distribution to approximate
        :type distr: StateDistribution
        :param n: the number of particles to approximate with
        :type n: int
        :return: a particle filter approximating ``distr`` with ``n`` particles
        :rtype: ParticleFilter
        """
        return ParticleFilter([distr() for _ in range(n)])

    @staticmethod
    def _from_normalized_particles(particles: Sequence[Particle]) -> ParticleFilter:
        """A private function, creates a particle filter from provided particles

        Assumes (but checks) whether the total weight of ``particles`` is approximately 1.0

        :param particles: the particles to create a filter from
        :type particles: Sequence[Particle]
        :return: a particle filter that contains ``particles``
        :rtype: ParticleFilter
        """
        assert isclose(
            ParticleFilter.total_weight(particles), 1.0
        ), f"total weight is {ParticleFilter.total_weight(particles)}"

        pf = ParticleFilter([])
        pf.particles = list(particles)

        return pf

    @staticmethod
    def from_particles(particles: Sequence[Particle]) -> ParticleFilter:
        """Creates a particle filter from ``particles``

        Note:
            Does not copy any of the particles, and particles may change in the
            future through belief updates


        :param particles: the particles to create the filter from
        :type particles: Sequence[Particle]
        :return: a particle filter from given ``particles``
        :rtype: ParticleFilter
        """

        total_weight = ParticleFilter.total_weight(particles)

        return ParticleFilter._from_normalized_particles(
            list(Particle(p.state, p.weight / total_weight) for p in particles)
        )


def rejection_sample():
    """TODO"""
