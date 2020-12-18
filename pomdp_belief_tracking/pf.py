"""Particle Filters and their update functions

Particle filters are approximations of distributions. They represent them
through :py:class:`Particle`, a :py:class:`~pomdp_belief_tracking.types.State`
with a relative weight.

In this module we implement such a :py:class:`ParticleFilter` and some
functions to update them:

    - :py:func:`rejection_sample`
"""

from __future__ import annotations

from functools import partial
from math import isclose
from operator import eq
from random import uniform
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from tqdm import tqdm  # type: ignore
from typing_extensions import Protocol

from pomdp_belief_tracking.types import (
    Action,
    BeliefUpdate,
    Info,
    Observation,
    Simulator,
    State,
    StateDistribution,
)


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
    def total_weight(particles: Iterable[Particle]):
        """Computes the total weight in ``particles``

        :param particles: particles to compute the total weight of
        :type particles: Sequence[Particle]
        """
        return sum(p.weight for p in particles)

    def probability_of(
        self, s: State, equality_function: Callable[[State, State], bool] = eq
    ) -> float:
        """Returns the probability of ``s`` with equality ``equality_function``

        :param s: the state of which we will compute the probability
        :type s: State
        :param equality_function: returns whether two states are the same, defaults to eq
        :type equality_function: Callable[[State, State], bool]
        :return: a number 0 ... 1
        :rtype: float
        """
        return ParticleFilter.total_weight(
            p for p in self if equality_function(p.state, s)
        )

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


class StopCondition(Protocol):
    """Whether to 'stop' sampling protocol"""

    def __call__(self, info: Info) -> bool:
        """The signature for the sampling stop condition

        This function is called at the start of each sampling attempt, provided
        some runtime ``info``, this function decides whether to continue
        sampling

        :param info: run time information
        :type info: Info
        :return: true if the condition for stopping is met
        :rtype: bool
        """


def have_sampled_enough(desired_num: int, info: Info) -> bool:
    """Returns ``true`` if "num_accepted" in ``info`` has reached ``desired_num``

    Given ``desired_num``, this implements :py:class`StopCondition`

    :param desired_num: number of desired samples
    :type desired_num: int
    :param info: run time info (should have an entry "num_accepted" -> int)
    :type info: Info
    :return: true if number of accepted samples is greater or equal to ``desired_num``
    :rtype: bool
    """
    assert desired_num > 0 and info["num_accepted"] >= 0

    return desired_num <= info["num_accepted"]


Sample = TypeVar("Sample")


class ProposalDistribution(Protocol):
    """The signature for the proposal distribution for the general rejection sampling"""

    def __call__(self, s: State, info: Info) -> Tuple[State, Any]:
        """Proposes an 'updated' sample from some initial ``s``

        :param s: a sample at t
        :type s: State
        :param info: run time information
        :type info: Info
        :return: an updated sample at t+1 and any additional context
        :rtype: Tuple[State, Any]
        """


class AcceptFunction(Protocol):
    """The signature of the acceptance function in general rejection sampling"""

    def __call__(self, s: State, ctx: Any, info: Info) -> bool:
        """Returns whether the sample ``s`` is accepted given some context ``ctx``

        :param s: the updated sample to be tested
        :type s: State
        :param ctx: output of :py:func:`AcceptFunction`
        :type ctx: Any
        :param info: run time information
        :type info: Info
        :return: whether the sample is accepted
        :rtype: bool
        """


class ProcessRejected(Protocol):
    """A function that processes a rejected sample in general rejection sampling"""

    def __call__(self, s: State, ctx: Any, info: Info) -> None:
        """Processes a rejected sample ``s`` with some context ``ctx``

        An example usage could be that if, for optimizations sake, the sample
        ``s`` was not copied from the original distribution, then this can
        reset

        :param s: the rejected sample
        :type s: State
        :param ctx: output of :py:func:`AcceptFunction`
        :type ctx: Any
        :param info: run time information
        :type info: Info
        :return: Nothing, only side effects
        :rtype: None
        """


def reject_noop(s: State, ctx: Any, info: Info) -> None:
    """A placeholder for :py:class:`ProcessRejected`: does nothing

    :param s: the accepted sample
    :type s: State
    :param ctx: the output of :py:func:`AcceptFunction`
    :type ctx: Any
    :param info: run time information (ignored)
    :type info: Info
    :return: Only side effects
    :rtype: None
    """


class ProcessAccepted(Protocol):
    """A function that processes an accepted sample in general rejection sampling"""

    def __call__(self, s: State, ctx: Any, info: Info) -> State:
        """Processes an accepted sample ``s`` with some context ``ctx``

        An example usage could be that if, for optimizations sake, the sample
        ``s`` was not copied from the original distribution, then this can
        reset it and return a copy

        :param s: accepted sample
        :type s: State
        :param ctx: output of :py:func:`AcceptFunction`
        :type ctx: Any
        :param info: run time information
        :type info: Info
        :return: processed sample
        :rtype: State
        """


def accept_noop(s: State, ctx: Any, info: Info) -> State:
    """A placeholder for :py:class:`ProcessAccepted`: does nothing

    :param s: the accepted sample (ignored)
    :type s: State
    :param ctx: the output of :py:func:`AcceptFunction` (ignored)
    :type ctx: Any
    :param info: run time information (ignored)
    :type info: Info
    :return: ``s``
    :rtype: State
    """
    return s


class AcceptionProgressBar(ProcessAccepted):
    """A :py:class:`ProcessAccepted` call that prints out a progress bar

    The progress bar is printed by ``tqdm``, and will magnificently fail if
    something else is printed or logged during.

    XXX: not tested because UI is hard to test, please modify with care
    """

    def __init__(self, total_expected_samples: int):
        """Sets up a progress bar for up to ``total_expected_samples`` samples

        We assume that ``total_expected_samples`` will be _exactly_ the number
        of samples to be accepted (i.e. calls to ``__call__``). Any less will
        not close the progress bar, any more and the progress bar will reset.

        :param total_expected_samples: 'length' of progress bar
        :type total_expected_samples: int
        """
        super().__init__()
        self._total_expected_calls = total_expected_samples

        # ``tqdm`` starts the progress bar upon initiation. At this point the
        # belief update is not happening yet, so we do not want to print it
        self.pbar: Optional[tqdm] = None  # pylint: disable=unsubscriptable-object

    def __call__(self, s: State, ctx: Any, info: Info) -> State:
        """Called upon accepting a sample. Updates progress bar

        Closes bar upon reaching ``total_expected_samples``

        :param s: accepted state (returned without modification)
        :type s: State
        :param ctx: context of acception (ignored)
        :type ctx: Any
        :param info: run time information (# accepted samples)
        :type info: Info
        :return: ``s`` as input
        :rtype: State
        """

        if info["num_accepted"] == 0:
            # the first sample is accepted, LGTM
            self.pbar = tqdm(total=self._total_expected_calls)

        assert self.pbar

        self.pbar.update()

        if info["num_accepted"] == self._total_expected_calls - 1:
            # last sample accepted!
            self.pbar.close()

        return s


def general_rejection_sample(
    stop_condition: StopCondition,
    proposal_distr: ProposalDistribution,
    accept_function: AcceptFunction,
    distr: Callable[[], State],
    process_accepted: ProcessAccepted = accept_noop,
    process_rejected: ProcessRejected = reject_noop,
) -> Tuple[List[State], Info]:
    """General rejection sampling signature

    Our actual implementation of rejection sampling in particle filtering is
    using this implementation. The input of this function is supposed to allow
    any type of advanced usage of rejection sampling. The underlying code is as follows::

        while not stop_condition:
            sample ~ distr
            update = proposal_distr(sample)
            if accept_function(sample):
                add sample

    Here we allow process functions to do extra processing on rejected and
    accepted samples. This can be useful for optimization or methods that do
    not **quite** fit this scheme.

    Will create run time :py:class:`~pomdp_belief_tracking.types.Info` and
    populate "iteration" -> # attempts and "num_accepted" -> # accepted
    particles. This is passed through all the major components, so that they in
    turn can populate or make use of the information. This is ultimately
    returned to the caller, allowing for reporting and such.

    :param stop_condition: the function controlling whether to stop sampling
    :type stop_condition: StopCondition
    :param proposal_distr: the proposal update function of samples
    :type proposal_distr: ProposalDistribution
    :param accept_function: decides whether samples are accepted
    :type accept_function: AcceptFunction
    :param distr: the initial distribution to sample from
    :type distr: Callable[[], State]
    :param process_accepted: how to process a sample once accepted, defaults to noop
    :type process_accepted: ProcessAccepted
    :param process_rejected: how to process a sample once rejected, defaults to noop
    :type process_rejected: ProcessRejected
    :return: a list of samples and run time info
    :rtype: Tuple[List[State], Info]
    """

    info: Info = {"num_accepted": 0, "iteration": 0}
    accepted: List[State] = []

    while not stop_condition(info):

        sample = distr()
        proposal, proposal_info = proposal_distr(sample, info)

        if accept_function(proposal, proposal_info, info):
            sample = process_accepted(sample, proposal_info, info)

            accepted.append(sample)
            info["num_accepted"] += 1
        else:
            process_rejected(sample, proposal_info, info)

        info["iteration"] += 1

    return accepted, info


def rejection_sample(
    sim: Simulator,
    observation_matches: Callable[[Observation, Observation], bool],
    n: int,
    process_acpt: ProcessAccepted,
    process_rej: ProcessRejected,
    initial_state_distribution: StateDistribution,
    a: Action,
    o: Observation,
) -> Tuple[ParticleFilter, Info]:
    """Implements rejection sampling

    Calls :py:func:`general_rejection_sample` with the appropriate members.
    Creates a proposal function that calls ``sim`` on a ``state`` and given
    ``a`` and accepts the sample if the simulated observation equals ``o``.
    Finally wraps the returns list of particles in a :py:class`ParticleFilter`

    :param sim: POMDP dynamics simulator
    :type sim: Simulator
    :param observation_matches: how to check for observation equality
    :type observation_matches: Callable[[Observation, Observation], bool]
    :param n: size of particle filter to return
    :type n: int
    :param process_acpt: function to call when accepting a sample
    :type process_acpt: ProcessAccepted
    :param process_rej: function to call when rejecting a sample
    :type process_rej: ProcessRejected
    :param initial_state_distribution: current / previous belief
    :type initial_state_distribution: StateDistribution
    :param a: taken action
    :type a: Action
    :param o: perceived observation
    :type o: Observation
    :return: next belief and run time information
    :rtype: Tuple[ParticleFilter, Info]
    """
    assert n > 0

    def transition_func(s: State, info: Info) -> Tuple[State, Observation]:
        """``sim`` with given ``a``"""
        return sim(s, a)

    def accept_func(s: State, ctx: Observation, info: Info) -> bool:
        """Accept samples with same observation"""
        return observation_matches(o, ctx)

    particles, info = general_rejection_sample(
        partial(have_sampled_enough, n),
        proposal_distr=transition_func,
        accept_function=accept_func,
        distr=initial_state_distribution,
        process_accepted=process_acpt,
        process_rejected=process_rej,
    )

    return ParticleFilter(particles), info


def create_rejection_sampling(
    sim: Simulator,
    n: int,
    observation_matches: Callable[[Observation, Observation], bool] = eq,
    process_acpt: ProcessAccepted = accept_noop,
    process_rej: ProcessRejected = reject_noop,
) -> BeliefUpdate:
    """Partial function that returns a regular RS belief update
    A simple wrapper around :py:func:`rejection_sample`
    :param sim: A
    :type sim: Simulator
    :param n: number of samples to accept
    :type n: int
    :param observation_matches: method to test equality between observations, defaults to eq
    :type observation_matches: Optional[Callable[[Observation, Observation], bool]]
    :param process_acpt: method to call upon accepting a sample, defaults to accept_noop
    :type process_acpt: Optional[ProcessAccepted]
    :param process_rej: method to call upon rejecting a sample, defaults to reject_noop
    :type process_rej: Optional[ProcessRejected]
    :return: rejection sampling for POMDPs
    :rtype: BeliefUpdate
    """

    def rs(
        p: StateDistribution,
        a: Action,
        o: Observation,
    ) -> Tuple[ParticleFilter, Info]:
        return rejection_sample(
            sim,
            observation_matches,
            n,
            process_acpt,
            process_rej,
            p,
            a,
            o,
        )

    return rs
