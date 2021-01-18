"""Provides a general implementation of rejection sampling:

.. autofunction:: general_rejection_sample
   :noindex:

Our belief update version of rejection sampling calls this function with the
appropriate parameters:

.. autofunction:: rejection_sample
   :noindex:

Which can best be created through our construction function (with sane
defaults, otherwise you can also apply partial):

.. autofunction:: create_rejection_sampling
   :noindex:

This package provides some useful functions to extend the belief update. Most
notably:

- :class:`AcceptionProgressBar`

.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
   Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
   signal processing magazine, 20(5), 19-38.

"""

from __future__ import annotations

from functools import partial
from operator import eq
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm  # type: ignore
from typing_extensions import Protocol

import pomdp_belief_tracking.pf.particle_filter
import pomdp_belief_tracking.pf.types
from pomdp_belief_tracking.pf.particle_filter import ParticleFilter
from pomdp_belief_tracking.types import (
    Action,
    BeliefUpdate,
    Info,
    Observation,
    Simulator,
    State,
    StateDistribution,
)


class StopCondition(Protocol):
    """Whether to 'stop' sampling protocol"""

    def __call__(self, info: Info) -> bool:
        """The signature for the sampling stop condition

        This function is called at the start of each sampling attempt, provided
        some runtime ``info``, this function decides whether to continue
        sampling

        :param info: run time information
        :return: true if the condition for stopping is met
        """


def have_sampled_enough(desired_num: int, info: Info) -> bool:
    """Returns ``true`` if "num_accepted" in ``info`` has reached ``desired_num``

    Given ``desired_num``, this implements :class`StopCondition`

    :param desired_num: number of desired samples
    :param info: run time info (should have an entry "num_accepted" -> int)
    :return: true if number of accepted samples is greater or equal to ``desired_num``
    """
    assert desired_num > 0 and info["rejection_sampling_num_accepted"] >= 0

    return desired_num <= info["rejection_sampling_num_accepted"]


class AcceptFunction(Protocol):
    """The signature of the acceptance function in general rejection sampling"""

    def __call__(self, s: State, ctx: Any, info: Info) -> bool:
        """Returns whether the sample ``s`` is accepted given some context ``ctx``

        :param s: the updated sample to be tested
        :param ctx: output of :func:`AcceptFunction`
        :param info: run time information
        :return: whether the sample is accepted
        """


class ProcessRejected(Protocol):
    """A function that processes a rejected sample in general rejection sampling"""

    def __call__(self, s: State, ctx: Any, info: Info) -> None:
        """Processes a rejected sample ``s`` with some context ``ctx``

        An example usage could be that if, for optimizations sake, the sample
        ``s`` was not copied from the original distribution, then this can
        reset

        :param s: the rejected sample
        :param ctx: output of :func:`AcceptFunction`
        :param info: run time information
        :return: Nothing, only side effects
        """


def reject_noop(s: State, ctx: Any, info: Info) -> None:
    """A placeholder for :class:`ProcessRejected`: does nothing

    :param s: the accepted sample
    :param ctx: the output of :func:`AcceptFunction`
    :param info: run time information (ignored)
    :return: Only side effects
    """


class ProcessAccepted(Protocol):
    """A function that processes an accepted sample in general rejection sampling"""

    def __call__(self, s: State, ctx: Any, info: Info) -> State:
        """Processes an accepted sample ``s`` with some context ``ctx``

        An example usage could be that if, for optimizations sake, the sample
        ``s`` was not copied from the original distribution, then this can
        reset it and return a copy

        :param s: accepted sample
        :param ctx: output of :func:`AcceptFunction`
        :param info: run time information
        :return: processed sample
        """


def accept_noop(s: State, ctx: Any, info: Info) -> State:
    """A placeholder for :class:`ProcessAccepted`: does nothing

    :param s: the accepted sample (ignored)
    :param ctx: the output of :func:`AcceptFunction` (ignored)
    :param info: run time information (ignored)
    :return: ``s``
    """
    return s


class AcceptionProgressBar(ProcessAccepted):
    """A :class:`ProcessAccepted` call that prints out a progress bar

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
        :param ctx: context of acception (ignored)
        :param info: run time information (# accepted samples)
        :return: ``s`` as input
        """

        if info["rejection_sampling_num_accepted"] == 0:
            # the first sample is accepted, LGTM
            self.pbar = tqdm(total=self._total_expected_calls)

        assert self.pbar
        self.pbar.update()

        if info["rejection_sampling_num_accepted"] == self._total_expected_calls - 1:
            # last sample accepted!
            self.pbar.close()

        return s


def general_rejection_sample(
    stop_condition: StopCondition,
    proposal_distr: pomdp_belief_tracking.pf.types.ProposalDistribution,
    accept_function: AcceptFunction,
    distr: StateDistribution,
    process_accepted: ProcessAccepted = accept_noop,
    process_rejected: ProcessRejected = reject_noop,
) -> Tuple[List[State], Info]:
    """General rejection sampling signature

    Our actual implementation of rejection sampling in particle filtering is
    using this implementation. The input of this function is supposed to allow
    any type of advanced usage of rejection sampling. The underlying code is as follows::

        while not stop_condition:
            sample ~ distr
            proposal = proposal_distr(sample)
            if accept_function(proposal):
                add proposal

    Here we allow process functions to do extra processing on rejected and
    accepted samples. This can be useful for optimization or methods that do
    not **quite** fit this scheme.

    Will create run time :class:`~pomdp_belief_tracking.types.Info` and
    populate "iteration" -> # attempts and "num_accepted" -> # accepted
    particles. This is passed through all the major components, so that they in
    turn can populate or make use of the information. This is ultimately
    returned to the caller, allowing for reporting and such.

    :param stop_condition: the function controlling whether to stop sampling
    :param proposal_distr: the proposal update function of samples
    :param accept_function: decides whether samples are accepted
    :param distr: the initial distribution to sample from
    :param process_accepted: how to process a sample once accepted, default :func:`accept_noop`
    :param process_rejected: how to process a sample once rejected, default :func:`reject_noop`
    :return: a list of samples and run time info
    """

    info: Info = {
        "rejection_sampling_num_accepted": 0,
        "rejection_sampling_iteration": 0,
    }
    accepted: List[State] = []

    while not stop_condition(info):

        sample = distr()

        proposal, proposal_info = proposal_distr(sample, info)

        if accept_function(proposal, proposal_info, info):
            accepted_proposal = process_accepted(proposal, proposal_info, info)

            accepted.append(accepted_proposal)
            info["rejection_sampling_num_accepted"] += 1
        else:
            process_rejected(proposal, proposal_info, info)

        info["rejection_sampling_iteration"] += 1

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

    Calls :func:`general_rejection_sample` with the appropriate members.
    Creates a proposal function that calls ``sim`` on a ``state`` and given
    ``a`` and accepts the sample if the simulated observation equals ``o``.
    Finally wraps the returns list of particles in a :class`ParticleFilter`

    :param sim: POMDP dynamics simulator
    :param observation_matches: how to check for observation equality
    :param n: size of particle filter to return
    :param process_acpt: function to call when accepting a sample
    :param process_rej: function to call when rejecting a sample
    :param initial_state_distribution: current / previous belief
    :param a: taken action
    :param o: perceived observation
    :return: next belief and run time information
    """
    assert n > 0

    def transition_func(s: State, info: Info) -> Tuple[State, Dict[str, Any]]:
        """``sim`` with given ``a``"""
        ss, o = sim(s, a)

        return ss, {"state": s, "action": a, "observation": o}

    def accept_func(s: State, ctx: Dict[str, Any], info: Info) -> bool:
        """Accept samples with same observation"""
        return observation_matches(o, ctx["observation"])

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

    A simple wrapper around :func:`rejection_sample`

    :param sim: generative POMDP
    :param n: number of samples to accept
    :param observation_matches: method to test equality between observations, defaults to eq
    :param process_acpt: method to call upon accepting a sample, defaults to :func:`accept_noop`
    :param process_rej: method to call upon rejecting a sample, defaults to :func:`reject_noop`
    :return: rejection sampling for POMDPs
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
