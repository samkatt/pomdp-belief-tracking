"""Some additional (shared) particle filtering types"""
from __future__ import annotations

from typing import Any, Tuple

from typing_extensions import Protocol

from pomdp_belief_tracking.types import Info, State


class ProposalDistribution(Protocol):
    """The signature for the proposal distribution for sampling"""

    def __call__(self, s: State, info: Info) -> Tuple[State, Any]:
        """Proposes an 'updated' sample from some initial ``s``

        :param s: a sample at t
        :type s: State
        :param info: run time information
        :type info: Info
        :return: an updated sample at t+1 and any additional context
        :rtype: Tuple[State, Any]
        """
