"""Defines some types for ease of reading"""

from typing import Any, Dict

from typing_extensions import Protocol

Info = Dict[str, Any]
"""The datatype used for information flow from implementation to caller"""


class Action(Protocol):
    """The abstract type representing actions"""


class Observation(Protocol):
    """The abstract type representing observations"""


class State(Protocol):
    """The abstract type representing states"""
