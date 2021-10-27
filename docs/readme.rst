=====================
POMDP belief tracking
=====================

This packages intends to be a library to be used by others. Generally it is
simply a collection of beliefs and their updates. A belief is a distribution
over the state that can be sampled from, while the update takes a current
belief, action and observation and produces a next belief.

At the moment we offer two families of implementations:

.. toctree::
   :maxdepth: 1

   exact-belief
   particle-filters

Design
======

A quick note on some design choices that have been made.

API
---

.. functional is liked

My preferred style of coding is functional, where state and mutability can be
avoided as much as possible. Hence, most of the code here is written from that
perspective, and the belief update functionality is provided through a
functional interface.

.. functional is limited from a library perspective

However, the belief is a crucial part and must be represented by some data
structure. Additionally not all belief updates can work with all beliefs. Hence
it can be much to ask for users to update and maintain them by themselves. As a
result, we provide some higher level API classes that bind these together.
These can be found in their respective modules.

Types
-----

I am unreasonably terrified of dynamic typed languages and have gone to
extremes to define as many as possible. Most of these are for internal use, but
you will come across some as a user of this library. Most of these types will
have no actual meaning, in particular:

.. autosummary::
   :nosignatures:

   pomdp_belief_tracking.types.Action
   pomdp_belief_tracking.types.Observation
   pomdp_belief_tracking.types.State

.. meaningless types: `Action`, `Observation` & `State`

Are domain specific and unimportant for implementation details. They are merely
used to allow type-checking and catching trivial bugs.

.. `Belief` type

Notable exception is any functionality *required* to do the update. These are
the actual construct that causes states to transition or provide the
probability estimates. Examples include:

.. autosummary::
   :nosignatures:

   pomdp_belief_tracking.pf.types.Simulator
   pomdp_belief_tracking.pf.types.TransitionFunction
