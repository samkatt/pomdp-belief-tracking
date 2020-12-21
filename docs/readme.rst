=====================
pomdp-belief-tracking
=====================

This packages intends to be a library to be used by others. We identify two
core components: belief representations and their updates. A
:class:`~pomdp_belief_tracking.types.StateDistribution` is a distribution over
the state that can be sampled from, while the
:class:`~pomdp_belief_tracking.types.BeliefUpdate` takes a current belief,
action and observation and produces a next belief.

.. beliefs

The belief at its core is distributions from which states can be sampled.
Depending on the specific type, more functionality can be expected. However,
for most purposes this definition will suffice:

.. automethod:: pomdp_belief_tracking.types.StateDistribution.__call__
   :noindex:

.. belief update

Similarly, the exact detail of the belief update will differ immensely, and
some update functions are only applicable to specific beliefs. Here we adopt
the following definition:


.. automethod:: pomdp_belief_tracking.types.BeliefUpdate.__call__
   :noindex:

Where :class:`~pomdp_belief_tracking.types.Info` is a dictionary that stores
information or context that can be populated by the belief update for reporting
and such.

.. features

.. toctree::
   :maxdepth: 2
   :caption: Features

   particle-filters
   exact-belief

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
result, we provide an actual :class:`~pomdp_belief_tracking.types.Belief` that
binds the two together.

.. autoclass:: pomdp_belief_tracking.types.Belief
   :noindex:
   :members: 

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

A notable exception is the :class:`~pomdp_belief_tracking.types.Simulator`,
which is assumed to a callable that samples transitions.

.. automethod:: pomdp_belief_tracking.types.Belief.__call__
   :noindex:
