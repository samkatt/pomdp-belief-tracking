=====================
pomdp-belief-tracking
=====================


.. image:: https://travis-ci.com/samkatt/pomdp-belief-tracking.svg?branch=main
       :target: https://travis-ci.com/samkatt/pomdp-belief-tracking

.. image:: https://readthedocs.org/projects/pomdp-belief-tracking/badge/?version=latest
        :target: https://pomdp-belief-tracking.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Methods for tracking and maintaining state estimates in POMDPs, hosted on Github_

* Free software: MIT license
* Documentation: https://pomdp-belief-tracking.readthedocs.io.

.. POMDPs

Partially observable Markov decision processes (POMDP
[kaelbling_planning_1998]_) is a mathematical framework for defining
reinforcement learning (RL) in environments with hidden state. To solve the RL
problem means to come up with a policy, a mapping from the past observations of
the environment to an action.

.. online planning

Online planning is the family of methods that assumes access to (a simulator
of) the dynamics and aims to figure out what action to take *during execution*.
For this it requires a belief, a probability distribution over the current
state. The planner takes a current belief of the current state of the
environment and a simulator, and spits out its favorite action.

This library provides APIs and implementation of **beliefs**.

.. todo::

  - Write out features
  - Link to documentation
  - Link to usage

.. [kaelbling_planning_1998] Kaelbling, Leslie Pack, Michael L. Littman, and
   Anthony R. Cassandra. “Planning and acting in partially observable
   stochastic domains.“ Artificial intelligence 101.1-2 (1998): 99-134.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Github: https://github.com/samkatt/pomdp-belief-tracking
