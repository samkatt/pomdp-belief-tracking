==================
Particle filtering
==================

Particle filtering techniques are popular and available in large varieties. We
provide:

- :func:`~pomdp_belief_tracking.pf.rejection_sampling.general_rejection_sample`
- :func:`~pomdp_belief_tracking.pf.importance_sampling.general_importance_sample`

The higher-level abstraction is given by their
:class:`~pomdp_belief_tracking.pf.types.GenerativeStateDistribution`, the
belief, and the :class:`~pomdp_belief_tracking.pf.types.BeliefUpdate`. Typical
usage is as follows:

.. code-block:: python

    def sample_initial_state() -> State:
        ... # assumed known

    def sim(s, a) -> Tuple[State, Observation]:
        ... # assumed known

    def observation_equals(o1, o2) -> bool:
        ... # assumed known

    # construct belief update given assumptions above
    num_particles = 100
    belief_update = create_rejection_sampling(sim, num_particles, observation_equals)

    # construct initial belief from `sample_initial_state`
    belief = lambda: sample_initial_state()

    while True:
        a = ... # given an action
        o = ... # given an observations

        belief, info = belief_update(belief, a, o)

The rest of this document discusses particle filters and their filtering
methods in more detail.

----------------
Particle filters
----------------

.. automodule:: pomdp_belief_tracking.pf.particle_filter
    :noindex:

------------------
Rejection Sampling
------------------

.. automodule:: pomdp_belief_tracking.pf.rejection_sampling
    :noindex:

-------------------
Importance Sampling
-------------------

.. automodule:: pomdp_belief_tracking.pf.importance_sampling
    :noindex:
