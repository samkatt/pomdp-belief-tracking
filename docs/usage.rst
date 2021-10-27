=====
Usage
=====


Here we give an example on how to track the belief in the canonical tiger
problem. First we define the environment:

.. literalinclude:: ../tests/test_integration.py
   :pyobject: Tiger

Then given some beliefs:

.. literalinclude:: ../tests/test_integration.py
   :pyobject: uniform_tiger_belief

We can update the belief according to, for example, rejection sampling::

    from pomdp_belief_tracking.pf.rejection_sampling import (
        ParticleFilter,
        accept_noop,
        create_rejection_sampling,
        reject_noop,
    )

    belief_update = create_rejection_sampling(
        Tiger.sim,
        100,
        process_acpt=accept_noop,
        process_rej=reject_noop,
    )

    b, run_time_info = belief_update(uniform_tiger_belief, Tiger.H, Tiger.L)

Or importance sampling::

    from pomdp_belief_tracking.pf.importance_sampling import create_importance_sampling

    n = 100

    def trans_func(s, a):
        return Tiger.sim(s, a)[0]

    def obs_func(s, a, ss, o):
        return Tiger.observation_model(a, ss)[o]

    belief_update = create_importance_sampling(trans_func, obs_func, n)

    b, run_time_info = belief_update(uniform_tiger_belief, Tiger.H, Tiger.L)
