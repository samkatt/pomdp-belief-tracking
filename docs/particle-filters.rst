================
Particle filters
================

Particle filters (PF [particle-filtering]_) approximate distributions with
**weighted particles**. A :py:class:`~pomdp_belief_tracking.pf.Particle` is a
:py:class:`~pomdp_belief_tracking.types.State`, and the weight corresponds to
its relative probability.

.. autoclass:: pomdp_belief_tracking.pf.Particle
   :members:
   :noindex:

The particle filter is a distribution, but provides
the additional API. Mainly some convenient constructors, and the ability to use
them as containers.

.. autoclass:: pomdp_belief_tracking.pf.ParticleFilter
   :members: probability_of, from_distribution, from_particles
   :special-members: __call__, __len__, __contains__, __iter__
   :noindex:

------------------------------
Particle filter belief updates
------------------------------

Particle filtering techniques are popular and available in large varieties. We
provide rejection sampling:

.. autofunction:: pomdp_belief_tracking.pf.rejection_sample
   :noindex:

Which can best be created, given a simulator and observation equality function
through::

    partial(rejection_sample, sim, observation_matches, num_samples)


.. autofunction:: pomdp_belief_tracking.pf.create_rejection_sampling
   :noindex:

This package provides some useful functions to extend the belief update. Most
notably:

- For :py:class:`~pomdp_belief_tracking.pf.rejection_sample`
    - :py:class:`~pomdp_belief_tracking.pf.AcceptionProgressBar`

.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
   Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
   signal processing magazine, 20(5), 19-38.
