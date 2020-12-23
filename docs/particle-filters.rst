================
Particle filters
================

Particle filters (PF [particle-filtering]_) approximate distributions with
**weighted particles**. A
:class:`~pomdp_belief_tracking.pf.particle_filter.Particle` is a
:class:`~pomdp_belief_tracking.types.State`, and the weight corresponds to its
relative probability.

.. autoclass:: pomdp_belief_tracking.pf.particle_filter.Particle
   :members:
   :noindex:

The particle filter is a distribution, but provides
the additional API. Mainly some convenient constructors, and the ability to use
them as containers.

.. autoclass:: pomdp_belief_tracking.pf.particle_filter.ParticleFilter
   :members: probability_of, from_distribution, from_particles
   :special-members: __call__, __len__, __contains__, __iter__
   :noindex:

Particle filtering techniques are popular and available in large varieties. We
provide:

- :func:`~pomdp_belief_tracking.pf.rejection_sampling.general_rejection_sample`

------------------
Rejection Sampling
------------------

.. automodule:: pomdp_belief_tracking.pf.rejection_sampling
    :noindex:
