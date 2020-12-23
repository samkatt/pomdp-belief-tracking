"""Particle Filters and their update functions

Particle filters are approximations of distributions. They represent them
through :class:`~pomdp_belief_tracking.pf.particle_filter.Particle`, a
:class:`~pomdp_belief_tracking.types.State` with a relative weight.

In this module we implement such a
:class:`~pomdp_belief_tracking.pf.particle_filter.ParticleFilter` and some
functions to update them:

    - :func:`~pomdp_belief_tracking.pf.rejection_sampling.rejection_sample`
"""
