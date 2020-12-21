================
Particle filters
================

Particle filters (PF [particle-filtering]_) approximate distributions with
**weighted particles**. A :class:`~pomdp_belief_tracking.pf.Particle` is a
:class:`~pomdp_belief_tracking.types.State`, and the weight corresponds to its
relative probability.

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

Particle filtering techniques are popular and available in large varieties. We
provide:

- :func:`~pomdp_belief_tracking.pf.general_rejection_sample`

------------------
Rejection Sampling
------------------

We provide a general implementation of rejection sampling:

.. autofunction:: pomdp_belief_tracking.pf.general_rejection_sample
   :noindex:

Our belief update version of rejection sampling calls this function with the
appropriate parameters:

.. autofunction:: pomdp_belief_tracking.pf.rejection_sample
   :noindex:

Which can best be created through our construction function (with sane
defaults, otherwise you can also apply partial):

.. autofunction:: pomdp_belief_tracking.pf.create_rejection_sampling
   :noindex:

This package provides some useful functions to extend the belief update. Most
notably:

- For :class:`~pomdp_belief_tracking.pf.rejection_sample`
    - :class:`~pomdp_belief_tracking.pf.AcceptionProgressBar`

.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
   Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
   signal processing magazine, 20(5), 19-38.
