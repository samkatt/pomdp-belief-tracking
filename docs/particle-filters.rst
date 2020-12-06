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
   :members:
   :special-members:
   :noindex:

------------------------------
Particle filter belief updates
------------------------------

Particle filtering techniques are popular and available in large varieties. We
provide rejection sampling:

.. autofunction:: pomdp_belief_tracking.pf.rejection_sample
   :noindex:


.. [particle-filtering] Djuric, P. M., Kotecha, J. H., Zhang, J., Huang, Y.,
   Ghirmai, T., Bugallo, M. F., & Miguez, J. (2003). Particle filtering. IEEE
   signal processing magazine, 20(5), 19-38.
