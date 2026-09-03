.. _time reversal symmetry:

Time-reversal symmetry
======================

Time reversal is represented by an additional :math:`\mathbb{Z}_2^T` parity. An irrep of :math:`O(3) \times \mathbb{Z}_2^T` is labeled by :math:`(l, p, t)`, where :math:`p` is the spatial parity and :math:`t` is the time-reversal parity.

In the string notation, ``e`` denotes :math:`+1` and ``o`` denotes :math:`-1`. The first letter is the spatial parity and the second letter is the time-reversal parity. For example, a position vector is ``1oe`` and a spin vector is ``1eo``.

.. jupyter-execute::

    from e3nn import o3

    position = o3.Irrep("1oe")
    spin = o3.Irrep("1eo")

For compatibility, the time-reversal label can be omitted and defaults to even.

.. jupyter-execute::

    o3.Irrep("1o") == o3.Irrep("1oe")

Representations
---------------

The representation matrix of an irrep is

.. math::

    p^k t^{k_T} D^l(\alpha, \beta, \gamma),

where :math:`k` and :math:`k_T` indicate whether spatial inversion and time reversal are applied. They can be specified independently with `e3nn.o3.Irrep.D_from_angles`.

.. jupyter-execute::

    import torch

    spin.D_from_angles(
        alpha=torch.tensor(0.0),
        beta=torch.tensor(0.0),
        gamma=torch.tensor(0.0),
        k=torch.tensor(0),
        kt=torch.tensor(1),
    )

Tensor products
---------------

Tensor products multiply both spatial and time-reversal parities. In particular,

.. math::

    p_\mathrm{out} = p_1 p_2, \qquad t_\mathrm{out} = t_1 t_2.

.. jupyter-execute::

    list(spin * position)

Linear maps only connect irreps with the same :math:`l`, :math:`p`, and :math:`t`. Pointwise nonlinearities follow the same parity rules for spatial inversion and time reversal.

Spherical harmonics
-------------------

If the input vector has parities :math:`p` and :math:`t`, spherical harmonics of degree :math:`l` have parities :math:`p^l` and :math:`t^l`. Set ``time_reversal=True`` when the input vector is odd under time reversal.

.. jupyter-execute::

    spherical_harmonics = o3.SphericalHarmonics(
        [0, 1, 2],
        normalize=True,
        time_reversal=True,
    )
    spherical_harmonics.irreps_in, spherical_harmonics.irreps_out

Equivariance testing
--------------------

`e3nn.util.test.equivariance_error` and `e3nn.util.test.assert_equivariant` test time reversal when ``do_time_reversal=True``. The special input type ``'spin'`` represents an axial vector that is odd under time reversal. Set ``do_only_rot_spin=True`` to additionally test independent global spin rotations.
