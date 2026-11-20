# Equation-of-state reference prototype

`eos_reference_prototype.py` implements the pure phases of the two-phase equation of state of arXiv:1007.1448, App. A
(pion + nucleon gas with a density-dependent interaction; bag-model plasma), their phase crossing, the T = 0 quintic
join, the interface tension and the collision kinematics. Units: MeV, fm. Requires numpy and scipy.

`python eos_reference_prototype.py` rewrites `eos_reference_values.json`, the source of the reference numbers used
in the plan and in the tests. The temperature dependence of the spline matching densities is not part of the
prototype (it is not specified in the papers; work package WP2b).
