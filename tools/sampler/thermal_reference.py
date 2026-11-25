"""High-precision (40 digit) reference values of n, e, P of one species for tests/test_sampler.cpp (B1).
Prints the C++ initialiser rows: {m, g, T, mu, stat, n, e, P}; stat 0 Boltzmann, 1 Fermi, 2 Bose.  Requires mpmath."""
import mpmath as mp
mp.mp.dps = 40
hc = mp.mpf("0.1973269804")
def ref(m, g, T, mu, stat):
    m, g, T, mu = map(mp.mpf, (m, g, T, mu))
    def f(E):
        x = (E - mu) / T
        return mp.e**(-x) if stat == 0 else 1 / (mp.e**x + (1 if stat == 1 else -1))
    E = lambda k: mp.sqrt(k*k + m*m)
    kmax = mp.sqrt((max(m, mu) + 80*T)**2 - m*m)
    pts = [mp.mpf(0)] + [mp.sqrt(max(e*e - m*m, 0)) for e in sorted(set([m + T*t for t in (0.5, 1, 2, 4, 8, 16, 32, 64)] + [mu + T*j for j in (-20, -10, -5, -2, 0, 2, 5, 10, 20, 40) if mu + T*j > m]))] + [kmax]
    pre = g / (2*mp.pi**2) / hc**3
    n = pre * mp.quad(lambda k: k*k*f(E(k)), pts)
    e = pre * mp.quad(lambda k: k*k*E(k)*f(E(k)), pts)
    P = pre * mp.quad(lambda k: k**4/(3*E(k))*f(E(k)), pts)
    return n, e, P
cases = [(0.94, 4, 0.12, 0.85, 1), (0.94, 4, 0.02, 1.0, 1), (0.94, 4, 0.12, -0.85, 1), (0.14, 3, 0.15, 0.0, 2), (0.14, 3, 0.05, 0.0, 2),
         (1.232, 4, 0.14, 0.9, 1), (0.94, 4, 0.01, 0.97, 1), (0.94, 4, 0.1, 0.6, 0), (0.14, 3, 0.12, 0.05, 0)]
for c in cases:
    n, e, P = ref(*c)
    print("    {%s, %s, %s, %s, %d, %s, %s, %s}," % (c[0], c[1], c[2], c[3], c[4], mp.nstr(n, 20), mp.nstr(e, 20), mp.nstr(P, 20)))
# Boltzmann closed form: n = g m^2 T K2(m/T) exp(mu/T) / (2 pi^2 hc^3), P = n T, e = 3 P + m n K1/K2
for (m, g, T, mu) in [(0.94, 4, 0.1, 0.6), (0.14, 3, 0.12, 0.05)]:
    m, g, T, mu = map(mp.mpf, (m, g, T, mu))
    n = g*m*m*T*mp.besselk(2, m/T)*mp.e**(mu/T)/(2*mp.pi**2*hc**3)
    print("closed", mp.nstr(n, 20), mp.nstr(n*T, 20), mp.nstr(3*n*T + m*n*mp.besselk(1, m/T)/mp.besselk(2, m/T), 20))
