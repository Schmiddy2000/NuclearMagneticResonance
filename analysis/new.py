from scipy.constants import hbar, e, m_p

gamma = 250.20 * 1e6
gamma_upper = (250.20 + 3.29) * 1e6
gamma_lower = (250.2 - 0.38) * 1e6
mu_N = hbar * e / (2 * m_p)

mu_z = gamma * 1/2 * hbar
mu_z_u = gamma_upper * 1/2 * hbar
mu_z_l = gamma_lower * 1/2 * hbar

r = mu_z / mu_N
r_u = mu_z_u / mu_N
r_l = mu_z_l / mu_N

print(r)
print(r - r_l)
print(r_u - r)