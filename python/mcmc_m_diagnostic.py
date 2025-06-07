import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad
import multiprocessing
import emcee
import matplotlib.pyplot as plt

def reshape_cov(raw, n):
    arr = raw.flatten()
    if arr.size == n*n:
        return arr.reshape((n, n))
    if arr.size == n*n + 1:
        return arr[:-1].reshape((n, n))
    if arr.size == n*n + n:
        return arr[:-n].reshape((n, n))
    raise ValueError(f"Cannot reshape array of size {arr.size} into {n}×{n}")

def ripple_Hz(zp, Om, eps, omega, phi, gamma, H0):
    r    = eps * np.exp(-gamma*zp) * np.cos(omega*zp + phi)
    r0   = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    if np.any(norm <= 0.5):
        return np.inf
    return H0 * norm * np.sqrt(Om*(1+zp)**3 + (1-Om))

def dL_ripple(zarr, theta):
    Om, eps, omega, phi, gamma, H0 = theta
    c = 299792.458  # km/s
    def integrand(zp):
        # correct: c / H(z') inside the integral
        return c / ripple_Hz(zp, Om, eps, omega, phi, gamma, H0)
    return np.array([
        (1 + zi) * quad(integrand, 0, zi)[0]
        for zi in zarr
    ])

def mu0_model(zarr, theta):
    return 5 * np.log10(dL_ripple(zarr, theta)) + 25

def log_probability(theta, z, mu_obs, invC, M_locked):
    Om, eps, omega, phi, gamma, H0 = theta
    # Tighter, physically motivated priors
    if not (0.05   < Om    < 0.5):    return -np.inf
    if not (-0.01  < eps   < 0.01):   return -np.inf
    if not (0.01   < omega < 0.3):    return -np.inf
    if not (-np.pi < phi   < np.pi):  return -np.inf
    if not (0.01   < gamma < 0.3):    return -np.inf
    if not (60.0   < H0    < 75.0):   return -np.inf

    mu_pred = mu0_model(z, theta) + M_locked
    delta   = mu_obs - mu_pred
    return -0.5 * delta @ invC @ delta

def main():
    parser = argparse.ArgumentParser(description="Mini‐MCMC Diagnostic Upgrade")
    logical_cpus = multiprocessing.cpu_count()
    default_procs = max(1, logical_cpus // 2)
    parser.add_argument("-p", "--procs", type=int, default=default_procs,
                        help=f"# of processes (default half of {logical_cpus})")
    args = parser.parse_args()

    print(f"Logical CPUs available: {logical_cpus}")
    print(f"Using {args.procs} processes for MCMC")

    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, "data")
    df = pd.read_csv(os.path.join(data, "Pantheon+SH0ES.dat"), sep=r'\s+')
    z_all, mu_all = df['zCMB'].values, df['MU_SH0ES'].values
    mask = z_all > 0.023
    z, mu_obs = z_all[mask], mu_all[mask]

    n_all = len(z_all)
    C1 = reshape_cov(np.loadtxt(os.path.join(data, "Pantheon+SH0ES_STAT+SYS.cov")), n_all)
    C2 = reshape_cov(np.loadtxt(os.path.join(data, "Pantheon+SH0ES_122221_VPEC.cov")), n_all)
    C = (C1 + C2)[np.ix_(mask, mask)]
    invC = np.linalg.inv(C)

    theta0 = [0.33259, -0.00679, 0.20, 0.28128, 0.25802, 68.90867]

    mu0    = mu0_model(z, theta0)
    ones   = np.ones_like(mu_obs)
    delta0 = mu_obs - mu0
    M_analytic = float(ones @ (invC @ delta0) / (ones @ (invC @ ones)))
    sigma_M    = float(np.sqrt(1.0 / (ones @ (invC @ ones))))
    print(f"Analytic M = {M_analytic:.5f} ± {sigma_M:.5f} mag")

    M_locked = M_analytic
    ndim, nwalkers, nsteps = len(theta0), 32, 500
    p0 = np.column_stack([
        np.random.uniform(0.2, 0.4,   nwalkers),  # Om
        np.random.uniform(-0.005,0.005,nwalkers),  # eps
        np.random.uniform(0.05,0.25,  nwalkers),  # omega
        np.random.uniform(-1,1,       nwalkers),  # phi
        np.random.uniform(0.05,0.2,   nwalkers),  # gamma
        np.random.uniform(67,70,      nwalkers)   # H0
    ])

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(z, mu_obs, invC, M_locked),
        pool=multiprocessing.Pool(args.procs)
    )
    sampler.run_mcmc(p0, nsteps, progress=True)
    sampler.pool.close(); sampler.pool.join()

    samples    = sampler.get_chain(discard=nsteps//2, flat=True)
    theta_med  = np.median(samples, axis=0)
    resid      = mu_obs - (mu0_model(z, theta_med) + M_locked)
    mean_resid = float(resid.mean())
    M_adjusted = float(M_locked + mean_resid)
    print(f"Mini‐MCMC residual mean = {mean_resid:.5f} mag")
    print(f"Adjusted M_locked       = {M_adjusted:.5f} mag")

    plt.figure(figsize=(12,5))
    for i,(res,title) in enumerate(zip(
        [mu_obs - (mu0 + M_analytic), mu_obs - (mu0_model(z,theta_med) + M_adjusted)],
        ["Residuals w/ Analytic M","Residuals after Mini‐MCMC"]
    )):
        plt.subplot(1,2,i+1)
        plt.scatter(z, res, s=3, alpha=0.6)
        plt.xscale('log'); plt.axhline(0, color='red', ls='--')
        plt.title(title)
    plt.tight_layout(); plt.show()

    # sanity checks at z=0.1
    theta_test = [0.3, 0.0, 0.2, 0.0, 0.1, 70.0]
    dL_r = dL_ripple(np.array([0.1]), theta_test)[0]
    mu_r = 5*np.log10(dL_r)+25
    print(f"\nRipple    d_L(z=0.1) = {dL_r:.2f} Mpc, mu = {mu_r:.2f}")

    theta_lcdm = theta_test.copy()
    theta_lcdm[1:5] = [0,0,0,0]
    dL_l = dL_ripple(np.array([0.1]), theta_lcdm)[0]
    mu_l = 5*np.log10(dL_l)+25
    print(f"ΛCDM      d_L(z=0.1) = {dL_l:.2f} Mpc, mu = {mu_l:.2f}  (expect ~38.3)")

if __name__ == "__main__":
    main()
