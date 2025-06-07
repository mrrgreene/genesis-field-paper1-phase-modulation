import os
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import quad
import multiprocessing
import emcee
import matplotlib.pyplot as plt
import json

def reshape_cov(raw, n):
    arr = raw.flatten()
    if arr.size == n * n:
        return arr.reshape((n, n))
    if arr.size == n * n + 1:
        return arr[:-1].reshape((n, n))
    if arr.size == n * n + n:
        return arr[:-n].reshape((n, n))
    raise ValueError(f"Cannot reshape array of size {arr.size} into {n}×{n}")

def ripple_Hz(zp, Om, eps, omega, phi, gamma, H0):
    r = eps * np.exp(-gamma * zp) * np.cos(omega * zp + phi)
    r0 = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    if np.any(norm <= 0.5):
        return np.inf
    return H0 * norm * np.sqrt(Om * (1 + zp)**3 + (1 - Om))

def dL_ripple(zarr, theta):
    Om, eps, omega, phi, gamma, H0 = theta
    c = 299792.458  # km/s
    def integrand(zp):
        return c / ripple_Hz(zp, Om, eps, omega, phi, gamma, H0)
    return np.array([(1 + zi) * quad(integrand, 0, zi)[0] for zi in zarr])

def mu0_model(zarr, theta):
    return 5 * np.log10(dL_ripple(zarr, theta)) + 25

def log_probability(theta, z, mu_obs, invC, M_locked):
    Om, eps, omega, phi, gamma, H0 = theta
    if not (0.05 < Om < 0.5): return -np.inf
    if not (-0.01 < eps < 0.01): return -np.inf
    if not (0.01 < omega < 0.3): return -np.inf
    if not (-np.pi < phi < np.pi): return -np.inf
    if not (0.01 < gamma < 0.3): return -np.inf
    if not (60.0 < H0 < 75.0): return -np.inf
    mu_pred = mu0_model(z, theta) + M_locked
    delta = mu_obs - mu_pred
    return -0.5 * delta @ invC @ delta

def main():
    parser = argparse.ArgumentParser(description="Genesis Field MCMC Residual Comparison")
    logical_cpus = multiprocessing.cpu_count()
    default_procs = max(1, logical_cpus // 2)
    parser.add_argument("-p", "--procs", type=int, default=default_procs)
    args = parser.parse_args()

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
    mu0 = mu0_model(z, theta0)
    ones = np.ones_like(mu_obs)
    delta0 = mu_obs - mu0
    M_analytic = float(ones @ (invC @ delta0) / (ones @ (invC @ ones)))
    print(f"Analytic M = {M_analytic:.5f} mag")

    M_locked = M_analytic
    ndim, nwalkers, nsteps = len(theta0), 32, 500
    p0 = np.column_stack([
        np.random.uniform(0.2, 0.4, nwalkers),
        np.random.uniform(-0.005, 0.005, nwalkers),
        np.random.uniform(0.05, 0.25, nwalkers),
        np.random.uniform(-1, 1, nwalkers),
        np.random.uniform(0.05, 0.2, nwalkers),
        np.random.uniform(67, 70, nwalkers)
    ])

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(z, mu_obs, invC, M_locked),
        pool=multiprocessing.Pool(args.procs)
    )
    sampler.run_mcmc(p0, nsteps, progress=True)
    sampler.pool.close(); sampler.pool.join()

    samples = sampler.get_chain(discard=nsteps//2, flat=True)
    theta_med = np.median(samples, axis=0)
    resid = mu_obs - (mu0_model(z, theta_med) + M_locked)

    # ΛCDM comparison
    theta_lcdm = theta_med.copy()
    theta_lcdm[1:5] = [0.0, 0.0, 0.0, 0.0]
    mu_lcdm = mu0_model(z, theta_lcdm) + M_locked
    resid_lcdm = mu_obs - mu_lcdm

    # Plot residual comparison
    plt.figure(figsize=(10, 5))
    plt.scatter(z, resid, s=8, color='black', alpha=0.7, label='Genesis Field')
    plt.plot(z, resid_lcdm, 'r--', linewidth=1.5, label=r'$\Lambda$CDM')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"$\mu_{\rm obs} - \mu_{\rm model}$")
    plt.xscale('log')
    plt.title(r"Residual Comparison: Genesis Field vs. $\Lambda$CDM")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Compute fit statistics for Genesis Field ===
    k = ndim
    n = len(z)
    chi2 = float(resid @ invC @ resid)
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    rms_resid = float(np.std(resid))

    # === Compute fit statistics for ΛCDM ===
    # In ΛCDM, only Om and H0 are free (2 parameters)
    chi2_lcdm = float(resid_lcdm @ invC @ resid_lcdm)
    k_lcdm = 2
    aic_lcdm = chi2_lcdm + 2 * k_lcdm
    bic_lcdm = chi2_lcdm + k_lcdm * np.log(n)
    rms_resid_lcdm = float(np.std(resid_lcdm))

    print("\n=== Fit Statistics ===")
    print(f"Genesis Field:")
    print(f"  χ²              = {chi2:.2f}")
    print(f"  AIC             = {aic:.2f}")
    print(f"  BIC             = {bic:.2f}")
    print(f"  χ²/dof          = {chi2 / (n - k):.3f}")
    print(f"  Residual RMS    = {rms_resid:.5f} mag\n")

    print(f"ΛCDM:")
    print(f"  χ²              = {chi2_lcdm:.2f}")
    print(f"  AIC             = {aic_lcdm:.2f}")
    print(f"  BIC             = {bic_lcdm:.2f}")
    print(f"  χ²/dof          = {chi2_lcdm / (n - k_lcdm):.3f}")
    print(f"  Residual RMS    = {rms_resid_lcdm:.5f} mag")

    # === Export summary to JSON ===
    theta_std = np.std(samples, axis=0)
    param_names = ["Omega_m", "eps", "omega", "phi", "gamma", "H0"]

    summary = {
        "M_locked": M_locked,
        "best_fit": {k: float(v) for k, v in zip(param_names, theta_med)},
        "uncertainty": {k: float(e) for k, e in zip(param_names, theta_std)},
        "genesis_field_fit_statistics": {
            "chi2": chi2,
            "aic": aic,
            "bic": bic,
            "chi2_dof": chi2 / (n - k),
            "residual_rms": rms_resid
        },
        "lcdm_fit_statistics": {
            "chi2": chi2_lcdm,
            "aic": aic_lcdm,
            "bic": bic_lcdm,
            "chi2_dof": chi2_lcdm / (n - k_lcdm),
            "residual_rms": rms_resid_lcdm
        },
        "ripple_suppressed": bool(abs(theta_med[1]) < theta_std[1])
    }

    os.makedirs("output", exist_ok=True)
    with open("output/sn_mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Summary saved to output/sn_mcmc_summary.json")

if __name__ == "__main__":
    main()
