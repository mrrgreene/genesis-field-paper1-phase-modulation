# mcmc_fixedM_upgraded.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import emcee
import corner
import os
import multiprocessing
import json

# === Reproducibility ===
np.random.seed(42)

# === Constants ===
c = 299792.458  # km/s
M_locked = -0.07256  # Updated from diagnostic calibration

# === Load Pantheon+SH0ES data with low-z cut ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "Pantheon+SH0ES.dat")
df = pd.read_csv(data_path, sep=r'\s+')[['zCMB', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']]
df.columns = ['z', 'mu', 'sigma_mu']
df = df[(df['sigma_mu'] < 0.15) & (df['z'] > 0.023)]  # Apply low-z cut
z, mu_obs, sigma_mu = df['z'].values, df['mu'].values, df['sigma_mu'].values

# === Ripple H(z) Model ===
def ripple_Hz(z, Om, eps, omega, phi, gamma, H0):
    r    = eps * np.exp(-gamma * z) * np.cos(omega * z + phi)
    r0   = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    if np.any(norm <= 0.5):
        return np.inf
    return H0 * norm * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# === Luminosity Distance (corrected) ===
def dL_ripple(z, Om, eps, omega, phi, gamma, H0):
    def integrand(zp):
        return c / ripple_Hz(zp, Om, eps, omega, phi, gamma, H0)
    integral = np.array([quad(integrand, 0, zi, epsabs=1e-6)[0] for zi in z])
    return (1 + z) * integral  # in Mpc

# === Distance Modulus Model ===
def mu_model(z, Om, eps, omega, phi, gamma, H0):
    dL = dL_ripple(z, Om, eps, omega, phi, gamma, H0)
    return 5 * np.log10(dL) + 25 + M_locked

# === Log-Likelihood ===
def log_likelihood(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    mu_pred = mu_model(z, Om, eps, omega, phi, gamma, H0)
    if np.any(np.isnan(mu_pred)) or np.any(mu_pred > 200):
        return -np.inf
    return -0.5 * np.sum(((mu_obs - mu_pred) / sigma_mu) ** 2)

# === Prior (tightened) ===
def log_prior(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    if (
        0.05 < Om < 0.5 and
        -0.01 < eps < 0.01 and
        0.01 < omega < 0.3 and
        -np.pi < phi < np.pi and
        0.01 < gamma < 0.3 and
        60.0 < H0 < 75.0
    ):
        return 0.0
    return -np.inf

# === Posterior ===
def log_probability(theta):
    lp = log_prior(theta)
    return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

# === MCMC Setup ===
ndim, nwalkers = 6, 64

# Initialize walkers uniformly within tightened priors
pos = np.column_stack([
    np.random.uniform(0.2, 0.4,   nwalkers),  # Om
    np.random.uniform(-0.005,0.005,nwalkers),  # eps
    np.random.uniform(0.05,0.25,  nwalkers),  # omega
    np.random.uniform(-1,1,       nwalkers),  # phi
    np.random.uniform(0.05,0.2,   nwalkers),  # gamma
    np.random.uniform(67,71,      nwalkers)   # H0
])

# === Run MCMC ===
if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(f"Running MCMC with locked M = {M_locked:.5f} ...")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, 30000, progress=True)

    # Save raw chains
    np.save("sn_chain_mcmc_pantheon.npy", sampler.get_chain())
    np.save("sn_log_prob_mcmc_pantheon.npy", sampler.get_log_prob())

    # Posterior samples and corner plot
    samples = sampler.get_chain(discard=5000, thin=10, flat=True)
    labels = ["Ωₘ", "ε", "ω", "φ", "γ", "H₀"]
    fig = corner.corner(samples, labels=labels, truths=np.median(samples, axis=0))
    plt.show()

    # === Best-fit parameters ===
    param_names = labels
    best_fit = np.median(samples, axis=0)
    uncertainty = np.std(samples, axis=0)
    print("\n=== Best-Fit Parameters (Median ± 1σ) ===")
    for name, val, err in zip(param_names, best_fit, uncertainty):
        print(f"{name} = {val:.5f} ± {err:.5f}")

    # === Residuals Plot ===
    mu_pred = mu_model(z, *best_fit)
    residuals = mu_obs - mu_pred
    plt.figure(figsize=(10, 4))
    plt.errorbar(z, residuals, yerr=sigma_mu, fmt='o', color='black', ecolor='gray', alpha=0.8)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"Residual $[\mu_{\mathrm{obs}} - \mu_{\mathrm{model}}]$")
    plt.title(f"Residuals with Locked M = {M_locked}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Fit Statistics ===
    n = len(z)
    k = ndim
    chi2 = np.sum((residuals / sigma_mu) ** 2)
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)

    print("\n=== Fit Statistics (Locked M) ===")
    print(f"χ²              = {chi2:.2f}")
    print(f"AIC             = {aic:.2f}")
    print(f"BIC             = {bic:.2f}")
    print(f"χ²/dof          = {chi2 / (n - k):.3f}")

    # === Post-Run Diagnostics ===

    # 1. Sanity check at z = 0.1
    z_test = np.array([0.1])
    dL_check = dL_ripple(z_test, *best_fit)[0]
    mu_check = 5 * np.log10(dL_check) + 25 + M_locked
    print(f"\n✅ Sanity Check (z=0.1): dL = {dL_check:.2f} Mpc, mu = {mu_check:.2f}  (expected ~38.3)")

    # 2. Residual statistics
    mean_resid = float(np.mean(residuals))
    rms_resid = float(np.std(residuals))
    print(f"✅ Residual mean = {mean_resid:.5f} mag")
    print(f"✅ Residual RMS  = {rms_resid:.5f} mag")

    # 3. Parameter edge check
    param_bounds = [
        (0.05, 0.5),
        (-0.01, 0.01),
        (0.01, 0.3),
        (-np.pi, np.pi),
        (0.01, 0.3),
        (60.0, 75.0)
    ]
    edge_flags = {}
    for name, (lo, hi), val in zip(labels, param_bounds, best_fit):
        edge_flags[name] = not (lo < val < hi)
        if edge_flags[name]:
            print(f"🚨 Warning: {name} = {val:.5f} is near its prior boundary [{lo}, {hi}].")

    # === Save summary to JSON with diagnostics ===
    summary = {
        "M_locked": M_locked,
        "best_fit": {k: float(v) for k, v in zip(param_names, best_fit)},
        "uncertainty": {k: float(v) for k, v in zip(param_names, uncertainty)},
        "chi2": chi2,
        "aic": aic,
        "bic": bic,
        "chi2_dof": chi2 / (n - k),
        "diagnostics": {
            "z=0.1 check": {"dL_Mpc": float(dL_check), "mu": float(mu_check)},
            "residuals": {"mean": mean_resid, "rms": rms_resid},
            "param_edge_flags": edge_flags
        }
    }
    with open("sn_mcmc_pantheon_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n✅ Saved mcmc_summary.json with diagnostics")
