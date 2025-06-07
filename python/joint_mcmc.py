# joint_mcmc_with_acdm_comparison.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import emcee
import corner
import os
import multiprocessing
import json

# === Reproducibility ===
np.random.seed(42)

# === Constants ===
c = 299792.458  # speed of light in km/s
M_locked = -0.07256  # Absolute magnitude offset fixed from SH0ES calibration

# === Load Pantheon+SH0ES Supernova Data with Low-z Cut ===
script_dir = os.path.dirname(os.path.abspath(__file__))
pantheon_path = os.path.join(script_dir, "data", "Pantheon+SH0ES.dat")
df_sn = pd.read_csv(pantheon_path, sep=r'\s+')[["zCMB", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"]]
df_sn.columns = ['z', 'mu', 'sigma_mu']
df_sn = df_sn[df_sn['z'] > 0.023]  # Apply low-redshift cut
z_sn = df_sn['z'].values
mu_obs = df_sn['mu'].values
sigma_mu = df_sn['sigma_mu'].values

# === Load and Combine H(z) Data ===
bao_df = pd.DataFrame({
    "z":         [0.106, 0.15, 0.32, 0.57, 0.61, 0.73, 2.34],
    "Hz":        [69.0, 67.0, 79.2, 96.8, 97.3, 97.9, 222.0],
    "sigma_Hz":  [19.6, 12.0, 5.6, 3.4, 2.1, 2.7, 7.0]
})
cc_df = pd.DataFrame({
    "z":         [0.07, 0.12, 0.17, 0.179, 0.199, 0.27, 0.4, 0.48, 0.88, 1.3, 1.75, 2.0],
    "Hz":        [69,   68.6, 83,   75,    75,    77,   95,  97,   90,   168,  202,   222],
    "sigma_Hz":  [19.6, 26.2, 8,    4.9,   5,     14,   17,  60,   40,   17,   40,    41]
})
farooq_df = pd.DataFrame({
    "z":         [0.07, 0.09, 0.12, 0.17, 0.179, 0.199, 0.2, 0.27, 0.28, 0.352, 0.4,
                  0.44, 0.48, 0.57, 0.593, 0.6, 0.68, 0.73, 0.781, 0.875, 0.88, 0.9,
                  1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965, 2.34, 2.36],
    "Hz":        [69,   69,   68.6, 83,   75,    75,    72.9, 77,  88.8, 83,    95,
                  82.6, 97,  96.8, 104,  87.9,  92,    97.3,105,   125,   90,    117,
                  154,  168, 160,   177,  140,   202,  186.5, 222,   226],
    "sigma_Hz": [19.6, 12,   26.2, 8,    4.9,   5,     29.6, 14,  36.6, 14,    17,
                 7.8,  60,   3.4,  13,   6.1,   8,     7,    12,   17,    40,    23,
                 20,   17,   33.6, 18,   14,    40,    50.4, 7,     8]
})

hz_all = pd.concat([bao_df, cc_df, farooq_df], ignore_index=True)
hz_clean = hz_all.drop_duplicates(subset="z", keep="first").sort_values(by="z").reset_index(drop=True)
z_Hz = hz_clean['z'].values
Hz_obs = hz_clean['Hz'].values
sigma_Hz = hz_clean['sigma_Hz'].values

# === Ripple-modulated H(z) Model ===
def ripple_Hz(z_array, Om, eps, omega, phi, gamma, H0):
    """
    Returns H(z) = H0 * [1 + eps * exp(-gamma * z) * cos(omega * z + phi)] /
                     [1 + eps * cos(phi)] * sqrt(Om * (1+z)^3 + (1-Om))
    """
    r = eps * np.exp(-gamma * z_array) * np.cos(omega * z_array + phi)
    r0 = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    # Prevent unphysical normalization
    if np.any(norm <= 0.5):
        return np.full_like(z_array, np.inf)
    return H0 * norm * np.sqrt(Om * (1 + z_array)**3 + (1 - Om))

# === Fast Interpolated Luminosity Distance with Ripple ===
def dL_ripple(z_array, Om, eps, omega, phi, gamma, H0):
    z_grid = np.linspace(0, 2.5, 400)
    Hz_grid = ripple_Hz(z_grid, Om, eps, omega, phi, gamma, H0)
    if np.any(np.isnan(Hz_grid)) or np.any(Hz_grid <= 0):
        return np.full_like(z_array, np.nan)
    # Comoving distance integral
    D_C = cumulative_trapezoid(c / Hz_grid, z_grid, initial=0.0)
    D_L = (1 + z_grid) * D_C
    DL_interp = interp1d(z_grid, D_L, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return DL_interp(z_array)

# === Distance Modulus Model for SNe ===
def mu_model(z_array, Om, eps, omega, phi, gamma, H0):
    dL = dL_ripple(z_array, Om, eps, omega, phi, gamma, H0)
    return 5 * np.log10(dL) + 25 + M_locked

# === Log-Likelihood for Pantheon+ SNe ===
def log_likelihood_sn(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    mu_pred = mu_model(z_sn, Om, eps, omega, phi, gamma, H0)
    if np.any(np.isnan(mu_pred)) or np.any(mu_pred > 200):
        return -np.inf
    chi2_sn = np.sum(((mu_obs - mu_pred) / sigma_mu) ** 2)
    return -0.5 * chi2_sn

# === Log-Likelihood for H(z) Chronometers ===
def log_likelihood_Hz(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    Hz_pred = ripple_Hz(z_Hz, Om, eps, omega, phi, gamma, H0)
    if np.any(np.isnan(Hz_pred)) or np.any(Hz_pred <= 0) or np.any(Hz_pred > 1e4):
        return -np.inf
    chi2_hz = np.sum(((Hz_obs - Hz_pred) / sigma_Hz) ** 2)
    return -0.5 * chi2_hz

# === Combined Log-Likelihood ===
def log_likelihood_joint(theta):
    ll_sn = log_likelihood_sn(theta)
    if not np.isfinite(ll_sn):
        return -np.inf
    ll_hz = log_likelihood_Hz(theta)
    if not np.isfinite(ll_hz):
        return -np.inf
    return ll_sn + ll_hz

# === LCDM H(z) Model ===
def Hz_LCDM(z_array, Om, H0):
    return H0 * np.sqrt(Om * (1 + z_array)**3 + (1 - Om))

# === LCDM Distance Modulus ===
def mu_LCDM(z_array, Om, H0):
    # Use ripple dL with eps=0, omega=0, phi=0, gamma=0
    dL = dL_ripple(z_array, Om, 0.0, 0.0, 0.0, 0.0, H0)
    return 5 * np.log10(dL) + 25 + M_locked

# === Prior Definition (Relaxed) ===
def log_prior(theta):
    Om, eps, omega, phi, gamma, H0 = theta
    if (
        0.10 < Om     < 0.50 and   # relaxed range
        -0.02 < eps    < 0.02 and   # allow larger ripple amplitude
        0.01 < omega   < 1.00 and   # allow higher frequencies
        -np.pi < phi   < np.pi and
        0.01 < gamma   < 0.50 and   # allow slower/longer damping
        60.0 < H0      < 77.0       # slight widening around H0
    ):
        return 0.0
    return -np.inf

# === Full Posterior Probability ===
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_joint(theta)
    return lp + ll if np.isfinite(ll) else -np.inf

# === MCMC Setup ===
ndim, nwalkers = 6, 64

# Initialize walkers uniformly within relaxed priors
pos = np.column_stack([
    np.random.uniform(0.10, 0.50,  nwalkers),  # Om
    np.random.uniform(-0.02, 0.02, nwalkers),  # eps
    np.random.uniform(0.05, 0.80,  nwalkers),  # omega
    np.random.uniform(-1, 1,       nwalkers),  # phi
    np.random.uniform(0.05, 0.50,  nwalkers),  # gamma
    np.random.uniform(64, 75,      nwalkers)   # H0
])

# === Run Joint MCMC ===
if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(f"Running Joint MCMC with locked M = {M_locked:.5f} (Relaxed Priors)...\n")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, 30000, progress=True)

    # Save raw chains
    np.save("joint_chain_relaxed.npy", sampler.get_chain())
    np.save("joint_log_prob_relaxed.npy", sampler.get_log_prob())

    # Extract posterior samples
    samples = sampler.get_chain(discard=5000, thin=10, flat=True)
    labels = ["Ωₘ", "ε", "ω", "φ", "γ", "H₀"]

    # === Corner Plot of Joint Posterior ===
    fig_corner = corner.corner(samples, labels=labels, truths=np.median(samples, axis=0))
    # fig_corner.suptitle(r"Joint $\mu(z)$ + $H(z)$ Posterior (Relaxed Priors)", y=1.02, fontsize=16)
    plt.show()

    # === Best-Fit Parameters (Median ± 1σ) ===
    best_fit = np.median(samples, axis=0)
    uncertainty = np.std(samples, axis=0)
    print("\n=== Best-Fit Parameters (Joint, Relaxed Priors: Median ± 1σ) ===")
    for name, val, err in zip(labels, best_fit, uncertainty):
        print(f"{name} = {val:.5f} ± {err:.5f}")

    Om_bf, eps_bf, omega_bf, phi_bf, gamma_bf, H0_bf = best_fit

    # === Plot Combined Model vs. Data ===
    # 1. Pantheon+ Residuals
    plt.figure(figsize=(10, 4))
    mu_joint = mu_model(z_sn, Om_bf, eps_bf, omega_bf, phi_bf, gamma_bf, H0_bf)
    residuals_sn = mu_obs - mu_joint
    plt.errorbar(
        z_sn,
        residuals_sn,
        yerr=sigma_mu,
        fmt='o',
        color='black',
        ecolor='gray',
        alpha=0.8,
        label="SN Residuals"
    )
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"Residual $[\mu_{\mathrm{obs}} - \mu_{\mathrm{model}}]$")
    plt.title(r"Pantheon+ Residuals (Joint Fit, Relaxed Priors)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. H(z) Comparison with Genesis Field and ΛCDM
    plt.figure(figsize=(10, 4))
    Hz_joint = ripple_Hz(z_Hz, Om_bf, eps_bf, omega_bf, phi_bf, gamma_bf, H0_bf)
    Hz_acdm  = Hz_LCDM(z_Hz, Om_bf, H0_bf)
    plt.errorbar(
        z_Hz,
        Hz_obs,
        yerr=sigma_Hz,
        fmt='o',
        color='blue',
        ecolor='gray',
        alpha=0.8,
        label=r"Observed $H(z)$"
    )
    plt.plot(z_Hz, Hz_joint, 'r-', label="Genesis Field Joint Fit")
    plt.plot(z_Hz, Hz_acdm, 'k--', label=r"$\Lambda$CDM (Same $\Omega_m$, $H_0$)", linewidth=1.5)
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"$H(z)$ [km/s/Mpc]")
    plt.title(r"$H(z)$ Comparison: Genesis Field vs. $\Lambda$CDM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    # === Fit Statistics (Joint, Relaxed Priors) ===
    chi2_sn = np.sum(((mu_obs - mu_joint) / sigma_mu) ** 2)
    chi2_hz = np.sum(((Hz_obs - Hz_joint) / sigma_Hz) ** 2)
    chi2_joint = chi2_sn + chi2_hz

    n_sn = len(z_sn)
    n_hz = len(z_Hz)
    n_total = n_sn + n_hz
    k_joint = ndim

    aic_joint = chi2_joint + 2 * k_joint
    bic_joint = chi2_joint + k_joint * np.log(n_total)

    print("\n=== Joint Fit Statistics (Relaxed Priors) ===")
    print(f"χ²_SN            = {chi2_sn:.2f}")
    print(f"χ²_Hz            = {chi2_hz:.2f}")
    print(f"χ²_total         = {chi2_joint:.2f}")
    print(f"AIC_joint        = {aic_joint:.2f}")
    print(f"BIC_joint        = {bic_joint:.2f}")
    print(f"χ²/dof_joint     = {chi2_joint / (n_total - k_joint):.3f}")

    # === Compute ACDM Metrics at Joint Best-Fit (2 parameters) ===
    mu_acdm = mu_LCDM(z_sn, Om_bf, H0_bf)
    chi2_sn_acdm = np.sum(((mu_obs - mu_acdm) / sigma_mu) ** 2)

    Hz_acdm_pred = Hz_acdm  # already computed above
    chi2_hz_acdm = np.sum(((Hz_obs - Hz_acdm_pred) / sigma_Hz) ** 2)

    chi2_acdm = chi2_sn_acdm + chi2_hz_acdm
    k_acdm = 2  # Om and H0 only

    aic_acdm = chi2_acdm + 2 * k_acdm
    bic_acdm = chi2_acdm + k_acdm * np.log(n_total)

    print("\n=== ΛCDM Fit Statistics at Joint Best-Fit ===")
    print(f"χ²_SN_ACDM       = {chi2_sn_acdm:.2f}")
    print(f"χ²_Hz_ACDM       = {chi2_hz_acdm:.2f}")
    print(f"χ²_total_ACDM    = {chi2_acdm:.2f}")
    print(f"AIC_acdm         = {aic_acdm:.2f}")
    print(f"BIC_acdm         = {bic_acdm:.2f}")
    print(f"χ²/dof_acdm      = {chi2_acdm / (n_total - k_acdm):.3f}")

    # === Save Joint Summary to JSON ===
    summary_joint = {
        "M_locked": M_locked,
        "best_fit": {k: float(v) for k, v in zip(labels, best_fit)},
        "uncertainty": {k: float(v) for k, v in zip(labels, uncertainty)},
        "chi2_sn": chi2_sn,
        "chi2_hz": chi2_hz,
        "chi2_joint": chi2_joint,
        "aic_joint": aic_joint,
        "bic_joint": bic_joint,
        "chi2_dof_joint": chi2_joint / (n_total - k_joint),
        "chi2_sn_acdm": chi2_sn_acdm,
        "chi2_hz_acdm": chi2_hz_acdm,
        "chi2_acdm": chi2_acdm,
        "aic_acdm": aic_acdm,
        "bic_acdm": bic_acdm,
        "chi2_dof_acdm": chi2_acdm / (n_total - k_acdm)
    }
    with open("joint_mcmc_acdm_comparison_summary.json", "w") as f:
        json.dump(summary_joint, f, indent=2)
    print("\n✅ Saved joint_mcmc_acdm_comparison_summary.json with diagnostics")
