# Genesis Field H(z) Relaxed Fit — Ripple Emergence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import emcee
import corner

# === Reproducibility ===
np.random.seed(42)

# === Step 1: Include Observational Data ===
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

z = hz_clean["z"].values
Hz_obs = hz_clean["Hz"].values
sigma_Hz = hz_clean["sigma_Hz"].values

# === Step 2: Ripple model ===
Om_fixed = 0.36711  # Updated from full Pantheon fit

def ripple_Hz(z, eps, omega, phi, gamma, H0):
    r = eps * np.exp(-gamma * z) * np.cos(omega * z + phi)
    r0 = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    return H0 * norm * np.sqrt(Om_fixed * (1 + z)**3 + (1 - Om_fixed))

# === Step 3: Likelihood and Prior ===
def log_likelihood(theta):
    eps, omega, phi, gamma, H0 = theta
    model = ripple_Hz(z, eps, omega, phi, gamma, H0)
    return -0.5 * np.sum(((Hz_obs - model) / sigma_Hz)**2)

def log_prior(theta):
    eps, omega, phi, gamma, H0 = theta
    if (
        -0.1 < eps < 0.1 and
        0.01 < omega < 1.0 and
        -np.pi < phi < np.pi and
        0.005 < gamma < 1.0 and
        60.0 < H0 < 80.0
    ):
        return 0.0
    return -np.inf

def log_prob(theta):
    lp = log_prior(theta)
    return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

# === Step 4: Run MCMC ===
ndim, nwalkers, nsteps = 5, 50, 30000
initial = [0.01, 0.3, 0.0, 0.3, 68.0]
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
print("Running Genesis Field MCMC (Relaxed Fit, Ripple Emergence)...")
sampler.run_mcmc(pos, nsteps, progress=True)

# === Step 5: Analyze and Save ===
samples = sampler.get_chain(discard=5000, thin=10, flat=True)
labels = ["ε", "ω", "φ", "γ", "H₀"]

# Corner plot
fig = corner.corner(samples, labels=labels, truths=np.median(samples, axis=0))
plt.tight_layout()
plt.show()

# Save chains
np.save("hz_chain_mcmc_exploratory.npy", sampler.get_chain())
np.save("hz_log_prob_mcmc_exploratory.npy", sampler.get_log_prob())

# Print best-fit parameters
medians = np.median(samples, axis=0)
stds = np.std(samples, axis=0)

print("\nBest-fit Genesis Field Parameters (Relaxed Fit):\n")
for label, med, std in zip(labels, medians, stds):
    print(f"{label:>3} = {med: .5f} ± {std:.5f}")

# === ΛCDM Baseline Fit ===
def Hz_LCDM(z, H0):
    return H0 * np.sqrt(Om_fixed * (1 + z)**3 + (1 - Om_fixed))

from scipy.optimize import minimize

def chi2_LCDM(H0):
    model = Hz_LCDM(z, H0)
    return np.sum(((Hz_obs - model) / sigma_Hz)**2)

result = minimize(chi2_LCDM, x0=[70.0])
H0_lcdm = result.x[0]
Hz_fit_lcdm = Hz_LCDM(z, H0_lcdm)

resid_lcdm = Hz_obs - Hz_fit_lcdm
chi2_lcdm = np.sum((resid_lcdm / sigma_Hz) ** 2)
n_data = len(z)
k_lcdm = 1  # Only H0 is fit
aic_lcdm = chi2_lcdm + 2 * k_lcdm
bic_lcdm = chi2_lcdm + k_lcdm * np.log(n_data)
rms_lcdm = np.std(resid_lcdm)

print("\n=== ΛCDM Baseline Fit ===")
print(f"H₀                = {H0_lcdm:.5f}")
print(f"χ²                = {chi2_lcdm:.2f}")
print(f"AIC               = {aic_lcdm:.2f}")
print(f"BIC               = {bic_lcdm:.2f}")
print(f"χ²/dof            = {chi2_lcdm / (n_data - k_lcdm):.3f}")
print(f"Residual RMS      = {rms_lcdm:.5f} km/s/Mpc")
