import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# ————————————————————————————————
# Load Pantheon+SH0ES SN data
# ————————————————————————————————
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, "data", "Pantheon+SH0ES.dat")

df = (pd.read_csv(data_path, sep=r"\s+")
        [["zCMB","MU_SH0ES","MU_SH0ES_ERR_DIAG"]]
        .rename(columns={
            "zCMB":"z", 
            "MU_SH0ES":"mu_obs", 
            "MU_SH0ES_ERR_DIAG":"sigma"
        }))
df = df[df.sigma < 0.15]

z       = df.z.values
mu_obs  = df.mu_obs.values
sigma   = df.sigma.values
N       = len(z)

# ————————————————————————————————
# Genesis Field model parameters
# (fixed from your MCMC medians)
# ————————————————————————————————
c       = 299792.458   # km/s
H0_g    = 68.91        # Genesis H0
Om_g    = 0.3326
eps     = -0.0068
omega   = 0.4896
phi     = 0.2813
gamma   = 0.2580

# ————————————————————————————————
# ΛCDM model parameters
# (choose your fiducial e.g. Planck / Pantheon)
# ————————————————————————————————
H0_l    = 70.0
Om_l    = 0.30

# ————————————————————————————————
# Zero‐point baseline
# ————————————————————————————————
M0      = 24.2   # Pantheon baseline (so μ=5log10 dL + 25 + M0)
# we will solve ∆M analytically below

# ————————————————————————————————
# Model definitions (no zero‐point yet)
# ————————————————————————————————
def H_g(z):
    r = eps * np.exp(-gamma*z) * np.cos(omega*z + phi)
    return H0_g * (1 + r) * np.sqrt(Om_g*(1+z)**3 + (1-Om_g))

def H_lcdm(z):
    return H0_l * np.sqrt(Om_l*(1+z)**3 + (1-Om_l))

def mu_noM(Hfunc, H0):
    """Compute μ(z) = 5 log10[dL(z)] + 25, without extra ∆M."""
    def invE(zp):
        return H0 / Hfunc(zp)
    I = np.array([quad(invE, 0, zi, epsabs=1e-6)[0] for zi in z])
    dL = (c / H0) * (1 + z) * I
    return 5 * np.log10(dL) + 25.0

# compute the “bare” distance moduli
mu_g0 = mu_noM(H_g, H0_g)
mu_l0 = mu_noM(H_lcdm, H0_l)

# ————————————————————————————————
# Analytic zero‐point fit:
# minimize χ² w.r.t ∆M ⇒ weighted mean of residuals
# ∆M_best = Σ[(mu_obs - mu0)/σ²] / Σ[1/σ²]
# ————————————————————————————————
w = 1.0 / sigma**2

dM_g = np.sum((mu_obs - mu_g0)*w) / np.sum(w)
dM_l = np.sum((mu_obs - mu_l0)*w) / np.sum(w)

mu_g = mu_g0 + dM_g
mu_l = mu_l0 + dM_l

# ————————————————————————————————
# Compute fit statistics
# k = 1 free parameter (∆M) in each
# χ² = Σ[(resid/σ)²]
# AIC = χ² + 2k
# BIC = χ² + k ln N
# ————————————————————————————————
def fit_stats(mu_model, label):
    resid = mu_obs - mu_model
    chi2  = np.sum((resid/sigma)**2)
    k     = 1
    aic   = chi2 + 2*k
    bic   = chi2 + k*np.log(N)
    return chi2, chi2/N, aic, bic

chi2_g, chi2N_g, aic_g, bic_g = fit_stats(mu_g, "Genesis")
chi2_l, chi2N_l, aic_l, bic_l = fit_stats(mu_l, "LCDM")

# ————————————————————————————————
# Print table to terminal
# ————————————————————————————————
print("\nSN Fit Statistics Comparison (Pantheon+SH0ES)")
print("————————————————————————————————————————")
print(f"{'Model':<12} χ²      χ²/N    AIC      BIC")
print(f"{'-'*52}")
print(f"{'Genesis':<12} {chi2_g:7.2f} {chi2N_g:7.2f} {aic_g:7.2f} {bic_g:7.2f}")
print(f"{'ΛCDM':<12} {chi2_l:7.2f} {chi2N_l:7.2f} {aic_l:7.2f} {bic_l:7.2f}\n")

# ————————————————————————————————
# Plot everything
# ————————————————————————————————
plt.figure(figsize=(8,6))
plt.errorbar(z, mu_obs, yerr=sigma, fmt='o', ms=4, color='k',
             label="Pantheon+SH0ES")
plt.plot(z, mu_g, '-', lw=2, color='C0', label=f"Genesis  ΔM={dM_g:+.3f}")
plt.plot(z, mu_l, '--', lw=2, color='C1', label=f"ΛCDM     ΔM={dM_l:+.3f}")
plt.xlabel("Redshift $z$")
plt.ylabel(r"Distance modulus $\mu(z)$ [mag]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
