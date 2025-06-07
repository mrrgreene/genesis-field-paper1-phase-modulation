import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === Best-fit Genesis parameters (latest MCMC output) ===
# Tight-fit (from updated tight chain run)
H0_tight       = 69.06064
Om_tight       = 0.36711  # updated from Pantheon
eps_tight      = 0.00032
omega_tight    = 0.16940
phi_tight      = 0.08966
gamma_tight    = 0.16634

# Relaxed-fit (from updated relaxed chain run)
H0_relaxed     = 65.95492
Om_relaxed     = 0.36711  # fixed to Pantheon as well
eps_relaxed    = -0.03995
omega_relaxed  = 0.75323
phi_relaxed    = 0.05286
gamma_relaxed  = 0.32725

# === Model functions ===
def ripple_Hz(z, H0, Om, eps, omega, phi, gamma):
    r = eps * np.exp(-gamma * z) * np.cos(omega * z + phi)
    r0 = eps * np.cos(phi)
    norm = (1 + r) / (1 + r0)
    return H0 * norm * np.sqrt(Om * (1 + z)**3 + (1 - Om))

def Hz_LCDM(z):
    return 70.0 * np.sqrt(0.3 * (1 + z)**3 + 0.7)

# === Observational datasets ===
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

overlap_z = set(bao_df.z).union(cc_df.z)
farooq_filtered = farooq_df[~farooq_df.z.isin(overlap_z)]
hz_df = pd.concat([bao_df, cc_df, farooq_filtered], ignore_index=True)

# === Fit statistics ===
n = len(hz_df)
Hz_tight   = ripple_Hz(hz_df.z, H0_tight, Om_tight, eps_tight, omega_tight, phi_tight, gamma_tight)
Hz_relaxed = ripple_Hz(hz_df.z, H0_relaxed, Om_relaxed, eps_relaxed, omega_relaxed, phi_relaxed, gamma_relaxed)
Hz_lcdm    = Hz_LCDM(hz_df.z)

chi2_tight   = np.sum(((hz_df.Hz - Hz_tight) / hz_df.sigma_Hz)**2)
chi2_relaxed = np.sum(((hz_df.Hz - Hz_relaxed) / hz_df.sigma_Hz)**2)
chi2_lcdm    = np.sum(((hz_df.Hz - Hz_lcdm) / hz_df.sigma_Hz)**2)

k_tight = 6
k_relaxed = 6
k_lcdm = 2

aic_tight   = chi2_tight + 2 * k_tight
bic_tight   = chi2_tight + k_tight * np.log(n)
aic_relaxed = chi2_relaxed + 2 * k_relaxed
bic_relaxed = chi2_relaxed + k_relaxed * np.log(n)
aic_lcdm    = chi2_lcdm + 2 * k_lcdm
bic_lcdm    = chi2_lcdm + k_lcdm * np.log(n)

stats = pd.DataFrame({
    "χ²":    [chi2_tight, chi2_relaxed, chi2_lcdm],
    "χ²/N":  [chi2_tight/n, chi2_relaxed/n, chi2_lcdm/n],
    "AIC":   [aic_tight, aic_relaxed, aic_lcdm],
    "BIC":   [bic_tight, bic_relaxed, bic_lcdm]
}, index=["Genesis Tight", "Genesis Relaxed", "ΛCDM"])

print("\nH(z) Fit Statistics Comparison:\n")
print(stats.to_string(float_format="%.2f"))

# === Print Parameters ===
print("\nGenesis Tight Fit Parameters:")
print(f"  ε = {eps_tight:.5f}, ω = {omega_tight:.5f}, φ = {phi_tight:.5f}, γ = {gamma_tight:.5f}, H₀ = {H0_tight:.5f}")
print("\nGenesis Relaxed Fit Parameters:")
print(f"  ε = {eps_relaxed:.5f}, ω = {omega_relaxed:.5f}, φ = {phi_relaxed:.5f}, γ = {gamma_relaxed:.5f}, H₀ = {H0_relaxed:.5f}")

# === Plot ===
zgrid = np.linspace(0.01, 2.5, 300)
plt.figure(figsize=(10,6))
plt.plot(zgrid, ripple_Hz(zgrid, H0_tight, Om_tight, eps_tight, omega_tight, phi_tight, gamma_tight),
         label="Genesis Field (Tight)", color="blue", linewidth=2)
plt.plot(zgrid, ripple_Hz(zgrid, H0_relaxed, Om_relaxed, eps_relaxed, omega_relaxed, phi_relaxed, gamma_relaxed),
         label="Genesis Field (Relaxed)", color="black", linewidth=2)
plt.plot(zgrid, Hz_LCDM(zgrid), '--', label="ΛCDM Model", color="gray", linewidth=1.8)

plt.errorbar(bao_df.z, bao_df.Hz, yerr=bao_df.sigma_Hz, fmt='o', label="BAO", color="blue", alpha=0.6)
plt.errorbar(cc_df.z,  cc_df.Hz,  yerr=cc_df.sigma_Hz, fmt='s', label="Cosmic Chronometers", color="green", alpha=0.6)
plt.errorbar(farooq_filtered.z, farooq_filtered.Hz, yerr=farooq_filtered.sigma_Hz,
             fmt='^', label="Farooq & Ratra", color="red", alpha=0.6)

plt.xlabel("Redshift $z$")
plt.ylabel("$H(z)$ [km/s/Mpc]")
plt.title("Figure 4.3: Hubble Parameter Comparison — Genesis vs ΛCDM vs Observations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
