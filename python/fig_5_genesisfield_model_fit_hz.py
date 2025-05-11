import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Observational Data ===
z = np.array([0.070, 0.090, 0.120, 0.170, 0.179, 0.199, 0.200, 0.270, 0.280, 0.352,
              0.380, 0.3802, 0.400, 0.4004, 0.4247, 0.440, 0.4497, 0.4783, 0.480,
              0.510, 0.593, 0.600, 0.610, 0.680, 0.730, 0.781, 0.875, 0.880, 0.900,
              1.037, 1.300, 1.363, 1.430, 1.530, 1.750, 1.965, 2.340, 2.360])

Hz = np.array([69, 69, 68.6, 83, 75, 75, 72.9, 77, 88.8, 83,
               81.5, 83, 95, 77, 87.1, 82.6, 92.8, 80.9, 97,
               90.4, 104, 87.9, 97.3, 92, 97.3, 105, 125, 90,
               117, 154, 168, 160, 177, 140, 202, 186.5, 222, 226])

Hz_err = np.array([19.6, 12, 26.2, 8, 4, 5, 29.6, 14, 36.6, 14,
                   1.9, 13.5, 17, 10.2, 11.2, 7.8, 12.9, 9, 62,
                   1.9, 13, 6.1, 2.1, 8, 7, 12, 17, 40,
                   23, 20, 17, 33.6, 18, 14, 40, 50.4, 7, 8])

# === Ripple-on-LCDM Model with Fixed Damping ===
def ripple_model(z, H0, Omega_m, epsilon, omega, phi):
    gamma_fixed = 0.15
    Hz_LCDM = H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
    return Hz_LCDM * (1 + epsilon * np.exp(-gamma_fixed * z) * np.sin(omega * z + phi))

# === Standard ΛCDM for comparison ===
def lcdm_model(z, H0, Omega_m):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

# === Initial Guesses and Bounds ===
initial_guess = [70, 0.3, 0.01, 5.0, 0.0]  # H0, Omega_m, epsilon, omega, phi
bounds = ([60, 0.2, 0.001, 0.5, -np.pi], [80, 0.4, 0.05, 10.0, np.pi])

# === Fit Genesis Ripple Model ===
popt, pcov = curve_fit(ripple_model, z, Hz, sigma=Hz_err, absolute_sigma=True,
                       p0=initial_guess, bounds=bounds, maxfev=200000)

# === Smooth Redshift Array for Plotting ===
z_smooth = np.linspace(min(z), max(z), 500)

# === Best-fit Ripple Prediction ===
Hz_fit = ripple_model(z_smooth, *popt)

# === ΛCDM prediction using best-fit H0 and Omega_m ===
Hz_lcdm = lcdm_model(z_smooth, popt[0], popt[1])

# === Metrics ===
Hz_pred = ripple_model(z, *popt)
chi_sq = np.sum(((Hz - Hz_pred) / Hz_err)**2)
reduced_chi_sq = chi_sq / (len(Hz) - len(popt))
r_squared = 1 - np.sum((Hz - Hz_pred)**2) / np.sum((Hz - np.mean(Hz))**2)

# === Print Results ===
param_names = ["H0", "Omega_m", "epsilon", "omega", "phi"]
param_uncert = np.sqrt(np.diag(pcov))
print("\nFinal Ripple Model Fit Parameters (γ fixed = 0.15):")
print(f"{'Parameter':>10} {'Value':>15} {'± Uncertainty':>20}")
for name, val, err in zip(param_names, popt, param_uncert):
    print(f"{name:>10} {val:15.6f} {err:20.6f}")

print(f"\nReduced Chi-squared: {reduced_chi_sq:.3f}")
print(f"R-squared: {r_squared:.4f}")

# === Final Plot ===
plt.figure(figsize=(12, 7))
plt.errorbar(z, Hz, yerr=Hz_err, fmt='o', color='black', capsize=3, label='Observed $H(z)$')
plt.plot(z_smooth, Hz_fit, label='Genesis Field Ripple Fit (γ=0.15)', color='blue', linewidth=2.5)
plt.plot(z_smooth, Hz_lcdm, '--', color='gray', linewidth=2.5, label='Standard $\Lambda$CDM')

# === Labels and styling ===
plt.xlabel("Redshift $z$", fontsize=14)
plt.ylabel("Hubble Parameter $H(z)$ [km/s/Mpc]", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# === Show Plot ===
plt.show()
