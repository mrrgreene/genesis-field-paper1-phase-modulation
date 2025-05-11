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

# === Ripple-on-LCDM Model ===
def ripple_model(z, H0, Omega_m, epsilon, omega, phi, gamma):
    Hz_LCDM = H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
    return Hz_LCDM * (1 + epsilon * np.exp(-gamma * z) * np.sin(omega * z + phi))

# === Initial Guesses and Bounds ===
initial_guess = [70, 0.3, 0.01, 5.0, 0.0]
bounds = ([60, 0.2, 0.001, 0.5, -np.pi], [80, 0.4, 0.05, 10.0, np.pi])

# === Gamma Sensitivity Analysis ===
gamma_values = np.linspace(0.1, 0.2, 50)
chi_sq_values = []

for gamma_fixed in gamma_values:
    def model_fixed_gamma(z, H0, Omega_m, epsilon, omega, phi):
        return ripple_model(z, H0, Omega_m, epsilon, omega, phi, gamma_fixed)

    popt, _ = curve_fit(model_fixed_gamma, z, Hz, sigma=Hz_err, absolute_sigma=True,
                        p0=initial_guess, bounds=bounds, maxfev=200000)

    Hz_pred = model_fixed_gamma(z, *popt)
    chi_sq = np.sum(((Hz - Hz_pred) / Hz_err)**2)
    chi_sq_values.append(chi_sq / (len(Hz) - len(popt)))

# === Sensitivity Plot ===
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, chi_sq_values, '-o', color='navy', markersize=5)
plt.axvline(x=0.15, color='red', linestyle='--', linewidth=2,
            label='Chosen γ = 0.15')
plt.xlabel("Damping Parameter γ", fontsize=14)
plt.ylabel("Reduced Chi-squared ($\chi^2_\nu$)", fontsize=14)
plt.title("Sensitivity Analysis for Damping Parameter γ", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# === Display Plot ===
plt.show()
