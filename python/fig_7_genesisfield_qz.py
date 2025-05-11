import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Observational Data (Farooq Cosmic Chronometers) ---
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

# --- Final Genesis Ripple Model (γ fixed = 0.15) ---
def ripple_model(z, H0, Omega_m, epsilon, omega, phi):
    gamma_fixed = 0.15
    Hz_LCDM = H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
    return Hz_LCDM * (1 + epsilon * np.exp(-gamma_fixed * z) * np.sin(omega * z + phi))

# === Use Your Final Fit ===
# Replace this with your actual final best-fit result
# For example (these match your last fit):
popt = [72.954349, 0.247681, 0.050000, 3.298487, -2.443573]

# --- Smooth Redshift Grid ---
z_smooth = np.linspace(min(z), max(z), 500)
Hz_smooth = ripple_model(z_smooth, *popt)

# --- Compute q(z) from H(z) ---
dHz_dz = np.gradient(Hz_smooth, z_smooth)
q_z = - (1 + z_smooth) * dHz_dz / Hz_smooth - 1

# --- Plot Deceleration Parameter q(z) ---
plt.figure(figsize=(12, 7))
plt.plot(z_smooth, q_z, color='blue', linewidth=2.5, label='Genesis Field Ripple $q(z)$ (γ = 0.15)')
plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, label='Transition (q = 0)')
plt.xlabel("Redshift $z$", fontsize=14)
plt.ylabel("Deceleration Parameter $q(z)$", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
