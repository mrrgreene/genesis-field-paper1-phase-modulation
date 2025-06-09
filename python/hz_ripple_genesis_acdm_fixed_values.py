import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# === Load H(z) Data ===
bao = pd.DataFrame({"z": [0.106, 0.15, 0.32, 0.57, 0.61, 0.73, 2.34],
                    "Hz": [69.0, 67.0, 79.2, 96.8, 97.3, 97.9, 222.0],
                    "sigma": [19.6, 12.0, 5.6, 3.4, 2.1, 2.7, 7.0]})
cc = pd.DataFrame({"z": [0.07, 0.12, 0.17, 0.179, 0.199, 0.27, 0.4, 0.48, 0.88, 1.3, 1.75, 2.0],
                   "Hz": [69, 68.6, 83, 75, 75, 77, 95, 97, 90, 168, 202, 222],
                   "sigma": [19.6, 26.2, 8, 4.9, 5, 14, 17, 60, 40, 17, 40, 41]})
faro = pd.DataFrame({"z": [0.07, 0.09, 0.12, 0.17, 0.179, 0.199, 0.2, 0.27, 0.28, 0.352, 0.4,
                           0.44, 0.48, 0.57, 0.593, 0.6, 0.68, 0.73, 0.781, 0.875, 0.88, 0.9,
                           1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965, 2.34, 2.36],
                     "Hz": [69, 69, 68.6, 83, 75, 75, 72.9, 77, 88.8, 83, 95,
                            82.6, 97, 96.8, 104, 87.9, 92, 97.3, 105, 125, 90, 117,
                            154, 168, 160, 177, 140, 202, 186.5, 222, 226],
                     "sigma": [19.6, 12, 26.2, 8, 4.9, 5, 29.6, 14, 36.6, 14, 17,
                               7.8, 60, 3.4, 13, 6.1, 8, 7, 12, 17, 40, 23,
                               20, 17, 33.6, 18, 14, 40, 50.4, 7, 8]})

df = pd.concat([bao, cc, faro]).drop_duplicates(subset="z").sort_values("z").reset_index(drop=True)

z = df["z"].values
Hz = df["Hz"].values
sigma = df["sigma"].values

# === Fit ΛCDM ===
Om = 0.36711
def H_lcdm(z, H0): return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

from scipy.optimize import curve_fit
H0_fit, _ = curve_fit(H_lcdm, z, Hz, sigma=sigma, p0=[68])
Hz_lcdm = H_lcdm(z, H0_fit[0])
residuals = Hz - Hz_lcdm

# === Focus on residuals in 0.1 < z < 0.6
mask = (z > 0.1) & (z < 0.6)
z_band = z[mask]
res_band = residuals[mask]
sigma_band = sigma[mask]

# === Fit ripple to residuals: ΔH(z) ≈ A cos(ω z + φ)
def ripple_residual(z, amp, omega, phi):
    return amp * np.cos(omega * z + phi)

omega_grid = np.linspace(0.5, 6.0, 500)
best_chi2 = np.inf
best_fit = None

for omega in omega_grid:
    try:
        popt, _ = curve_fit(lambda z, amp, phi: ripple_residual(z, amp, omega, phi),
                            z_band, res_band, sigma=sigma_band, p0=[5, 0])
        model = ripple_residual(z_band, *popt, omega)
        chi2 = np.sum(((res_band - model) / sigma_band) ** 2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_fit = (popt[0], omega, popt[1])
    except:
        continue

# === Output fixed values ===
print("=== Genesis Field Fixed Ripple Parameters ===")
print(f"ω* (frequency) = {best_fit[1]:.4f}")
print(f"φ* (phase)     = {best_fit[2]:.4f}")
