import numpy as np
import matplotlib.pyplot as plt

# Redshift range
z = np.linspace(0, 2.5, 500)

# Baseline ΛCDM expansion (approximate for this illustrative figure)
H_LCDM = 70 * (1 + 0.5 * z)

# Best-fit Genesis Field ripple parameters (from your final model)
epsilon = 0.05          # ripple amplitude
omega = 3.30            # ripple frequency
gamma = 0.15            # damping rate

# Genesis Field ripple-modulated expansion
H_Genesis = H_LCDM * (1 + epsilon * np.exp(-gamma * z) * np.sin(omega * z - 2.44))  # phi from your fit

# ±1.5% observational uncertainty band
error = 0.015 * H_LCDM

# Plotting
plt.figure(figsize=(12, 8))

# ΛCDM curve
plt.plot(z, H_LCDM, linestyle='--', color='black', linewidth=2.5, label="Standard $\\Lambda$CDM Expansion")

# Genesis Field prediction
plt.plot(z, H_Genesis, linestyle='-', color='blue', linewidth=2.5, label="Genesis Field Ripple Prediction")

# Observational precision band
plt.fill_between(z, H_LCDM - error, H_LCDM + error, color='gray', alpha=0.3, label="Future Observational Precision (±1.5%)")

# Ripple difference shading
plt.fill_between(z, H_LCDM, H_Genesis, where=(H_Genesis > H_LCDM), color='lightblue', alpha=0.4, interpolate=True)
plt.fill_between(z, H_LCDM, H_Genesis, where=(H_Genesis < H_LCDM), color='lightcoral', alpha=0.4, interpolate=True)

# Axis labels and title
plt.xlabel("Redshift $z$", fontsize=16)
plt.ylabel("Expansion Rate $H(z)$ [arb. units]", fontsize=16)
plt.title("Projected Comparison: Genesis Field Ripple vs. Standard $\\Lambda$CDM Expansion", fontsize=18)

# Aesthetic tweaks
plt.xlim(0, 2.5)
plt.ylim(min(H_LCDM - error) * 0.98, max(H_LCDM + error) * 1.02)
plt.legend(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()

# Show the plot
plt.show()
