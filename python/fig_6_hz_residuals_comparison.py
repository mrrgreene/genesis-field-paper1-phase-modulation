import numpy as np
import matplotlib.pyplot as plt

# Redshift range
z = np.linspace(0, 2.5, 500)

# Approximate ΛCDM model: H(z) = H0 * (1 + 0.5z) for illustration
H_LCDM = 70 * (1 + 0.5 * z)  # Baseline expansion curve

# Genesis Field ripple parameters (best-fit from your model)
epsilon = 0.05          # ripple amplitude
omega = 3.30            # ripple frequency
gamma = 0.15            # damping rate
phi_shift = -2.44       # phase shift from your fit

# Genesis Field model prediction
H_Genesis = H_LCDM * (1 + epsilon * np.exp(-gamma * z) * np.sin(omega * z + phi_shift))

# Residuals: pretend H_LCDM is the mean of observed data
res_LCDM = H_LCDM - H_LCDM
res_Genesis = H_LCDM - H_Genesis

# ±1.5% observational uncertainty band
error = 0.015 * H_LCDM

# Plotting
plt.figure(figsize=(10, 6))
plt.axhline(0, linestyle='--', color='black', linewidth=1)

# Plot residuals
plt.plot(z, res_Genesis, label="Genesis Field Residuals", color='blue', linewidth=2)
plt.plot(z, res_LCDM, label="ΛCDM Residuals", color='gray', linewidth=2, linestyle=':')

# Uncertainty band
plt.fill_between(z, -error, error, color='gray', alpha=0.2, label="±1.5% Observational Band")

# Axis labels and title
plt.xlabel("Redshift $z$", fontsize=14)
plt.ylabel(r"$\Delta H(z) = H_{\mathrm{data}} - H_{\mathrm{model}}$", fontsize=14)
plt.title("Comparison of Residuals: Genesis Field vs. $\Lambda$CDM", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
