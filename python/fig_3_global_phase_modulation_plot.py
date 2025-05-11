import numpy as np
import matplotlib.pyplot as plt

# Define time array
t = np.linspace(0, 30, 1000)  # 0 to 30 Gyr

# Define model parameters
omega_c = 10      # Baseline trend slope
epsilon = 50      # Exaggerated ripple amplitude
gamma = 0.1       # Damping rate
omega_m = 1.0     # Ripple frequency
phi_0 = 0         # Phase offset

# Define ripple δ(t)
delta_t = epsilon * np.exp(-gamma * t) * np.cos(omega_m * t + phi_0)

# Define full coherence phase φ(t)
phi_t = omega_c * t + delta_t

# Plot
plt.figure(figsize=(10,5))
plt.plot(t, phi_t, label=r"Global Coherence Phase $\phi(t) = \omega_c t + \delta(t)$", color='darkred')
plt.plot(t, omega_c * t, '--', label=r"Baseline trend ($\omega_c t$)", color='gray')
plt.xlabel("Cosmic Time $t$ (Gyr)")
plt.ylabel("Global Phase $\phi(t)$ (arbitrary units)")
plt.title("Global Coherence Phase Modulation (Amplitude Exaggerated for Clarity)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
