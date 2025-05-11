import numpy as np
import matplotlib.pyplot as plt

# Define time array (cosmic time in Gyr)
t = np.linspace(0, 30, 1000)  # 0 to 30 Gyr, 1000 points

# Define ripple parameters
epsilon = 10        # Initial amplitude (arbitrary units, you can adjust)
gamma = 0.1         # Damping rate (per Gyr)
omega_m = 1.0       # Modulation angular frequency (rad/Gyr)
phi_0 = 0           # Phase offset (radians)

# Define the ripple function δ(t)
delta_t = epsilon * np.exp(-gamma * t) * np.cos(omega_m * t + phi_0)

# Plot
plt.figure(figsize=(10,5))
plt.plot(t, delta_t, label=r"Realistically Damped Ripple $\delta(t)$", color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Cosmic Time $t$ (Gyr)")
plt.ylabel("Ripple Amplitude")
plt.title("Realistic Ripple with Amplitude Damping (Energy Loss Over Time)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
