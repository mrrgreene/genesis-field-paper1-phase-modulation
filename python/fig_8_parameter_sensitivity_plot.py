import numpy as np
import matplotlib.pyplot as plt

# Define parameters and their ranges, tighter focus
epsilon_prime = np.linspace(0.012, 0.022, 100)  # Zoomed-in ripple amplitude range
gamma_prime = np.linspace(0.12, 0.28, 100)      # Zoomed-in damping rate range

# Creating meshgrid for sensitivity analysis
epsilon_grid, gamma_grid = np.meshgrid(epsilon_prime, gamma_prime)

# Hypothetical observational constraint function (focused region)
allowed_region = np.exp(-(epsilon_grid - 0.0175)**2/(2*0.0035**2)) * np.exp(-(gamma_grid - 0.2)**2/(2*0.04**2))

# Plotting the sensitivity contours
plt.figure(figsize=(10, 7))
contours = plt.contourf(epsilon_grid, gamma_grid, allowed_region, cmap='Blues', levels=30)
plt.colorbar(contours, label='Allowed Parameter Space Probability')

# Labeling the plot clearly
plt.xlabel("Ripple Amplitude Parameter $\\epsilon'$", fontsize=14)
plt.ylabel("Damping Rate Parameter $\\gamma'$", fontsize=14)
plt.title("Zoomed Parameter Sensitivity of Genesis Field Ripple Structure", fontsize=16)

# Improved annotations: repositioning for clean appearance
plt.annotate('Current observational constraints', 
             xy=(0.0185, 0.23), xytext=(0.016, 0.26),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=12, weight='bold')

plt.annotate('Future missions\n(Euclid, JWST, Rubin)', 
             xy=(0.016, 0.19), xytext=(0.013, 0.14),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=12, weight='bold')

# Adjust ticks for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Tight layout for better visibility
plt.tight_layout()

# Save figure if needed
# plt.savefig('parameter_sensitivity_plot.pdf')

# Display plot
plt.show()
