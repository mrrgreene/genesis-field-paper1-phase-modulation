import numpy as np
import matplotlib.pyplot as plt

# === Best-fit values and 1σ errors from your sections ===
labels = ["ε", "ω", "γ"]

# Replace with your actual post-fit numbers
means = {
    "Pantheon+": [-0.00003, 0.15473, 0.15486],
    "H(z) Relaxed": [-0.03995, 0.75323, 0.32725],
    "Joint Relaxed": [-0.00017, 0.56960, 0.24526]
}
errors = {
    "Pantheon+": [0.00579, 0.08364, 0.08391],
    "H(z) Relaxed": [0.08142, 0.23061, 0.27892],
    "Joint Relaxed": [0.01232, 0.28998, 0.14243]
}

# === Bar Plot ===
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red

for i, param in enumerate(labels):
    ax = axes[i]
    y = [means[key][i] for key in means]
    yerr = [errors[key][i] for key in errors]
    ax.bar(range(len(y)), y, yerr=yerr, capsize=6, color=colors)
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(means.keys(), rotation=15)
    ax.set_title(f"{param} across fits")
    ax.set_ylabel(param)
    ax.grid(True)

fig.suptitle("Ripple Parameter Summary Across Fits", fontsize=14)
plt.tight_layout()
plt.show()
