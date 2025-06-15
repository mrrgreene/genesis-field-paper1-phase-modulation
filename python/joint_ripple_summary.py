import numpy as np
import matplotlib.pyplot as plt
import json
import os

# === Load MCMC summaries from JSON files in output/ folder ===
output_dir = "output"
pantheon_path = os.path.join(output_dir, "sn_mcmc_pantheon_summary.json")
joint_path = os.path.join(output_dir, "joint_mcmc_acdm_comparison_summary.json")

with open(pantheon_path) as f:
    pantheon_summary = json.load(f)

with open(joint_path) as f:
    joint_summary = json.load(f)

# === Hardcoded H(z) Relaxed (no JSON available) ===
hz_relaxed = {
    "ε":  (-0.03995, 0.08142),
    "ω":  (0.75323, 0.23061),
    "γ":  (0.32725, 0.27892)
}

# === Extract parameters of interest ===
labels = ["ε", "ω", "γ"]
pantheon = pantheon_summary["best_fit"]
pantheon_err = pantheon_summary["uncertainty"]

joint = joint_summary["best_fit"]
joint_err = joint_summary["uncertainty"]

# === Prepare bar chart data ===
means = {
    "Pantheon+": [pantheon[l] for l in labels],
    "H(z) Relaxed": [hz_relaxed[l][0] for l in labels],
    "Joint Relaxed": [joint[l] for l in labels]
}
errors = {
    "Pantheon+": [pantheon_err[l] for l in labels],
    "H(z) Relaxed": [hz_relaxed[l][1] for l in labels],
    "Joint Relaxed": [joint_err[l] for l in labels]
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
