# Interactive gravitational lensing model with categorized astronomical objects
import numpy as np
import matplotlib.pyplot as plt

# Lens model function
def lens_model(mass_solar, D_lens_Gpc, D_source_Gpc, 
               ellipticity_GR, ellipticity_QSSM, source_pos, object_name):
    G = 6.67430e-11
    c = 299792458
    solar_mass = 1.989e30

    M_lens = mass_solar * solar_mass
    D_lens = D_lens_Gpc * 1e9 * 3.086e16
    D_source = D_source_Gpc * 1e9 * 3.086e16
    D_ls = D_source - D_lens

    theta_E = np.sqrt((4 * G * M_lens / c**2) * (D_ls / (D_lens * D_source)))
    theta_E_arcsec = np.degrees(theta_E) * 3600

    phi = np.linspace(0, 2 * np.pi, 1000)
    a_GR = theta_E_arcsec
    b_GR = theta_E_arcsec * (1 - ellipticity_GR)
    a_QSSM = theta_E_arcsec
    b_QSSM = theta_E_arcsec * (1 - ellipticity_QSSM)

    x_GR = a_GR * np.cos(phi)
    y_GR = b_GR * np.sin(phi)
    x_QSSM = a_QSSM * np.cos(phi)
    y_QSSM = b_QSSM * np.sin(phi)

    plt.figure(figsize=(9, 9))
    plt.plot(x_GR, y_GR, '-', linewidth=2, color='blue', label=f'GR (axis ratio={1 - ellipticity_GR:.2f})')
    plt.plot(x_QSSM, y_QSSM, '--', linewidth=2, color='red', label=f'QSSM (axis ratio={1 - ellipticity_QSSM:.2f})')
    plt.plot(0, 0, 'ko', markersize=10, label='Lens Position')
    plt.plot(source_pos[0], source_pos[1], 'g*', markersize=15, label='Source Position')
    plt.title(f"{object_name} Elliptical Einstein Rings (GR vs QSSM)")
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    print("-" * 60)
    print("📡 Lens Model Summary:")
    print("-" * 60)
    print(f"Object: {object_name}")
    print(f"Mass: {mass_solar:.2e} solar masses ({M_lens:.2e} kg)")
    print("-" * 60)
    print(f"{'Observable':<30}{'GR prediction':<20}{'QSSM prediction':<20}")
    print("-" * 70)
    print(f"{'Major Axis (arcsec)':<30}{a_GR:<20.4f}{a_QSSM:<20.4f}")
    print(f"{'Minor Axis (arcsec)':<30}{b_GR:<20.4f}{b_QSSM:<20.4f}")
    print(f"{'Axis Ratio':<30}{(b_GR/a_GR):<20.4f}{(b_QSSM/a_QSSM):<20.4f}")
    print(f"{'Ellipticity difference':<30}{'0.0000':<20}{ellipticity_QSSM - ellipticity_GR:<20.4f}")
    print(f"{'Source position (arcsec)':<30}{'':<20}({source_pos[0]:.2f}, {source_pos[1]:.2f})")

# Categorized astronomical objects
categories = {
    "Black Holes": {
        "1": ("Sagittarius A*", 4.3e6, 0.008, 0.009, 0.05, 0.1, (0.05, 0.05)),
        "2": ("M87 (Virgo A)", 6.5e9, 0.016, 0.017, 0.1, 0.12, (0.1, 0.1))
    },
    "Galaxies": {
        "3": ("Galaxy SDP.81", 1e11, 1.2, 3.6, 0.2, 0.25, (0.3, 0.3)),
        "4": ("Einstein Cross (Q2237+030)", 1e11, 0.5, 1.5, 0.1, 0.15, (0.2, 0.2)),
        "5": ("Twin Quasar (QSO 0957+561)", 2e11, 1.3, 2.8, 0.15, 0.18, (0.25, 0.25))
    },
    "Galaxy Clusters": {
        "6": ("Abell 1689", 1e15, 0.6, 1.8, 0.3, 0.35, (0.4, 0.4)),
        "7": ("MACS J0416.1-2403", 8e14, 1.3, 3.5, 0.25, 0.3, (0.35, 0.35)),
        "8": ("RX J1347.5-1145", 1e15, 1.5, 4.0, 0.3, 0.35, (0.4, 0.4)),
        "9": ("CL0024+1654", 2e14, 1.0, 3.0, 0.2, 0.25, (0.3, 0.3)),
        "10": ("Bullet Cluster (1E 0657-558)", 2e15, 1.2, 3.2, 0.3, 0.32, (0.35, 0.35))
    }
}
# gello
# Selection menu
print("Select a gravitational lensing object by category:")
for cat, objs in categories.items():
    print(f"\n{cat}:")
    for key, (name, _, _, _, _, _, _) in objs.items():
        print(f"  {key}: {name}")

selection = input("Enter your choice: ").strip()
for objs in categories.values():
    if selection in objs:
        name, mass, D_lens, D_source, e_gr, e_qssm, source_pos = objs[selection]
        lens_model(mass, D_lens, D_source, e_gr, e_qssm, source_pos, name)
        break
else:
    print("Invalid selection.")