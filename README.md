# Genesis Field Paper I: Phase Modulation and the Hubble Tension

This repository contains the LaTeX sources, figures, handwritten derivations, and Python scripts for **Paper I** in the Genesis Field Theory (FQMT) series. **Paper I** introduces a ripple modulation model based on global coherence phase dynamics, providing a novel coherence-driven resolution to the persistent cosmological Hubble tension between local and distant measurements.

---

## 📚 Project Structure

```
GENESIS-FIELD-PAPER1-PHASE-MODULATION/
├── paper/
│   ├── genesis_field_paper.pdf         # Final compiled manuscript
│   ├── main.tex                        # Main LaTeX file
│   ├── sections/                       # Individual section files (.tex)
│   ├── appendix/                       # Supplemental appendix files
│   ├── figures/                        # All manuscript figures (PDF, EPS)
│   ├── handwritten_notes/              # Original handwritten derivations
│   │   └── derivation_notes.pdf
│   └── bib/                            # Bibliography files (.bib)
│
├── python/
│   ├── requirements.txt                # Python dependencies
│   ├── fig_2_ripple_modulation_plot.py
│   ├── fig_3_global_phase_modulation_plot.py
│   ├── fig_5_genesisfield_model_fit_hz.py
│   ├── fig_6_hz_residuals_comparison.py
│   ├── fig_7_genesisfield_qz.py
│   ├── fig_8_parameter_sensitivity_plot.py
│   ├── fig_9_genesis_field_ripple_vs_lcdm_projection.py
│   └── fig_10_gamma_sensitivity_plot.py
│
├── LICENSE                             # Project license (CC BY 4.0)
└── README.md                           # Project overview
```

---

## 🚀 Workflow

* **LaTeX Editing:** Manuscript editing and compilation via [Overleaf](https://overleaf.com), synchronized to this repository.
* **Figure Generation:** Python scripts (tested in VS Code and Google Colab) produce figures for publication. All scripts and resulting figures are version-controlled.
* **Version Control:** Complete project versioning provided by GitHub ensures reproducibility and transparency.

---

## 💻 Python Dependencies

Install required Python libraries using:

```bash
pip install -r python/requirements.txt
```

The scripts utilize the following key dependencies:

* `numpy==1.26.4`
* `scipy==1.13.0`
* `matplotlib==3.8.4`

---

## 🧑‍💻 Author

**Richard V. Greene**
Independent Researcher
[richvgreene@gmail.com](mailto:richvgreene@gmail.com)

---

## 📜 License

This work is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:

* **Share** — copy and redistribute the material in any medium or format.
* **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:

* **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

> [View License](https://creativecommons.org/licenses/by/4.0/)

---

🔗 **Associated Manuscript:** [Genesis Field: Phase Modulation and the Hubble Tension (PDF)](paper/genesis_field_paper.pdf)

🔗 **GitHub Repository:** [https://github.com/mrrgreene/genesis-field-paper1-phase-modulation](https://github.com/mrrgreene/genesis-field-paper1-phase-modulation)
