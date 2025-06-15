# Genesis Field Paper I: Phase Modulation and the Hubble Tension

This repository contains the LaTeX sources, figures, handwritten derivations, and Python analysis code for **Paper I** in the Genesis Field Theory (FQMT) series. This first paper introduces a ripple modulation model arising from **global vacuum phase coherence**, offering a falsifiable, coherence-based explanation for the persistent **Hubble tension** between local and early-universe measurements.

The Genesis Field framework treats spacetime as a quantum fluid governed by global phase dynamics, yielding distinctive ripple features in the cosmic expansion rate $H(z)$. These predictions are empirically tested against Pantheon+ supernovae and cosmic chronometer datasets using full MCMC pipelines.

---

## 📘 Access Policy

This repository contains the full statistical pipeline behind the Genesis Field model, including MCMC fits and ripple-modulated expansion predictions.

🔒 The code is temporarily private while the manuscript is under review with journals and pending arXiv endorsement.

### 🔓 Want Access?

I'm happy to provide private access to researchers, reviewers, or collaborators.

To endorse my submission to the `astro-ph.CO` section of arXiv, use:

🔗 **https://arxiv.org/auth/endorse?x=OEVQQI**

If that link doesn’t work, visit:

🔗 **http://arxiv.org/auth/endorse.php**  
and enter:

🔐 **Endorsement Code: OEVQQI**

Thanks for supporting open science.  
📧 richvgreene@gmail.com

---

## 📁 Project Structure

```
GENESIS-FIELD-PAPER1-PHASE-MODULATION/
├── paper/
│   ├── genesis_field_paper.pdf         # Final compiled manuscript
│   ├── main.tex                        # Main LaTeX file
│   ├── sections/                       # Individual section .tex files
│   ├── appendix/                       # Derivations, glossary, and code
│   ├── figures/                        # All manuscript figures (PDF)
│   ├── handwritten_notes/              # Scanned derivation notes (PDF)
│   └── bib/                            # BibTeX references
│
├── python/
│   ├── data/                           # Pantheon+SH0ES data and covariance
│   │   ├── Pantheon+SH0ES.dat
│   │   ├── Pantheon+SH0ES_STAT+SYS.cov
│   │   └── Pantheon+SH0ES_122221_VPEC.cov
│   ├── mcmc_pantheon.py                      # Section IV.A: SN-only MCMC (ripple suppressed)
│   ├── hz_mcmc_ripple_tight.py               # Section IV.B: H(z) tight-fit (Pantheon-calibrated)
│   ├── hz_mcmc_ripple_relax.py               # Section IV.C: H(z) relaxed-fit (ripple emergence)
│   ├── joint_mcmc.py                         # Section IV.D: Joint SN + H(z) MCMC
│   ├── joint_ripple_summary.py               # Figure 11: Bar plot summary of ripple parameters
│   ├── mcmc_m_diagnostic.py                  # Appendix C: M₀ analytic zero-point fit
│   ├── mcmc_m_diagnostic_comparison.py       # Appendix C: Residual vs. analytic M₀
│   ├── sn_genesis_acdm_model.py              # Residuals comparison: Genesis vs. ΛCDM (SN)
│   ├── hz_genesis_acdm_model.py              # Ripple overlays and residuals (H(z))
│   └── requirements.txt                      # Python package dependencies
│
├── output/                             # JSON summaries of fits, residuals, and diagnostics
│   └── *.json
│
├── LICENSE                             # CC BY 4.0 License
├── PRD_cover_letter.txt                # Draft cover letter for PRD submission
└── README.md                           # Project overview (this file)
```

---

## 🚀 Reproducibility Workflow

- **LaTeX Editing**: All manuscript source files are built using RevTeX 4.2 and are Overleaf-compatible.
- **MCMC Pipeline**: All cosmological fits are implemented using [`emcee`](https://emcee.readthedocs.io/), with both tight and relaxed priors, and full model diagnostics (χ², AIC, BIC, RMS).
- **Plot Generation**: All plots are produced using the included Python scripts and can be regenerated to match the manuscript figures.
- **Zero-Point Calibration**: The SN magnitude zero-point $M$ is analytically derived and fixed (`mcmc_m_diagnostic.py`).
- **Cross-Validation**: Model performance is tested across SN-only, H(z)-only, and joint datasets for full empirical discipline.

---

## 💻 Python Setup

Install required libraries via:

```bash
pip install -r python/requirements.txt
```

Core dependencies include:

- `numpy==1.26.4`
- `scipy==1.13.0`
- `matplotlib==3.8.4`
- `emcee==3.1.4`
- `corner==2.2.1`
- `pandas==2.2.2`

---

## 📊 Mapping: Paper Sections ↔ Scripts

| Paper Section        | Purpose                                                              | Script(s) |
|----------------------|----------------------------------------------------------------------|-----------|
| **IV.A**             | SN-only MCMC with ripple suppression (ΛCDM recovery)                | `mcmc_pantheon.py` |
| **IV.B**             | H(z) tight-fit with Omega_m from SN (ripple suppressed)             | `hz_mcmc_ripple_tight.py` |
| **IV.C**             | H(z) relaxed-fit with ripple activation                             | `hz_mcmc_ripple_relax.py` |
| **IV.D**             | Joint SN + H(z) MCMC with relaxed priors                            | `joint_mcmc.py` |
| **Figure 11**        | Ripple parameter bar chart across IV.A–IV.D                         | `joint_ripple_summary.py` |
| **V.C**              | Fixed ripple fit (only epsilon, H0 free) — decisive test vs ΛCDM    | `hz_ripple_genesis_acdm_fixed_values.py` |
| *Core model support* | mu(z), d_L(z), and ripple-modulated H(z) used across MCMC scripts   | `mcmc_m_diagnostic.py`, `mcmc_m_diagnostic_comparison.py` |
| **Model Overlays**   | Genesis vs ΛCDM curve definitions for SN mu(z) and H(z)             | `sn_genesis_acdm_model.py`, `hz_genesis_acdm_model.py` |

---

## 📂 Data Access

- **SN and $H(z)$ data**: Included in `python/data/` (Pantheon+SH0ES and chronometer/BAO compilations).
- **Fit results**: All MCMC chains and statistical summaries are exported to `output/*.json`.
- **Residual plots**: Reproducible via the scripts and consistent with figures in the manuscript.

---

## 🧑‍💻 Author

**Richard V. Greene**  
Independent Researcher  
📧 richvgreene@gmail.com  
🆔 ORCID: [0009-0002-2430-8184](https://orcid.org/0009-0002-2430-8184)

---

## 📖 How to Cite

If you use this work or its datasets/code, please cite:

> Greene, R.V. *Genesis Field Framework: Phase Modulation and the Hubble Tension*. Preprint, 2025.  
> GitHub Repository: [https://github.com/mrrgreene/genesis-field-paper1-phase-modulation](https://github.com/mrrgreene/genesis-field-paper1-phase-modulation)

A Zenodo DOI will be provided upon final archive.

---

## 📜 License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to:

- **Share** — copy and redistribute the material in any medium or format  
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially  

**Under the following terms:**

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.  
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

> 📄 [View Full License](https://creativecommons.org/licenses/by/4.0/)

---

🔗 **Associated Manuscript:** [Genesis Field: Phase Modulation and the Hubble Tension (PDF)](paper/genesis_field_paper.pdf)  
🔗 **GitHub Repository:** [https://github.com/mrrgreene/genesis-field-paper1-phase-modulation](https://github.com/mrrgreene/genesis-field-paper1-phase-modulation)
