# QSPR + SOAP + MBTR Descriptor Extractor (Streamlit App)

A powerful, interactive app for extracting **QSPR**, **SOAP**, and **MBTR** descriptors
from crystal structures (CIF files), featuring adaptive layer detection,
uncertainty propagation, and normalization options.

---

## Features

- Upload multiple `.cif` files
- Detect surface and layer structure using DBSCAN
- Compute:
  - QSPR bond and surface descriptors
  - SOAP & MBTR structural fingerprints (via DScribe)
- Predict binding energy (Alam et al. 2017)
- Normalize by atoms or bonds
- Download results as CSV
- Live progress + statistics visualization

---

## Installation

```bash
git clone https://github.com/<yourname>/qspr_soap_mbtr_app.git
cd qspr_soap_mbtr_app
pip install -r requirements.txt
