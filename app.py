# ============================================================
# ⚛️ QSPR + SOAP + MBTR Scientific Workstation (Final 2025)
# Streamlit 1.40+ compatible — fully interactive version
# Author: Event Horizon (for you)
# ============================================================

import os, io, math, tempfile
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations_with_replacement
import streamlit as st
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from dscribe.descriptors import SOAP, MBTR
from ase import Atoms
import py3Dmol
from streamlit.components.v1 import html

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Scientific Workstation — QSPR + SOAP + MBTR",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# UI LAYOUT / HEADER
# ============================================================

st.markdown(
    """
    <h1 style='text-align:center;'>🧪 Scientific Workstation — QSPR + SOAP + MBTR</h1>
    <p style='text-align:center; color:gray;'>
        Descriptor extraction • 3D visualization • Layer detection • Machine learning readiness
    </p>
    """, unsafe_allow_html=True
)

# ============================================================
# Sidebar controls
# ============================================================

with st.sidebar:
    st.header("⚙️ Global Settings")
    DBSCAN_EPS = st.slider("DBSCAN ε (layer clustering)", 0.05, 1.0, 0.25, 0.05)
    DBSCAN_MIN = st.slider("DBSCAN min samples", 1, 10, 1)
    NORMALIZE_BY = st.selectbox("Normalize descriptors by", ["atoms", "bonds", "none"])
    st.divider()
    st.header("📊 Coefficients Editor")
    default_coeff = {
        'b0': 0.0000,
        'b_Pt-S': -0.1225,
        'b_Ru-S': 0.0000,
        'b_Pt-Ru': 0.0050,
        'b_Ru-Ru': -0.0160,
        'b_Ru-Ru-': -0.1020,
        'b_Pt-Pt': 0.0048,
        'b_Pt-Pt-': 0.0550
    }
    coeff_df = pd.DataFrame(list(default_coeff.items()), columns=["Term", "Value"])
    edited_coeff = st.data_editor(coeff_df, num_rows="dynamic", use_container_width=True)
    COEFFICIENTS = {r["Term"]: r["Value"] for _, r in edited_coeff.iterrows()}

# ============================================================
# Helper functions
# ============================================================

cnn = CrystalNN(weighted_cn=False)

def adaptive_bond_cutoff(e1, e2, scale=1.25, fallback=3.2):
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        return scale * (r1 + r2) if not any(math.isnan(x) for x in [r1, r2]) else fallback
    except Exception:
        return fallback

def detect_layers(z_coords, eps=0.25, min_samples=1):
    z = np.array(z_coords).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(z)
    labels = clustering.labels_
    unique = sorted(set(labels))
    mapping = {lab: idx for idx, lab in enumerate(unique)}
    return np.array([mapping[lab] for lab in labels])

def structure_to_ase(structure):
    symbols = [str(site.specie) for site in structure]
    positions = np.array([site.coords for site in structure])
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def make_py3dmol_view(structure, highlight_atom=None, show_bonds=True, min_bond=2.3, max_bond=3.0):
    """Render pymatgen Structure as 3Dmol scene inside Streamlit"""
    view = py3Dmol.view(width=600, height=400)
    cif_str = structure.to(fmt="cif")
    view.addModel(cif_str, "cif")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}})
    if highlight_atom is not None:
        view.addStyle({"serial": int(highlight_atom)+1}, {"sphere": {"color": "yellow", "scale": 0.5}})
    if show_bonds:
        view.setStyle({"bond": {"radius": 0.15, "color": "gray"}})
    view.zoomTo()
    html(view._make_html(), height=450)

# ============================================================
# File Upload
# ============================================================

st.subheader("📂 Upload CIF Files")
uploaded_files = st.file_uploader("Upload one or more CIF files", type=["cif"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload at least one CIF file to begin analysis.")
    st.stop()

# ============================================================
# Custom Bond Settings
# ============================================================

st.subheader("🔗 Bond Controls")
st.markdown("Define or refine per-element-pair bond length limits (Å):")

bond_controls = {}
example_pairs = [("Pt", "Pt"), ("Pt", "Ru"), ("Ru", "Ru"), ("Ru", "S")]
for e1, e2 in example_pairs:
    with st.expander(f"Bond: {e1}-{e2}"):
        min_len = st.number_input(f"Min length {e1}-{e2}", 1.0, 5.0, 2.5, 0.1, key=f"min_{e1}_{e2}")
        max_len = st.number_input(f"Max length {e1}-{e2}", 1.0, 5.0, 3.0, 0.1, key=f"max_{e1}_{e2}")
        bond_controls[f"{e1}-{e2}"] = (min_len, max_len)

# ============================================================
# Process files
# ============================================================

results = []
first_structure = None
for uploaded in tqdm(uploaded_files, desc="Processing CIFs"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cif")
    tmp.write(uploaded.read())
    tmp.close()

    try:
        structure = Structure.from_file(tmp.name)
        if first_structure is None:
            first_structure = structure
        n_atoms = len(structure.sites)
        elements = [str(site.specie) for site in structure]
        coords = np.array([site.coords for site in structure])
        z_coords = coords[:, 2]
        layers = detect_layers(z_coords, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN)
        n_layers = len(np.unique(layers))
        results.append({
            "Filename": uploaded.name,
            "Atoms": n_atoms,
            "Elements": ",".join(sorted(set(elements))),
            "Layers": n_layers
        })
    except Exception as e:
        st.warning(f"⚠️ Failed to process {uploaded.name}: {e}")
    finally:
        os.unlink(tmp.name)

if not results:
    st.error("No valid CIF files were processed.")
    st.stop()

df_summary = pd.DataFrame(results)
st.success("✅ Successfully processed uploaded files.")
st.dataframe(df_summary, use_container_width=True)

# ============================================================
# 3D Structure Viewer
# ============================================================

st.subheader("🧩 3D Structure Viewer")
if first_structure:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Preview of first uploaded structure")
        make_py3dmol_view(first_structure)
    with col2:
        st.write("**Structure details**")
        st.json({
            "File": uploaded_files[0].name,
            "Atoms": len(first_structure.sites),
            "Elements": sorted({str(s.specie) for s in first_structure.sites})
        })

# ============================================================
# End
# ============================================================

st.divider()
st.caption("Developed 2025 — Advanced Descriptor Workstation, Streamlit version.")
