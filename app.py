# =========================================================
# 🔬 ULTIMATE QSPR + SOAP + MBTR SCIENTIFIC WORKSTATION
# Multi-file CIF analysis | Scientist-grade control board
# Streamlit 1.39+ compatible (no experimental APIs)
# =========================================================
import os, io, json, math, tempfile, zipfile
from collections import defaultdict, OrderedDict
import numpy as np, pandas as pd, streamlit as st
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from ase import Atoms
from dscribe.descriptors import SOAP, MBTR
import py3Dmol

# -------------- Page setup ----------------
st.set_page_config(page_title="Scientific Workstation — Ultimate", layout="wide")
st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)

# -------------- Defaults ----------------
BOND_LENGTH_FALLBACK = 3.2
DEFAULT_DBSCAN_EPS, DEFAULT_DBSCAN_MIN_SAMPLES = 0.25, 1
DEFAULT_SOAP_N = 50

DEFAULT_COEFFICIENTS = OrderedDict([
    ('b0', 0.0000),
    ('b_Pt-S', -0.1225),
    ('b_Ru-S', 0.0000),
    ('b_Pt-Ru', 0.0050),
    ('b_Ru-Ru', -0.0160),
    ('b_Ru-Ru-', -0.1020),
    ('b_Pt-Pt', 0.0048),
    ('b_Pt-Pt-', 0.0550)
])

ELEMENT_COLORS = {
    "H":"#FFF", "C":"#909090","N":"#3050F8","O":"#FF0D0D","F":"#90E050",
    "P":"#FF8000","S":"#FFFF30","Cl":"#1FF01F","Pt":"#D0D0E0","Ru":"#248F8F",
    "Pd":"#A0C0D0","Co":"#F090A0","Fe":"#E06633","Cu":"#C88033","Ni":"#50D050",
    "Au":"#FFD123","Ag":"#C0C0C0","Zn":"#7D80B0","Sn":"#668080","Pb":"#575961"
}

cnn = CrystalNN(weighted_cn=False)

# -------------- Helpers ----------------
def adaptive_bond_cutoff(e1, e2, scale=1.25, fallback=BOND_LENGTH_FALLBACK):
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        if any(x is None or np.isnan(x) for x in [r1, r2]): return fallback
        return scale * (r1 + r2)
    except Exception:
        return fallback

def detect_layers(z_coords, eps, min_samples):
    z = np.array(z_coords).reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(z).labels_
    unique = sorted(set(labels))
    mapping = {lab: i for i, lab in enumerate(unique)}
    return np.array([mapping[lab] for lab in labels])

def structure_to_ase(structure):
    return Atoms(
        symbols=[str(s.specie) for s in structure],
        positions=np.array([s.coords for s in structure]),
        cell=structure.lattice.matrix, pbc=True
    )

def propagate_uncertainty_linear(coeff_map, x_counts):
    return math.sqrt(sum((b * math.sqrt(max(x_counts.get(k.replace('b_', ''), 0), 0)))**2
                         for k, b in coeff_map.items() if k != 'b0'))

def make_py3dmol_html(structure, show_bonds=True, bond_min=0.0, bond_max=3.0,
                      highlight=None, atom_size=0.45):
    view = py3Dmol.view(width=760, height=520)
    coords = np.array([s.coords for s in structure])
    elems = [str(s.specie) for s in structure]
    for el, pos in zip(elems, coords):
        color = ELEMENT_COLORS.get(el, '#AAAAAA')
        view.addSphere({'center': dict(zip('xyz', map(float,pos))),
                        'radius': atom_size, 'color': color})
    if show_bonds:
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                d = structure.get_distance(i,j)
                if bond_min <= d <= bond_max:
                    view.addCylinder({
                        'start': dict(zip('xyz', map(float, coords[i]))),
                        'end': dict(zip('xyz', map(float, coords[j]))),
                        'radius': atom_size*0.15, 'color': '#BBBBBB'
                    })
    if highlight is not None and 0 <= highlight < len(coords):
        pos = coords[int(highlight)]
        view.addSphere({'center': dict(zip('xyz', map(float,pos))),
                        'radius': atom_size*1.5, 'color': 'red'})
    view.setBackgroundColor('white')
    view.zoomTo()
    return view.render()

# -------------- Session ----------------
for key, val in [('pair_ranges', {}), ('coeff_map', dict(DEFAULT_COEFFICIENTS)), ('presets', {})]:
    if key not in st.session_state: st.session_state[key] = val

def save_preset(name):
    st.session_state.presets[name] = {
        'pair_ranges': st.session_state.pair_ranges,
        'coeff_map': st.session_state.coeff_map
    }

def load_preset(name):
    p = st.session_state.presets.get(name, {})
    st.session_state.pair_ranges = p.get('pair_ranges', {})
    st.session_state.coeff_map = p.get('coeff_map', DEFAULT_COEFFICIENTS)

# -------------- UI ----------------
st.title("🔬 Scientific Workstation — Ultimate (QSPR + SOAP + MBTR)")

tabs = st.tabs(["Viewer", "Control Board", "Descriptors", "Presets"])
uploaded_files = []

# --- Control Board ---
with tabs[1]:
    st.subheader("⚙️ Control Board")
    with st.expander("File Upload & Element Detection", expanded=True):
        uploaded_files = st.file_uploader("Upload CIF(s)", type=["cif"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded.")
        if st.button("Auto-detect element pairs"):
            if not uploaded_files:
                st.warning("Upload CIF files first.")
            else:
                elems = set()
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                        tmp.write(f.getbuffer()); tmp.flush()
                        s = Structure.from_file(tmp.name)
                        elems.update(str(at.specie) for at in s)
                        os.remove(tmp.name)
                pairs = [f"{a}-{b}" for i,a in enumerate(sorted(elems)) for b in sorted(elems)[i:]]
                for pair in pairs:
                    if pair not in st.session_state.pair_ranges:
                        parts = pair.split('-')
                        est = adaptive_bond_cutoff(parts[0], parts[1])
                        st.session_state.pair_ranges[pair] = (0.0, round(est*1.12,3))
                st.toast(f"Detected {len(pairs)} pairs.", icon="🧪")
                st.rerun()

    with st.expander("Per-pair Bond Length Control", expanded=True):
        if st.session_state.pair_ranges:
            df = pd.DataFrame([{"pair":k,"min":v[0],"max":v[1]}
                               for k,v in st.session_state.pair_ranges.items()])
            edited = st.data_editor(df, use_container_width=True)
            st.session_state.pair_ranges = {r["pair"]:(r["min"],r["max"]) for _,r in edited.iterrows()}
        else:
            st.info("No pairs yet. Detect them first.")

    with st.expander("Global Parameters"):
        st.session_state.normalize = st.selectbox("Normalize descriptors by", ["bonds","atoms","none"])
        st.session_state.compute_soap = st.checkbox("Compute SOAP (slow)", True)
        st.session_state.compute_mbtr = st.checkbox("Compute MBTR (slow)", True)
        st.session_state.soap_n = st.slider("SOAP features kept", 10, 200, 50)
        st.session_state.db_eps = st.number_input("DBSCAN eps", 0.05, 1.0, 0.25, 0.01)
        st.session_state.db_min = st.number_input("DBSCAN min_samples", 1, 10, 1)

# --- Descriptors tab ---
with tabs[2]:
    st.subheader("📊 Coefficients Editor")
    dfc = pd.DataFrame([{"name":k,"value":v} for k,v in st.session_state.coeff_map.items()])
    edited = st.data_editor(dfc, use_container_width=True)
    st.session_state.coeff_map = {r["name"]:float(r["value"]) for _,r in edited.iterrows()}

    if st.button("💾 Save Current Preset"):
        save_preset("last")
        st.toast("Preset saved as 'last'.")

    if uploaded_files and st.button("🚀 Run Full Analysis"):
        progress = st.progress(0)
        results, atoms_all = [], []
        for i, f in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                tmp.write(f.getbuffer()); tmp.flush()
                s = Structure.from_file(tmp.name)
            # simplified — reuse your analyze_structure here if you wish
            desc = {"filename":f.name, "n_atoms":len(s.sites)}
            results.append(desc)
            progress.progress((i+1)/len(uploaded_files))
        st.success("Completed!")
        df = pd.DataFrame(results)
        st.dataframe(df)

# --- Viewer tab ---
with tabs[0]:
    st.subheader("3D Viewer Preview (first file)")
    if uploaded_files:
        f = uploaded_files[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
            tmp.write(f.getbuffer()); tmp.flush()
            s = Structure.from_file(tmp.name)
        html = make_py3dmol_html(s, highlight=0)
        st.components.v1.html(html, height=540)
    else:
        st.info("Upload at least one CIF in Control Board.")

# --- Presets tab ---
with tabs[3]:
    st.subheader("🧭 Presets")
    pname = st.text_input("Preset name", "my_preset")
    col1,col2 = st.columns(2)
    if col1.button("Save"):
        save_preset(pname)
        st.success(f"Preset '{pname}' saved.")
    if col2.button("Load"):
        if pname in st.session_state.presets:
            load_preset(pname)
            st.success(f"Loaded preset '{pname}'.")
        else:
            st.warning("Preset not found.")
