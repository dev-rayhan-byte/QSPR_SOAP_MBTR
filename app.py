"""
Ultimate Streamlit Scientific Workstation
- Multi-file CIF handling
- Global element/pair control + per-pair bond min/max
- Full control board, coefficient editor, presets
- QSPR + SOAP + MBTR integration
- Polished 3D viewer (py3Dmol) with color map
Author: Adapted for user (2025) - fixed/cleaned
"""

import os
import io
import json
import math
import tempfile
import zipfile
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from ase import Atoms
from dscribe.descriptors import SOAP, MBTR
import py3Dmol

# ---------------- page config ----------------
st.set_page_config(page_title="Scientific Workstation — Ultimate", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>footer {visibility: hidden;} </style>", unsafe_allow_html=True)

# ---------------- defaults ----------------
BOND_LENGTH_FALLBACK = 3.2
DEFAULT_DBSCAN_EPS = 0.25
DEFAULT_DBSCAN_MIN_SAMPLES = 1
DEFAULT_SOAP_N = 50

# Default coefficient map (editable by user)
DEFAULT_COEFFICIENTS = OrderedDict([
    ('b0', 0.0000),
    ('b_Pt-S', -0.1225),
    ('b_Ru-S', 0.0000),
    ('b_Pt-Ru', 0.0050),
    ('b_Ru-Ru', -0.0160),
    ('b_Ru-Ru-', -0.1020),
    ('b_Pt-Pt', 0.0048),
    ('b_Pt-Pt-', 0.0550),
])

# quick element color map (extend as needed)
ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050",
    "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Pt": "#D0D0E0", "Ru": "#248F8F",
    "Pd": "#A0C0D0", "Co": "#F090A0", "Fe": "#E06633", "Cu": "#C88033", "Ni": "#50D050",
    "Au": "#FFD123", "Ag": "#C0C0C0", "Zn": "#7D80B0", "Sn": "#668080", "Pb": "#575961",
}

# initialize CrystalNN
cnn = CrystalNN(weighted_cn=False)

# ---------------- helper funcs ----------------
def adaptive_bond_cutoff(e1, e2, scale=1.25, fallback=BOND_LENGTH_FALLBACK):
    """Adaptive cutoff using covalent radii, fallback if missing."""
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in [r1, r2]):
            return fallback
        return scale * (r1 + r2)
    except Exception:
        return fallback

def detect_layers(z_coords, eps=DEFAULT_DBSCAN_EPS, min_samples=DEFAULT_DBSCAN_MIN_SAMPLES):
    z = np.array(z_coords).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(z)
    labels = clustering.labels_
    unique = sorted(set(labels))
    mapping = {lab: idx for idx, lab in enumerate(unique)}
    return np.array([mapping[lab] for lab in labels])

def find_neighbors_validated(structure, idx, max_check=128):
    neighbors_idx = set()
    try:
        for info in cnn.get_nn_info(structure, idx):
            j = info["site_index"]
            e1, e2 = str(structure[idx].specie), str(structure[j].specie)
            d = structure.get_distance(idx, j)
            if d <= adaptive_bond_cutoff(e1, e2) * 1.05:
                neighbors_idx.add(j)
    except Exception:
        pass
    if not neighbors_idx:
        for j, site in enumerate(structure.sites):
            if j == idx:
                continue
            e1, e2 = str(structure[idx].specie), str(site.specie)
            d = structure.get_distance(idx, j)
            if d <= adaptive_bond_cutoff(e1, e2):
                neighbors_idx.add(j)
            if len(neighbors_idx) >= max_check:
                break
    return sorted(neighbors_idx)

def structure_to_ase(structure):
    symbols = [str(s.specie) for s in structure]
    positions = np.array([s.coords for s in structure])
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def propagate_uncertainty_linear(coeff_map, x_counts):
    sigma2 = 0.0
    for ck, b in coeff_map.items():
        if ck == 'b0': continue
        # ck like 'b_Pt-S' -> xkey 'Pt-S'
        xkey = ck.replace('b_', '')
        if xkey in x_counts:
            N = max(x_counts[xkey], 0)
            sigma_x = math.sqrt(N)
            sigma2 += (b * sigma_x)**2
    return math.sqrt(sigma2)

# robust viewer builder using py3Dmol
def make_py3dmol_view(structure: Structure, show_bonds=True, bond_min=0.0, bond_max=3.0,
                      highlight_atom_index=None, show_poly=False, poly_neighbors=None,
                      atom_size=0.45, style='sphere'):
    coords = np.array([s.coords for s in structure.sites])
    elements = [str(s.specie) for s in structure.sites]
    view = py3Dmol.view(width=780, height=520)
    # add atoms as spheres
    for i, (el, pos) in enumerate(zip(elements, coords)):
        color = ELEMENT_COLORS.get(el, ELEMENT_COLORS.get(el.capitalize(), '#AAAAAA'))
        view.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                        'radius': atom_size, 'color': color, 'opacity': 1.0})
    # bonds (cylinders)
    if show_bonds:
        n = len(elements)
        for i in range(n):
            for j in range(i+1, n):
                d = structure.get_distance(i, j)
                if bond_min <= d <= bond_max:
                    view.addCylinder({'start': {'x': float(coords[i,0]), 'y': float(coords[i,1]), 'z': float(coords[i,2])},
                                      'end':   {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                                      'radius': atom_size*0.15, 'color': '#BBBBBB', 'opacity': 1.0})
    # highlight atom
    if highlight_atom_index is not None and 0 <= int(highlight_atom_index) < len(coords):
        pos = coords[int(highlight_atom_index)]
        view.addSphere({'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                        'radius': atom_size*1.4, 'color': 'red', 'opacity': 1.0})
        if show_poly and poly_neighbors:
            for j in poly_neighbors:
                view.addCylinder({'start': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                                  'end': {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                                  'radius': atom_size*0.08, 'color': 'orange', 'opacity': 1.0})
    view.setBackgroundColor('#ffffff')
    view.zoomTo()
    # return HTML snippet suitable for Streamlit embedding
    try:
        html = view._make_html()
    except Exception:
        # fallback to show() result if api differs
        html = view.show()
    return html

# ---------------- analysis core ----------------
def analyze_structure(structure, filename,
                      per_pair_ranges=None,
                      normalize_by="bonds", dbscan_eps=DEFAULT_DBSCAN_EPS, dbscan_min_samples=DEFAULT_DBSCAN_MIN_SAMPLES,
                      compute_soap=True, compute_mbtr=True, soap_first_n=DEFAULT_SOAP_N,
                      coeff_map=None):
    n_atoms = len(structure.sites)
    elements = [str(site.specie) for site in structure.sites]
    coords = np.array([site.coords for site in structure])
    z_coords = coords[:, 2] if n_atoms > 0 else np.array([])

    # layer detection
    if n_atoms > 0:
        layer_indices = detect_layers(z_coords, eps=dbscan_eps, min_samples=dbscan_min_samples)
        layer_means = {l: np.mean(z_coords[layer_indices == l]) for l in np.unique(layer_indices)}
        label_map = {old: new for new, old in enumerate(sorted(layer_means, key=lambda L: layer_means[L]))}
        layer_indices = np.array([label_map[l] for l in layer_indices])
        n_layers = int(layer_indices.max()) + 1
        surface_mask = (layer_indices == layer_indices.min()) | (layer_indices == layer_indices.max())
    else:
        layer_indices = np.array([], dtype=int)
        n_layers = 0
        surface_mask = np.array([], dtype=bool)

    # per-atom table
    atom_rows = []
    for i, c in enumerate(coords):
        atom_rows.append({
            "filename": filename,
            "atom_index": i,
            "element": elements[i],
            "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
            "layer_index": int(layer_indices[i]) if n_atoms>0 else -1,
            "surface_flag": bool(surface_mask[i]) if n_atoms>0 else False
        })
    atom_df = pd.DataFrame(atom_rows)

    # bond counting using per_pair_ranges if provided
    bond_counts_intra = defaultdict(int)
    bond_counts_inter = defaultdict(int)
    coordination = np.zeros(n_atoms)
    for i in range(n_atoms):
        neighs = find_neighbors_validated(structure, i)
        coordination[i] = len(neighs)
        for j in neighs:
            if j <= i: continue
            e1, e2 = elements[i], elements[j]
            pair_key = "-".join(sorted([e1, e2]))
            d = structure.get_distance(i, j)
            # determine cutoff from per_pair_ranges or adaptive
            if per_pair_ranges and pair_key in per_pair_ranges:
                min_r, max_r = per_pair_ranges[pair_key]
            else:
                min_r = 0.0
                max_r = adaptive_bond_cutoff(e1, e2) * 1.12
            if not (min_r <= d <= max_r):
                continue
            if n_atoms > 0 and layer_indices[i] == layer_indices[j]:
                bond_counts_intra[pair_key] += 1
            else:
                bond_counts_inter[pair_key] += 1

    surface_counts = defaultdict(int)
    for i, surf in enumerate(surface_mask):
        if surf:
            surface_counts[f"{elements[i]}-S"] += 1

    # standard descriptor keys
    x = {
        'Pt-S': surface_counts.get('Pt-S', 0),
        'Ru-S': surface_counts.get('Ru-S', 0),
        'Pt-Ru': bond_counts_intra.get('Pt-Ru', 0) + bond_counts_inter.get('Pt-Ru', 0),
        'Pt-Pt': bond_counts_intra.get('Pt-Pt', 0),
        'Ru-Ru': bond_counts_intra.get('Ru-Ru', 0),
        'Pt-Pt-': bond_counts_inter.get('Pt-Pt', 0),
        'Ru-Ru-': bond_counts_inter.get('Ru-Ru', 0)
    }

    total_bonds = sum(bond_counts_intra.values()) + sum(bond_counts_inter.values())
    desc = {
        "filename": filename,
        "n_atoms": n_atoms,
        "n_layers": n_layers,
        "avg_coordination": float(np.mean(coordination)) if n_atoms>0 else 0.0,
        "surface_ratio": float(np.sum(surface_mask) / n_atoms) if n_atoms>0 else 0.0,
        "total_bonds": int(total_bonds)
    }
    desc.update(x)

    # normalization
    x_norm = {k: v for k, v in x.items()}
    if normalize_by == "atoms" and n_atoms > 0:
        x_norm = {k: v / n_atoms for k, v in x.items()}
    elif normalize_by == "bonds" and total_bonds > 0:
        x_norm = {k: v / total_bonds for k, v in x.items()}

    # binding energy prediction
    if coeff_map is None:
        coeff_map = DEFAULT_COEFFICIENTS
    Y = coeff_map.get('b0', 0.0) + sum(coeff_map.get('b_' + k, 0.0) * x_norm.get(k, 0.0) for k in x_norm)
    sigmaY = propagate_uncertainty_linear(coeff_map, x)
    desc['Binding_Energy_eV'] = float(Y)
    desc['Binding_Energy_sigma_eV_approx'] = float(sigmaY)

    # SOAP + MBTR
    try:
        ase_atoms = structure_to_ase(structure)
        species = list(sorted(set(ase_atoms.get_chemical_symbols())))
    except Exception:
        ase_atoms = None
        species = []

    if compute_soap and ase_atoms is not None and species:
        try:
            soap = SOAP(species=species, rcut=5.0, nmax=6, lmax=4, sigma=0.5, periodic=True)
            soap_vec = soap.create(ase_atoms)
            soap_mean = np.mean(soap_vec, axis=0)
            for i, val in enumerate(soap_mean[:soap_first_n]):
                desc[f"SOAP_{i}"] = float(val)
        except Exception as e:
            st.warning(f"SOAP failed for {filename}: {e}")

    if compute_mbtr and ase_atoms is not None and species:
        try:
            mbtr = MBTR(
                species=species,
                k2={"geometry": {"function": "inverse_distance"}, "grid": {"min": 0, "max": 1, "n": 30}},
                k3={"geometry": {"function": "angle"}, "grid": {"min": 0, "max": 180, "n": 30}},
                periodic=True,
                normalization="l2_each"
            )
            mbtr_vec = mbtr.create(ase_atoms)
            desc["MBTR_mean"] = float(np.mean(mbtr_vec))
            desc["MBTR_std"] = float(np.std(mbtr_vec))
        except Exception as e:
            st.warning(f"MBTR failed for {filename}: {e}")

    return desc, atom_df

# ---------------- UI: persistent session state helpers ----------------
if 'pair_ranges' not in st.session_state:
    st.session_state['pair_ranges'] = {}  # dict pair->(min,max)
if 'coeff_map' not in st.session_state:
    st.session_state['coeff_map'] = dict(DEFAULT_COEFFICIENTS)
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

def save_preset(name):
    st.session_state['presets'][name] = {
        'pair_ranges': st.session_state['pair_ranges'],
        'coeff_map': st.session_state['coeff_map']
    }

def load_preset(name):
    p = st.session_state['presets'][name]
    st.session_state['pair_ranges'] = p.get('pair_ranges', {})
    st.session_state['coeff_map'] = dict(p.get('coeff_map', DEFAULT_COEFFICIENTS))

# ---------------- UI layout ----------------
st.title("🔬 Scientific Workstation — Ultimate (QSPR + SOAP + MBTR)")

# top-level tabs
tab_view, tab_control, tab_descriptors, tab_presets = st.tabs(["Viewer", "Control Board", "Descriptors & Run", "Presets"])

# ---------- Viewer tab ----------
with tab_view:
    st.header("3D Structure Viewer (first file preview)")
    left, right = st.columns([2,1])
    with left:
        uploaded = st.session_state.get('uploaded_files', [])
        if not uploaded:
            st.info("Upload in the Control Board tab.")
        viewer_placeholder = st.empty()
    with right:
        st.markdown("**Viewer Controls**")
        v_show_bonds = st.checkbox("Show bonds (viewer)", value=True, key="v_show_bonds")
        v_bond_min = st.number_input("Viewer min bond (Å)", value=0.0, step=0.01, key="v_bond_min")
        v_bond_max = st.number_input("Viewer max bond (Å)", value=3.0, step=0.01, key="v_bond_max")
        v_show_poly = st.checkbox("Show coordination polyhedra (viewer)", value=False, key="v_show_poly")
        v_atom_index = st.number_input("Highlight atom index (0-based, viewer)", value=0, min_value=0, step=1, key="v_atom_index")
        st.markdown("---")
        st.write("Quick exports:")
        st.button("Screenshot view (not implemented)", disabled=True)

# ---------- Control Board ----------
with tab_control:
    st.header("Control Board — upload files & global settings")
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload CIF(s) — multiple OK", type=["cif"], accept_multiple_files=True, key="file_uploader_widget")
        # store a persistent reference in session_state for other tabs
        if uploaded:
            st.session_state['uploaded_files'] = uploaded
        st.markdown("**File list:**")
        uploaded_list = st.session_state.get('uploaded_files', [])
        if uploaded_list:
            for idx, f in enumerate(uploaded_list):
                size_kb = round(len(f.getbuffer())/1024,1)
                st.write(f"{idx+1}. {f.name} — {size_kb} KB")
    with col2:
        st.subheader("Global Analysis Settings")
        normalize_by = st.selectbox("Normalization for descriptors", options=["bonds","atoms","none"], index=0)
        compute_soap = st.checkbox("Compute SOAP (slow)", value=True)
        compute_mbtr = st.checkbox("Compute MBTR (slow)", value=True)
        soap_n = st.number_input("Keep first N SOAP features", min_value=10, max_value=1000, value=50)
        dbscan_eps = st.number_input("DBSCAN eps (layers)", value=0.25, step=0.01)
        dbscan_min = st.number_input("DBSCAN min_samples", value=1, step=1, min_value=1)
        st.markdown("**Per-pair bond controls**")
        st.markdown("You can set per pair min/max bond lengths. If blank, uses adaptive covalent estimates.")
        if st.button("Auto-detect element pairs from uploaded files"):
            # build pair set
            all_elements = set()
            uploaded_list = st.session_state.get('uploaded_files', [])
            if not uploaded_list:
                st.warning("Upload files first")
            else:
                for f in uploaded_list:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                        tmp.write(f.getbuffer()); tmp.flush(); p = tmp.name
                    try:
                        s = Structure.from_file(p)
                        all_elements.update([str(site.specie) for site in s.sites])
                    finally:
                        try: os.remove(p)
                        except: pass
                all_elements = sorted(all_elements)
                pairs = []
                for i, e1 in enumerate(all_elements):
                    for e2 in all_elements[i:]:
                        pairs.append("-".join(sorted([e1, e2])))
              # initialize pair ranges in session if missing
for pair in pairs:
    if pair not in st.session_state['pair_ranges']:
        parts = pair.split('-')
        est = adaptive_bond_cutoff(parts[0], parts[1])
        st.session_state['pair_ranges'][pair] = (0.0, round(est * 1.12, 3))

# ✅ Success message without forced rerun
st.success(f"Detected element pairs: {len(pairs)}")

# ✅ Remove experimental rerun (dangerous), use safe conditional rerun
if "just_added_pairs" not in st.session_state:
    st.session_state["just_added_pairs"] = True
    st.rerun()             # ✅ safe one-time rerun
else:
    st.session_state["just_added_pairs"] = False


st.markdown("Per-pair editor (editable):")

# show editable DataFrame for pairs
pr = st.session_state['pair_ranges']

if pr:
    df_pairs = pd.DataFrame([
        {"pair": k, "min": float(v[0]), "max": float(v[1])}
        for k, v in sorted(pr.items())
    ])

    # ✅ Use st.data_editor instead of deprecated version
    edited = st.data_editor(df_pairs, num_rows="dynamic")

    # write back to session_state
    new_map = {}
    for r in edited.to_dict(orient='records'):
        try:
            mn = float(r["min"])
            mx = float(r["max"])
        except Exception:
            # fallback on previous values
            old = pr.get(r["pair"], (0.0, BOND_LENGTH_FALLBACK))
            mn, mx = old

        new_map[r["pair"]] = (mn, mx)

    st.session_state['pair_ranges'] = new_map

else:
    st.info("No pairs detected yet. Click 'Auto-detect element pairs' after uploading CIFs.")

# ---------- Descriptor / run tab ----------
with tab_descriptors:
    st.header("Descriptors & Run")
    st.subheader("COEFFICIENTS editor (manually edit + save/load)")
    # coefficients editor
    coeff_df = pd.DataFrame([{"name": k, "value": float(v)} for k, v in st.session_state['coeff_map'].items()])
    edited_coeff = st.data_editor(coeff_df, num_rows="dynamic")
    # write back
    new_coeff = {}
    for row in edited_coeff.to_dict(orient='records'):
        try:
            new_coeff[str(row['name'])] = float(row['value'])
        except Exception:
            new_coeff[str(row['name'])] = 0.0
    st.session_state['coeff_map'] = new_coeff

    colA, colB = st.columns([1,1])
    with colA:
        run_button = st.button("Run full analysis (all uploaded files)")
    with colB:
        download_preset_btn = st.button("Save current preset as 'last_preset'")

    if download_preset_btn:
        save_preset("last_preset")
        st.success("Saved preset as 'last_preset'.")

    if run_button:
        uploaded_list = st.session_state.get('uploaded_files', [])
        if not uploaded_list:
            st.warning("Upload CIFs first in Control Board.")
        else:
            # processing loop
            n = len(uploaded_list)
            progress = st.progress(0)
            results = []
            atom_tables = []
            for i, f in enumerate(uploaded_list):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                        tmp.write(f.getbuffer()); tmp.flush(); p = tmp.name
                    s = Structure.from_file(p)
                    desc, atom_df = analyze_structure(
                        s, f.name,
                        per_pair_ranges=st.session_state['pair_ranges'],
                        normalize_by=normalize_by,
                        dbscan_eps=dbscan_eps,
                        dbscan_min_samples=dbscan_min,
                        compute_soap=compute_soap,
                        compute_mbtr=compute_mbtr,
                        soap_first_n=int(soap_n),
                        coeff_map=st.session_state['coeff_map']
                    )
                    results.append(desc)
                    atom_tables.append(atom_df)
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
                finally:
                    try: os.remove(p)
                    except: pass
                progress.progress(int((i+1)/n*100))

            if results:
                df_desc = pd.DataFrame(results)
                df_atoms = pd.concat(atom_tables, ignore_index=True) if atom_tables else pd.DataFrame()
                st.success("Analysis finished.")
                st.subheader("Descriptor results (preview)")
                st.dataframe(df_desc.head(15), use_container_width=True)
                coldl1, coldl2 = st.columns(2)
                with coldl1:
                    st.download_button("Download descriptors CSV", data=df_desc.to_csv(index=False).encode('utf-8'),
                                       file_name="QSPR_SOAP_MBTR_descriptors.csv", mime='text/csv')
                with coldl2:
                    # Excel export, both sheets
                    towrite = io.BytesIO()
                    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                        df_desc.to_excel(writer, sheet_name='descriptors', index=False)
                        df_atoms.to_excel(writer, sheet_name='per_atom', index=False)
                    towrite.seek(0)
                    st.download_button("Download Excel workbook", data=towrite, file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.subheader("Basic plots")
                if "Binding_Energy_eV" in df_desc.columns:
                    st.bar_chart(df_desc["Binding_Energy_eV"])

                # create zip of outputs
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    zf.writestr("descriptors.csv", df_desc.to_csv(index=False))
                    zf.writestr("per_atom.csv", df_atoms.to_csv(index=False))
                zip_buf.seek(0)
                st.download_button("Download ZIP (CSV files)", data=zip_buf, file_name="results_csvs.zip", mime="application/zip")

# ---------- Presets tab ----------
with tab_presets:
    st.header("Presets: save/load pair/coeff configurations")
    name = st.text_input("Preset name", value="my_preset")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Save preset"):
            save_preset(name)
            st.success(f"Saved preset '{name}'.")
    with c2:
        presets = list(st.session_state['presets'].keys())
        if presets:
            sel = st.selectbox("Load preset", options=presets)
            if st.button("Load selected preset"):
                load_preset(sel)
                st.success(f"Loaded preset '{sel}'.")
        else:
            st.info("No presets saved yet.")
    st.markdown("### Export / Import presets (JSON)")
    colx, coly = st.columns([1,1])
    with colx:
        if st.button("Export presets as JSON"):
            st.download_button("Download presets JSON", data=json.dumps(st.session_state['presets'], indent=2),
                               file_name="qspr_presets.json", mime="application/json")
    with coly:
        up = st.file_uploader("Import presets JSON", type=["json"])
        if up:
            try:
                p = json.loads(up.getvalue().decode())
                st.session_state['presets'].update(p)
                st.success("Imported presets.")
            except Exception as e:
                st.error(f"Import failed: {e}")

# ---------- update viewer preview when files present ----------
def update_preview():
    uploaded_list = st.session_state.get('uploaded_files', [])
    if uploaded_list:
        f = uploaded_list[0]
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                tmp.write(f.getbuffer()); tmp.flush(); path = tmp.name
            s0 = Structure.from_file(path)
            html = make_py3dmol_view(s0,
                                     show_bonds=st.session_state.get('v_show_bonds', True),
                                     bond_min=st.session_state.get('v_bond_min', 0.0),
                                     bond_max=st.session_state.get('v_bond_max', 3.0),
                                     highlight_atom_index=st.session_state.get('v_atom_index', 0),
                                     show_poly=st.session_state.get('v_show_poly', False))
            # Use streamlit components to render the returned HTML
            viewer_placeholder = st.session_state.get('_viewer_placeholder')
            if viewer_placeholder is None:
                # find placeholder if not stored previously (first run)
                # we created a `viewer_placeholder` in the Viewer tab's scope above; use components.html directly
                components.html(html, height=520, scrolling=True)
            else:
                components.html(html, height=520, scrolling=True)
        except Exception as e:
            st.error(f"Viewer failed: {e}")
        finally:
            try: os.remove(path)
            except: pass

# run preview (single call each execution)
update_preview()

st.sidebar.markdown("---")
st.sidebar.write("Scientific Workstation • Multi-file • Per-pair controls • Editable coefficients")
st.sidebar.write("Built for scientist-grade control. Save presets to reuse configurations.")
