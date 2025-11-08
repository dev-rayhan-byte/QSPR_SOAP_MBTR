# app.py
"""
Patched Ultimate Streamlit Scientific Workstation
Fixes:
- Removed problematic st.experimental_rerun()
- Moved uploader to top-level sidebar so all tabs share same uploaded files
- Reworked control flow and UX (forms, expanders, safer state updates)
- Better error handling and messaging
Author: Patched (2025)
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
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from ase import Atoms
from dscribe.descriptors import SOAP, MBTR
import py3Dmol

# ---------------- page config ----------------
st.set_page_config(page_title="Scientific Workstation — Ultimate (patched)", layout="wide")
st.markdown("<style>footer {visibility: hidden;} .small-muted { color: #9aa0a6; font-size:12px }</style>", unsafe_allow_html=True)

# ---------------- defaults ----------------
BOND_LENGTH_FALLBACK = 3.2
DEFAULT_DBSCAN_EPS = 0.25
DEFAULT_DBSCAN_MIN_SAMPLES = 1
DEFAULT_SOAP_N = 50

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

ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050",
    "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Pt": "#D0D0E0", "Ru": "#248F8F",
    "Pd": "#A0C0D0", "Co": "#F090A0", "Fe": "#E06633", "Cu": "#C88033", "Ni": "#50D050",
    "Au": "#FFD123", "Ag": "#C0C0C0", "Zn": "#7D80B0", "Sn": "#668080", "Pb": "#575961",
}

cnn = CrystalNN(weighted_cn=False)

# ---------------- session defaults ----------------
if 'pair_ranges' not in st.session_state:
    st.session_state['pair_ranges'] = {}
if 'coeff_map' not in st.session_state:
    st.session_state['coeff_map'] = dict(DEFAULT_COEFFICIENTS)
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}

# ---------------- helper functions ----------------
def adaptive_bond_cutoff(e1, e2, scale=1.25, fallback=BOND_LENGTH_FALLBACK):
    try:
        r1, r2 = Element(e1).covalent_radius, Element(e2).covalent_radius
        if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in (r1, r2)):
            return fallback
        return scale * (r1 + r2)
    except Exception:
        return fallback

def detect_layers(z_coords, eps=DEFAULT_DBSCAN_EPS, min_samples=DEFAULT_DBSCAN_MIN_SAMPLES):
    z = np.array(z_coords).reshape(-1,1)
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
            if j == idx: continue
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
        xkey = ck.replace('b_', '')
        if xkey in x_counts:
            N = max(x_counts[xkey], 0)
            sigma_x = math.sqrt(N)
            sigma2 += (b * sigma_x)**2
    return math.sqrt(sigma2)

def make_py3dmol_view(structure, show_bonds=True, bond_min=0.0, bond_max=3.0,
                      highlight_atom_index=None, show_poly=False, poly_neighbors=None,
                      atom_size=0.45):
    coords = np.array([s.coords for s in structure.sites])
    elements = [str(s.specie) for s in structure.sites]
    view = py3Dmol.view(width=820, height=520)
    # spheres
    for i, (el, pos) in enumerate(zip(elements, coords)):
        color = ELEMENT_COLORS.get(el, ELEMENT_COLORS.get(el.capitalize(), '#AAAAAA'))
        view.addSphere({'center': {'x':float(pos[0]), 'y':float(pos[1]), 'z':float(pos[2])},
                        'radius': atom_size, 'color': color, 'opacity':1.0})
    # bonds
    if show_bonds:
        n = len(elements)
        for i in range(n):
            for j in range(i+1, n):
                d = structure.get_distance(i, j)
                if bond_min <= d <= bond_max:
                    view.addCylinder({'start': {'x': float(coords[i,0]), 'y': float(coords[i,1]), 'z': float(coords[i,2])},
                                      'end':   {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                                      'radius': atom_size*0.15, 'color': '#BBBBBB', 'opacity':1.0})
    # highlight
    if highlight_atom_index is not None:
        pos = coords[int(highlight_atom_index)]
        view.addSphere({'center': {'x':float(pos[0]), 'y':float(pos[1]), 'z':float(pos[2])},
                        'radius': atom_size*1.3, 'color':'red', 'opacity':1.0})
        if show_poly and poly_neighbors:
            for j in poly_neighbors:
                view.addCylinder({'start': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                                  'end': {'x': float(coords[j,0]), 'y': float(coords[j,1]), 'z': float(coords[j,2])},
                                  'radius': atom_size*0.08, 'color':'orange', 'opacity':1.0})
    view.setBackgroundColor('#ffffff')
    view.zoomTo()
    return view.show()

def analyze_structure(structure, filename,
                      per_pair_ranges=None,
                      normalize_by="bonds", dbscan_eps=DEFAULT_DBSCAN_EPS, dbscan_min_samples=DEFAULT_DBSCAN_MIN_SAMPLES,
                      compute_soap=True, compute_mbtr=True, soap_first_n=DEFAULT_SOAP_N,
                      coeff_map=None):
    n_atoms = len(structure.sites)
    elements = [str(site.specie) for site in structure.sites]
    coords = np.array([site.coords for site in structure])
    z_coords = coords[:, 2]
    layer_indices = detect_layers(z_coords, eps=dbscan_eps, min_samples=dbscan_min_samples)
    layer_means = {l: np.mean(z_coords[layer_indices == l]) for l in np.unique(layer_indices)}
    label_map = {old: new for new, old in enumerate(sorted(layer_means, key=lambda L: layer_means[L]))}
    layer_indices = np.array([label_map[l] for l in layer_indices])
    n_layers = int(layer_indices.max()) + 1
    surface_mask = (layer_indices == layer_indices.min()) | (layer_indices == layer_indices.max())

    atom_rows = [{
        "filename": filename,
        "atom_index": i,
        "element": elements[i],
        "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
        "layer_index": int(layer_indices[i]),
        "surface_flag": bool(surface_mask[i])
    } for i, c in enumerate(coords)]
    atom_df = pd.DataFrame(atom_rows)

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
            if per_pair_ranges and pair_key in per_pair_ranges:
                min_r, max_r = per_pair_ranges[pair_key]
            else:
                min_r = 0.0
                max_r = adaptive_bond_cutoff(e1, e2)*1.12
            if not (min_r <= d <= max_r): continue
            if layer_indices[i] == layer_indices[j]:
                bond_counts_intra[pair_key] += 1
            else:
                bond_counts_inter[pair_key] += 1

    surface_counts = defaultdict(int)
    for i, surf in enumerate(surface_mask):
        if surf: surface_counts[f"{elements[i]}-S"] += 1

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

    x_norm = {k: v for k, v in x.items()}
    if normalize_by == "atoms" and n_atoms > 0:
        x_norm = {k: v / n_atoms for k, v in x.items()}
    elif normalize_by == "bonds" and total_bonds > 0:
        x_norm = {k: v / total_bonds for k, v in x.items()}

    if coeff_map is None:
        coeff_map = st.session_state.get('coeff_map', dict(DEFAULT_COEFFICIENTS))
    Y = coeff_map.get('b0', 0.0) + sum(coeff_map.get('b_' + k, 0.0) * x_norm.get(k, 0.0) for k in x_norm)
    sigmaY = propagate_uncertainty_linear(coeff_map, x)
    desc['Binding_Energy_eV'] = float(Y)
    desc['Binding_Energy_sigma_eV_approx'] = float(sigmaY)

    ase_atoms = structure_to_ase(structure)
    species = list(set(ase_atoms.get_chemical_symbols()))

    if compute_soap:
        try:
            soap = SOAP(species=species, rcut=5.0, nmax=6, lmax=4, sigma=0.5, periodic=True)
            soap_vec = soap.create(ase_atoms)
            soap_mean = np.mean(soap_vec, axis=0)
            for i, val in enumerate(soap_mean[:soap_first_n]):
                desc[f"SOAP_{i}"] = float(val)
        except Exception as e:
            st.warning(f"SOAP failed for {filename}: {e}")

    if compute_mbtr:
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

# ---------------- Sidebar: uploader + global quick controls ----------------
st.sidebar.title("Workstation • Upload & Quick Settings")
uploaded = st.sidebar.file_uploader("Upload CIF files (multiple allowed)", type=["cif"], accept_multiple_files=True)
st.sidebar.markdown("**Quick viewer settings**")
st.sidebar.checkbox("Show bonds (viewer)", value=True, key="v_show_bonds")
st.sidebar.number_input("Viewer min bond (Å)", value=0.0, step=0.01, key="v_bond_min")
st.sidebar.number_input("Viewer max bond (Å)", value=3.0, step=0.01, key="v_bond_max")
st.sidebar.checkbox("Show polyhedra (viewer)", value=False, key="v_show_poly")
st.sidebar.number_input("Highlight atom index (viewer)", min_value=0, value=0, key="v_atom_index")

# ---------------- Tabs and UI ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Viewer", "Control Board", "Descriptors & Run", "Presets"])

# ----- Viewer tab -----
with tab1:
    st.header("3D Viewer (preview of first uploaded file)")
    viewer_col, info_col = st.columns([2,1])
    with viewer_col:
        view_placeholder = st.empty()
        if uploaded and len(uploaded)>0:
            f = uploaded[0]
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                    tmp.write(f.getbuffer()); tmp.flush(); path = tmp.name
                s0 = Structure.from_file(path)
                html = make_py3dmol_view(s0,
                                         show_bonds=st.session_state['v_show_bonds'],
                                         bond_min=st.session_state['v_bond_min'],
                                         bond_max=st.session_state['v_bond_max'],
                                         highlight_atom_index=st.session_state['v_atom_index'],
                                         show_poly=st.session_state['v_show_poly'])
                view_placeholder.components.v1.html(html, height=520, scrolling=True)
            except Exception as e:
                view_placeholder.error(f"Viewer failed: {e}")
            finally:
                try: os.remove(path)
                except: pass
        else:
            view_placeholder.info("Upload one or more CIF files in the sidebar to preview.")

    with info_col:
        st.subheader("Quick controls")
        st.write("Use the sidebar to change viewer bond cutoffs and highlight atom.")
        if uploaded and len(uploaded)>0:
            st.write(f"Preview file: **{uploaded[0].name}**")
            st.write(f"Atom count (preview): **{len(Structure.from_file(tempfile.NamedTemporaryFile(delete=False, suffix='.cif').name)) if False else '—'}**")
        st.markdown("")

# ----- Control Board -----
with tab2:
    st.header("Control Board — File list and Pair controls")
    left, right = st.columns([2,1])
    with left:
        st.subheader("Uploaded files")
        if uploaded and len(uploaded)>0:
            for i, f in enumerate(uploaded):
                st.write(f"{i+1}. {f.name} — {round(len(f.getbuffer())/1024,1)} KB")
        else:
            st.info("No CIFs uploaded yet.")

        st.markdown("---")
        st.subheader("Auto-detect element pairs")
        st.write("Detect all element pairs across uploaded files (for per-pair bond controls).")
        if st.button("Detect pairs from uploaded files"):
            if not uploaded:
                st.warning("Upload CIF files first.")
            else:
                all_elements = set()
                for f in uploaded:
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
                # init missing
                for pair in pairs:
                    if pair not in st.session_state['pair_ranges']:
                        parts = pair.split('-')
                        est = adaptive_bond_cutoff(parts[0], parts[1])
                        st.session_state['pair_ranges'][pair] = (0.0, round(est*1.12, 3))
                st.success(f"Detected {len(pairs)} pairs. Edit them below (in the expander).")

        st.markdown("Per-pair bond controls (editable):")
        with st.expander("Show / Edit per-pair min/max ranges", expanded=True):
            pr = st.session_state['pair_ranges']
            if pr:
                df_pairs = pd.DataFrame([{"pair": k, "min": float(v[0]), "max": float(v[1])} for k, v in sorted(pr.items())])
                edited = st.experimental_data_editor(df_pairs, num_rows="dynamic")
                # validate and write back
                new_map = {}
                for r in edited.to_dict(orient='records'):
                    pair = r.get('pair')
                    try:
                        mn = float(r.get('min', 0.0))
                        mx = float(r.get('max', 0.0))
                    except Exception:
                        mn, mx = 0.0, st.session_state['pair_ranges'].get(pair, (0.0, BOND_LENGTH_FALLBACK))[1]
                    if mx < mn:
                        mx = mn
                    new_map[pair] = (mn, mx)
                st.session_state['pair_ranges'] = new_map
            else:
                st.info("No pairs detected. Use 'Detect pairs from uploaded files'.")

    with right:
        st.subheader("Global analysis quick settings")
        normalize_by = st.selectbox("Normalize descriptors by", options=["bonds", "atoms", "none"], index=0)
        compute_soap = st.checkbox("Compute SOAP (slow)", value=True)
        compute_mbtr = st.checkbox("Compute MBTR (slow)", value=True)
        soap_n = st.number_input("Keep first N SOAP features", min_value=10, max_value=1000, value=50)
        dbscan_eps = st.number_input("DBSCAN eps (layer detect)", value=0.25, step=0.01)
        dbscan_min = st.number_input("DBSCAN min_samples", value=1, step=1, min_value=1)
        st.markdown("Coefficient presets:")
        colp1, colp2 = st.columns([1,1])
        with colp1:
            if st.button("Load default coefficients"):
                st.session_state['coeff_map'] = dict(DEFAULT_COEFFICIENTS)
                st.success("Loaded default coefficients.")
        with colp2:
            if st.button("Clear pair ranges"):
                st.session_state['pair_ranges'] = {}
                st.info("Cleared per-pair ranges.")

# ----- Descriptors & Run -----
with tab3:
    st.header("Descriptors, Coefficients & Run")
    st.subheader("Editable coefficient table")
    coeff_df = pd.DataFrame([{"name": k, "value": float(v)} for k, v in st.session_state['coeff_map'].items()])
    edited_coeff = st.experimental_data_editor(coeff_df, num_rows="dynamic")
    # write back safely
    new_coeff = {}
    for r in edited_coeff.to_dict(orient='records'):
        name = str(r.get('name'))
        try:
            val = float(r.get('value', 0.0))
        except Exception:
            val = 0.0
        new_coeff[name] = val
    st.session_state['coeff_map'] = new_coeff

    st.markdown("---")
    run_col1, run_col2 = st.columns([1,1])
    with run_col1:
        if st.button("Run analysis on uploaded CIFs"):
            if not uploaded:
                st.warning("Upload CIFs first (sidebar).")
            else:
                n = len(uploaded)
                prog = st.progress(0)
                results = []
                atom_tables = []
                for i, f in enumerate(uploaded):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
                            tmp.write(f.getbuffer()); tmp.flush(); p = tmp.name
                        s = Structure.from_file(p)
                        desc, atom_df = analyze_structure(
                            s, f.name,
                            per_pair_ranges=st.session_state.get('pair_ranges', {}),
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
                    prog.progress(int((i+1)/n*100))
                if results:
                    df_desc = pd.DataFrame(results)
                    df_atoms = pd.concat(atom_tables, ignore_index=True)
                    st.success("Analysis complete.")
                    st.dataframe(df_desc.head(15))
                    cold1, cold2 = st.columns(2)
                    with cold1:
                        st.download_button("Download descriptors CSV", data=df_desc.to_csv(index=False).encode('utf-8'),
                                           file_name="QSPR_SOAP_MBTR_descriptors.csv", mime='text/csv')
                    with cold2:
                        towrite = io.BytesIO()
                        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                            df_desc.to_excel(writer, sheet_name='descriptors', index=False)
                            df_atoms.to_excel(writer, sheet_name='per_atom', index=False)
                        towrite.seek(0)
                        st.download_button("Download Excel workbook", data=towrite, file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("No results — check logs or input files.")

    with run_col2:
        if st.button("Export current settings (pairs + coeffs) as JSON"):
            payload = {"pair_ranges": st.session_state.get('pair_ranges', {}), "coeff_map": st.session_state.get('coeff_map', {})}
            st.download_button("Download settings JSON", data=json.dumps(payload, indent=2), file_name="settings.json", mime="application/json")

# ----- Presets -----
with tab4:
    st.header("Presets (save/load)")
    pname = st.text_input("Preset name", value="my_preset")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Save preset"):
            st.session_state['presets'][pname] = {"pair_ranges": st.session_state.get('pair_ranges', {}), "coeff_map": st.session_state.get('coeff_map', {})}
            st.success(f"Preset '{pname}' saved.")
    with c2:
        presets = list(st.session_state['presets'].keys())
        if presets:
            sel = st.selectbox("Select preset", options=presets)
            if st.button("Load selected"):
                p = st.session_state['presets'][sel]
                st.session_state['pair_ranges'] = p.get('pair_ranges', {})
                st.session_state['coeff_map'] = p.get('coeff_map', {})
                st.success(f"Loaded preset '{sel}'.")
        else:
            st.info("No presets saved yet.")

    st.markdown("---")
    if st.button("Export all presets JSON"):
        st.download_button("Download presets JSON", data=json.dumps(st.session_state['presets'], indent=2), file_name="presets.json", mime="application/json")
    up = st.file_uploader("Import presets JSON", type=["json"])
    if up:
        try:
            loaded = json.loads(up.getvalue().decode())
            st.session_state['presets'].update(loaded)
            st.success("Imported presets successfully.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# End of file
